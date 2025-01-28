import contextlib
import logging
import traceback

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .config import AppState
from .model_loader import ModelLoader
from .models import Block, StructureRequest
from .services import StructureGenerator

logger = logging.getLogger(__name__)


class SchematicGeneratorApp(FastAPI):
    state: AppState


@contextlib.asynccontextmanager
async def lifespan(app: SchematicGeneratorApp):
    logger.info("Loading model...")
    model_loader = ModelLoader()
    app.state.model = model_loader.load_model(
        app.state.checkpoint_path,
        app.state.model_path,
        app.state.model_id,
        app.state.model_revision,
        app.state.device,
    )
    logger.info("Model loaded successfully")
    yield


app = SchematicGeneratorApp(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    logger.error("Validation error", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Invalid request parameters"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(_: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "message": str(exc)},
    )


@app.post("/complete-structure/")
async def complete_structure(input: StructureRequest, request: Request):
    logger.info("Received structure generation request")
    try:
        generator = StructureGenerator(app.state.model)
        input_tensor = generator.prepare_input_tensor(input.structure)

        async def generate():
            for block_data in generator.generate_structure(
                input_tensor,
                input.temperature,
                input.start_radius,
                input.max_iterations,
                input.max_blocks,
                input.air_probability_iteration_scaling,
            ):
                response = Block(**block_data)
                yield response.model_dump_json() + "\n"

                # Check for client disconnection after each block
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping generation")
                    break

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Error during structure generation: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=str(e))
