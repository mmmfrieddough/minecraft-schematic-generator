import contextlib
import logging
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .model_loader import ModelLoader
from .models import Block, StructureRequest
from .services import StructureGenerator

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Loading model...")
    model_loader = ModelLoader()
    app.state.model = model_loader.load_model(app.state.mode, app.state.checkpoint_path)
    logger.info("Model loaded successfully")
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/complete-structure/")
async def complete_structure(input: StructureRequest, request: Request):
    try:
        logger.info("Starting structure generation")
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
