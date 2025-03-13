import contextlib
import logging
import traceback

import aiohttp
import semver
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    DictBlockTokenMapper,
)
from minecraft_schematic_generator.version import GITHUB_REPO, __version__

from .config import AppState
from .model_loader import ModelLoader
from .models import Block, StructureRequest
from .services import StructureGenerator

# Set PyMCTranslate logging level before importing it
logging.getLogger("PyMCTranslate").setLevel(logging.WARNING)
import PyMCTranslate  # noqa: E402

logger = logging.getLogger(__name__)


class SchematicGeneratorApp(FastAPI):
    state: AppState


async def check_latest_version(app: SchematicGeneratorApp):
    """Check GitHub for the latest release version."""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latest_version = data["tag_name"].lstrip("v")
                    current_version = __version__

                    if semver.compare(latest_version, current_version) > 0:
                        logger.warning(
                            f"A new version {latest_version} is available! "
                            f"You are currently running version {current_version}. "
                            f"Visit https://github.com/{GITHUB_REPO}/releases/latest to update."
                        )
                    else:
                        logger.info(f"Running latest version {current_version}")
                else:
                    raise Exception(f"Coudn't fetch latest version: {response.status}")
    except Exception as e:
        logger.warning(f"Failed to check for updates: {str(e)}")


@contextlib.asynccontextmanager
async def lifespan(app: SchematicGeneratorApp):
    logger.info(f"Starting Minecraft Schematic Generator v{__version__}")

    # Check for updates
    await check_latest_version(app)

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

    app.state.block_token_mapper = DictBlockTokenMapper(
        app.state.model.block_str_mapping
    )
    app.state.translation_manager = PyMCTranslate.new_translation_manager()
    minecraft_version = app.state.translation_manager.version_numbers("java")[-1]
    minecraft_version_str = ".".join(str(x) for x in minecraft_version)
    logger.info(
        f"Translation manager loaded for up to Minecraft {minecraft_version_str}"
    )

    yield


app = SchematicGeneratorApp(lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
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
        version_translator = app.state.translation_manager.get_version(
            input.platform, input.version_number
        )
        block_token_mapper = BlockTokenConverter(
            app.state.block_token_mapper, version_translator
        )
        generator = StructureGenerator(app.state.model, block_token_mapper)
        input_tensor = generator.prepare_input_tensor(input.palette, input.structure)

        async def generate():
            for block_data in generator.generate_structure(
                input_tensor,
                input.temperature,
                input.start_radius,
                input.max_iterations,
                input.max_blocks,
                input.max_alternatives,
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
