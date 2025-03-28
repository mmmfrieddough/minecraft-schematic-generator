import contextlib
import json
import logging

import aiohttp
import semver
import torch
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
from .models import StructureRequest, StructureResponse
from .services import StructureGenerator

# Set PyMCTranslate logging level before importing it
logging.getLogger("PyMCTranslate").setLevel(logging.WARNING)
import PyMCTranslate  # noqa: E402

logger = logging.getLogger(__name__)


class SchematicGeneratorApp(FastAPI):
    state: AppState


async def check_latest_version(app: SchematicGeneratorApp):
    """Check GitHub for the latest release version."""
    logger.debug("Checking for updates...")
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    latest_version = data["tag_name"].lstrip("v")
                    logger.debug(f"Latest version: {latest_version}")
                    current_version = __version__
                    logger.debug(f"Current version: {current_version}")

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
    app.state.model = ModelLoader.load_model(
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
        if input.model_type and input.model_type != "default":
            model_id = f"mmmfrieddough/minecraft-schematic-generator-{input.model_type}"
            if (
                app.state.model_id != model_id
                or app.state.model_revision != input.model_version
            ):
                logger.info(f"Loading model for model type: {input.model_type}")
                app.state.model = ModelLoader.load_model(
                    None,
                    None,
                    model_id,
                    input.model_version,
                    app.state.device,
                )
                logger.info("Model loaded successfully")
                app.state.model_id = model_id
                app.state.model_revision = input.model_version
        if input.inference_device and input.inference_device != app.state.device:
            logger.info(f"Changing device to {input.inference_device}")
            app.state.device = input.inference_device
            device = ModelLoader.configure_device(input.inference_device)
            app.state.model.to(device)
            logger.info("Device changed successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}",
        )

    try:
        version_translator = app.state.translation_manager.get_version(
            input.platform, input.version_number
        )
        block_token_mapper = BlockTokenConverter(
            app.state.block_token_mapper, version_translator
        )
        generator = StructureGenerator(app.state.model, block_token_mapper)
        input_tensor = generator.prepare_input_tensor(input.palette, input.structure)
    except Exception as e:
        logger.error(f"Error during input preparation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during input preparation: {str(e)}",
        )

    async def generate():
        try:
            for block_data in generator.generate_structure(
                input_tensor,
                input.temperature,
                input.start_radius,
                input.max_iterations,
                input.max_blocks,
                input.max_alternatives,
                input.min_alternative_probability,
            ):
                response = StructureResponse(type="block", **block_data)
                yield response.model_dump_json() + "\n"

                # Check for client disconnection after each block
                if await request.is_disconnected():
                    logger.info("Client disconnected, stopping generation")
                    break

            complete_json = {"type": "complete"}
            yield json.dumps(complete_json) + "\n"
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory error", exc_info=True)
            error_json = {
                "type": "error",
                "detail": "The model ran out of memory. Try using a smaller input size or smaller model.",
            }
            yield json.dumps(error_json) + "\n"
        except Exception as e:
            logger.error(f"Error during structure generation: {str(e)}", exc_info=True)
            error_json = {
                "type": "error",
                "detail": f"Error during structure generation: {str(e)}",
            }
            yield json.dumps(error_json) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
