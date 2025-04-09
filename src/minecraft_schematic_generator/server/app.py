import contextlib
import importlib
import json
import logging

import aiohttp
import semver
import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from minecraft_schematic_generator.constants import GITHUB_REPO
from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    DictBlockTokenMapper,
)

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
                    current_version = importlib.metadata.version(
                        "minecraft_schematic_generator"
                    )
                    logger.debug(f"Current version: {current_version}")

                    if semver.compare(latest_version, current_version) > 0:
                        logger.warning(
                            f"A new version {latest_version} is available! "
                            f"You are currently running version {current_version}. "
                            f"Visit https://github.com/{GITHUB_REPO}/releases/latest to update."
                        )
                    elif semver.compare(latest_version, current_version) < 0:
                        logger.info(
                            f"You are running a pre-release version {current_version}. "
                            f"The latest stable version is {latest_version}."
                        )
                    else:
                        logger.info("Running latest version")
                else:
                    raise Exception(f"Coudn't fetch latest version: {response.status}")
    except Exception as e:
        logger.warning(f"Failed to check for updates: {str(e)}")


async def load_default_model(app: SchematicGeneratorApp):
    """Load the default model."""
    logger.info("Loading default model...")
    app.state.model = ModelLoader.load_model(
        app.state.model_type,
        app.state.checkpoint_path,
        app.state.model_path,
        app.state.model_id,
        app.state.model_version,
        app.state.device,
    )
    logger.info("Default model loaded successfully")


@contextlib.asynccontextmanager
async def lifespan(app: SchematicGeneratorApp):
    logger.info(
        f"Starting Minecraft Schematic Generator v{importlib.metadata.version('minecraft_schematic_generator')}"
    )

    # Check for updates
    await check_latest_version(app)

    await load_default_model(app)
    app.state.current_model_type = "default"
    app.state.current_model_version = "default"
    app.state.current_inference_device = "default"

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
    all_errors = exc.errors()

    MAX_ERRORS = 5

    error_messages = []
    for error in all_errors[:MAX_ERRORS]:
        msg = error.get("msg")
        type = error.get("type")
        loc = ".".join(str(x) for x in error.get("loc"))
        error_messages.append(f"{loc}: {type} - {msg}")

    # Add message about hidden errors if there are more
    message = (
        "Invalid request parameters. Make sure you are using the latest mod version."
    )
    if len(all_errors) > MAX_ERRORS:
        message += f" (showing {MAX_ERRORS} of {len(all_errors)} errors)"

    logger.error(f"Validation error: {message}. Errors: {error_messages}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": error_messages, "message": message},
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
        # Model changing
        if not input.model_type or input.model_type == "default":
            if app.state.current_model_type != "default":
                await load_default_model(app)
                app.state.current_model_type = "default"
                app.state.current_model_version = "default"
        elif (
            input.model_type != app.state.current_model_type
            or input.model_version != app.state.current_model_version
        ):
            logger.info(
                f"Model changed from {app.state.current_model_type}:{app.state.current_model_version} to {input.model_type}:{input.model_version}"
            )
            app.state.model = ModelLoader.load_model(
                input.model_type,
                None,
                None,
                None,
                input.model_version,
                app.state.device,
            )
            app.state.current_model_type = input.model_type
            app.state.current_model_version = input.model_version
            logger.info("Model loaded successfully")

        # Device changing
        if not input.inference_device or input.inference_device == "default":
            if app.state.current_inference_device != "default":
                logger.info(f"Loading model on default device: {app.state.device}")
                device = ModelLoader.configure_device(app.state.device)
                app.state.model.to(device)
                app.state.current_inference_device = "default"
        elif input.inference_device != app.state.current_inference_device:
            logger.info(
                f"Device changed from {app.state.current_inference_device} to {input.inference_device}"
            )
            device = ModelLoader.configure_device(input.inference_device)
            app.state.model.to(device)
            app.state.current_inference_device = input.inference_device
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
        input_tensor = generator.prepare_input_tensor(
            input.ignore_replaceable_blocks, input.palette, input.structure
        )
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
