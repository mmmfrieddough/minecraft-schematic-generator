import logging

import torch

from minecraft_schematic_generator.constants import MODEL_TYPE_DIAMOND, MODEL_TYPE_IRON
from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Custom exception for model loading errors"""

    pass


class ModelLoader:
    @staticmethod
    def configure_device(device_type: str) -> torch.device:
        if device_type == "auto":
            logger.info("Auto selecting PyTorch device")
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_type)

        logger.info(f"Using device: {device}")

        if device == torch.device("cpu"):
            logger.warning("Performance may be slow when using CPU")

        return device

    @staticmethod
    def load_model(
        model_type: str | None,
        checkpoint_path: str | None,
        model_path: str | None,
        model_id: str | None,
        model_revision: str | None,
        device_type: str,
    ) -> TransformerMinecraftStructureGenerator:
        """Load model based on specified mode and path"""

        device = ModelLoader.configure_device(device_type)
        model_revision = None if model_revision == "" else model_revision

        if not model_type and not checkpoint_path and not model_path and not model_id:
            model_type = MODEL_TYPE_IRON if device_type == "cpu" else MODEL_TYPE_DIAMOND
            logger.info(
                f"No model specified, auto selected default model: {model_type}"
            )

        if model_type:
            model_id = f"mmmfrieddough/minecraft-schematic-generator-{model_type}"

        try:
            if checkpoint_path:
                logger.info(f"Loading model from local checkpoint: {checkpoint_path}")
                lightning_module = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
                    checkpoint_path, map_location=device
                )
                model = lightning_module.model
            elif model_path:
                logger.info(f"Loading model from local path: {model_path}")
                model = TransformerMinecraftStructureGenerator.from_pretrained(
                    model_path, map_location=str(device)
                )
            elif model_id:
                logger.info(
                    f"Loading model from Hugging Face: {model_id}, revision: {model_revision}"
                )
                model = TransformerMinecraftStructureGenerator.from_pretrained(
                    model_id, revision=model_revision, map_location=str(device)
                )
            else:
                raise ModelLoadError("No model specified")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise ModelLoadError(f"Model loading failed: {str(e)}") from e

        model.to(device)
        model.eval()
        return model
