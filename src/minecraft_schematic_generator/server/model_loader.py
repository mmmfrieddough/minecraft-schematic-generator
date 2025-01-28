import logging

import torch

from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model: TransformerMinecraftStructureGenerator = None

    def load_model(
        self,
        checkpoint_path: str,
        model_path: str,
        model_id: str,
        model_revision: str,
        device_type: str,
    ) -> TransformerMinecraftStructureGenerator:
        """Load model based on specified mode and path"""

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

        if checkpoint_path:
            logger.info(f"Loading model from local checkpoint: {checkpoint_path}")
            try:
                lightning_module = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
                    checkpoint_path
                )
                self.model = lightning_module.model
            except Exception as e:
                logger.error(f"Unable to load model from checkpoint: {e}")
                raise e
        elif model_path:
            try:
                logger.info(f"Loading model from local path: {model_path}")
                self.model = TransformerMinecraftStructureGenerator.from_pretrained(
                    model_path
                )
            except Exception as e:
                logger.error(f"Unable to load model from local path: {e}")
                raise e
        elif model_id:
            try:
                logger.info(
                    f"Loading model from Hugging Face: {model_id}, revision: {model_revision}"
                )
                self.model = TransformerMinecraftStructureGenerator.from_pretrained(
                    model_id, revision=model_revision
                )
            except Exception as e:
                logger.error(f"Unable to load model from Hugging Face: {e}")
                raise e
        else:
            raise ValueError("No model specified")

        self.model.to(device)
        self.model.eval()
        return self.model
