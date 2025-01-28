import logging

import torch

from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

MODEL_ID = "mmmfrieddough/minecraft-schematic-generator"

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model: TransformerMinecraftStructureGenerator = None

    def load_model(self, mode: str, checkpoint_path: str = None):
        """Load model based on specified mode and path"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        if mode == "local":
            if not checkpoint_path:
                raise ValueError("checkpoint_path must be specified in local mode")
            logger.info(f"Loading model from local checkpoint: {checkpoint_path}")
            lightning_module = (
                LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
                    checkpoint_path
                )
            )
            self.model = lightning_module.model

        elif mode == "production":
            logger.info(f"Loading model from Hugging Face: {MODEL_ID}")
            self.model = TransformerMinecraftStructureGenerator.from_pretrained(
                MODEL_ID
            )

        self.model.to(device)
        self.model.eval()
        return self.model
