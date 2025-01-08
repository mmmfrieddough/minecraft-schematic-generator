import logging

import torch

from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

MODEL_ID = "mmmfrieddough/minecraft-structure-generator"

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model = None

    def load_model(self, mode: str, checkpoint_path: str = None):
        """Load model based on specified mode and path"""
        if mode == "local":
            if not checkpoint_path:
                raise ValueError("checkpoint_path must be specified in local mode")
            logger.info(f"Loading model from local checkpoint: {checkpoint_path}")
            self.model = (
                LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
                    checkpoint_path
                )
            )

        elif mode == "production":
            # First load the base model from HF
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(f"Loading model from Hugging Face: {MODEL_ID}")
            base_model = TransformerMinecraftStructureGenerator.from_pretrained(
                MODEL_ID
            ).to(device)

            # Create Lightning module with same parameters as base model
            self.model = LightningTransformerMinecraftStructureGenerator(
                num_classes=base_model.num_classes,
                max_sequence_length=base_model.max_sequence_length,
                embedding_dropout=base_model.embedding_dropout.p,
                model_dim=base_model.embedding.embedding_dim,
                num_heads=base_model.decoder.layers[0].self_attn.num_heads,
                num_layers=len(base_model.decoder.layers),
                decoder_dropout=base_model.decoder.layers[0].dropout.p,
                max_learning_rate=1e-4,  # Default value, not used during inference
                warmup_steps=100,  # Default value, not used during inference
            ).to(device)

            # Copy the weights from base model to Lightning model
            self.model.model.load_state_dict(base_model.state_dict())

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.model.eval()
        return self.model
