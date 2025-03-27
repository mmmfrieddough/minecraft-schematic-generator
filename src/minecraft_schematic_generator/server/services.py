import logging

import torch

from minecraft_schematic_generator.constants import (
    AIR_BLOCK_ID,
    MASK_BLOCK_ID,
)
from minecraft_schematic_generator.converter import BlockTokenConverter
from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator


class StructureGenerator:
    def __init__(
        self,
        model: TransformerMinecraftStructureGenerator,
        block_token_mapper: BlockTokenConverter,
    ):
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.block_token_mapper = block_token_mapper

    def convert_block_to_token(self, block_str: str) -> int:
        try:
            return self.block_token_mapper.versioned_str_to_token(block_str)
        except KeyError:
            self.logger.warning(
                f"Block {block_str} not found in mapping. Returning unused token."
            )
            return self.block_token_mapper.get_unused_token()

    def prepare_input_tensor(
        self, palette: dict[int, str], structure: list[list[list[int]]]
    ) -> torch.Tensor:
        # Convert palette to tokens
        for k, v in palette.items():
            palette[k] = self.convert_block_to_token(v)

        # Create lookup table for palette
        max_id = max(palette.keys())
        lookup_table = torch.zeros(max_id + 1, dtype=torch.long)
        for block_id, token_id in palette.items():
            lookup_table[block_id] = token_id

        # Convert structure to tensor
        input_tensor = torch.tensor(structure)

        # Replace palette values with tokens
        input_tensor = lookup_table[input_tensor]

        # Mask air blocks
        input_tensor[input_tensor == AIR_BLOCK_ID] = MASK_BLOCK_ID

        return input_tensor

    def generate_structure(
        self,
        input_tensor: torch.Tensor,
        temperature: float,
        start_radius: int,
        max_iterations: int,
        max_blocks: int,
        max_alternatives: int,
        min_alternative_probability: float,
    ):
        try:
            input_tensor = input_tensor.to(next(self.model.parameters()).device)
            self.logger.info(
                "Starting structure generation",
                extra={
                    "tensor_shape": input_tensor.shape,
                    "temperature": temperature,
                    "start_radius": start_radius,
                    "max_iterations": max_iterations,
                    "max_blocks": max_blocks,
                    "max_alternatives": max_alternatives,
                },
            )

            blocks_generated = 0
            for (
                alternative_num,
                previous_alternative_num,
                token,
                z,
                y,
                x,
            ) in self.model.fill_structure(
                input_tensor,
                temperature,
                start_radius,
                max_iterations,
                max_blocks,
                max_alternatives,
                min_alternative_probability,
            ):
                blocks_generated += 1
                versioned_block_str = self.block_token_mapper.token_to_versioned_str(
                    token
                )
                yield {
                    "alternative_num": alternative_num,
                    "previous_alternative_num": previous_alternative_num,
                    "block_state": versioned_block_str,
                    "z": z,
                    "y": y,
                    "x": x,
                }

            self.logger.info(
                "Structure generation completed",
                extra={"blocks_generated": blocks_generated},
            )

        except Exception:
            self.logger.error(
                "Error during structure generation",
                exc_info=True,
                extra={"tensor_shape": input_tensor.shape, "temperature": temperature},
            )
            raise
