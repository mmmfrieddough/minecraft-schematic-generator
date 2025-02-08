import logging

import torch

from minecraft_schematic_generator.constants import (
    AIR_BLOCK_ID,
    MASK_BLOCK_ID,
    VOID_AIR_BLOCK_ID,
    VOID_AIR_BLOCK_STR,
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
            universal_block = self.block_token_mapper.versioned_str_to_universal_block(
                block_str
            )
            if universal_block.blockstate == VOID_AIR_BLOCK_STR:
                return VOID_AIR_BLOCK_ID
            return self.block_token_mapper.universal_block_to_token(universal_block)
        except KeyError:
            self.logger.warning(
                f"Block {universal_block.blockstate} not found in mapping. Returning unused token."
            )
            return self.block_token_mapper.get_unused_token()

    def prepare_input_tensor(self, structure: list) -> torch.Tensor:
        input_structure_ids = [
            [[self.convert_block_to_token(block_str) for block_str in y] for y in z]
            for z in structure
        ]
        input_tensor = torch.tensor(input_structure_ids)

        # Mask air blocks
        input_tensor[input_tensor == AIR_BLOCK_ID] = MASK_BLOCK_ID

        # Turn void air into air
        input_tensor[input_tensor == VOID_AIR_BLOCK_ID] = AIR_BLOCK_ID

        return input_tensor

    def generate_structure(
        self,
        input_tensor: torch.Tensor,
        temperature: float,
        start_radius: int,
        max_iterations: int,
        max_blocks: int,
        air_probability_iteration_scaling: float,
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
                },
            )

            blocks_generated = 0
            for token, z, y, x in self.model.fill_structure(
                input_tensor,
                temperature,
                start_radius,
                max_iterations,
                max_blocks,
                air_probability_iteration_scaling,
            ):
                blocks_generated += 1
                versioned_block_str = self.block_token_mapper.token_to_versioned_str(
                    token
                )
                yield {"block_state": versioned_block_str, "z": z, "y": y, "x": x}

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
