import torch
from schempy.components import BlockPalette

from minecraft_schematic_generator.converter import BlockTokenMapper
from minecraft_schematic_generator.data_preparer import clean_block_properties


class StructureGenerator:
    def __init__(self, model):
        self.model = model
        self.block_token_mapper = BlockTokenMapper()

    def convert_block_to_token(self, block_str: str) -> int:
        block = BlockPalette._parse_block_str(block_str.lower())
        clean_block_properties(block)
        try:
            return self.block_token_mapper.block_to_token(block)
        except KeyError:
            print(
                f"WARN: Block {block_str} not found in mapping. Returning unused token."
            )
            return self.block_token_mapper.find_next_available_token()

    def prepare_input_tensor(self, structure):
        input_structure_ids = [
            [[self.convert_block_to_token(block_str) for block_str in y] for y in z]
            for z in structure
        ]
        input_tensor = torch.tensor(input_structure_ids)
        input_tensor[input_tensor == 1] = 0  # Mask air blocks
        return input_tensor

    def generate_structure(
        self,
        input_tensor,
        temperature,
        start_radius,
        max_iterations,
        max_blocks,
        air_probability_iteration_scaling,
    ):
        for block, z, y, x in self.model.fill_structure(
            input_tensor,
            temperature,
            start_radius,
            max_iterations,
            max_blocks,
            air_probability_iteration_scaling,
        ):
            block = self.block_token_mapper.token_to_block(block)
            yield {"block_state": str(block), "z": z, "y": y, "x": x}
