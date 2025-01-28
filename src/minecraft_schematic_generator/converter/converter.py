from typing import Optional

import numpy as np
from schempy import Block, Schematic

from .mapping import BlockTokenMapper


class SchematicArrayConverter:
    def __init__(self):
        self.block_token_mapper = BlockTokenMapper()

    def schematic_to_array(self, schematic: Schematic, update_mapping: bool = False):
        """
        Convert schematic to an array.
        """
        # Get block data from the schematic and create an empty copy
        original_block_data = schematic.get_raw_block_data()
        converted_block_data = np.zeros_like(original_block_data)

        # Go through each block in the palette
        for block, index in schematic.get_block_palette().items():
            # Map the block to a token
            token = self.block_token_mapper.block_to_token(block, update_mapping)

            # Replace positions in the array with the token
            converted_block_data[original_block_data == index] = token

        # Swap the dimensions of the array
        return np.swapaxes(converted_block_data, 0, 1)

    def array_to_schematic(
        self, array: np.ndarray, schematic: Optional[Schematic] = None
    ):
        """
        Convert an array to a schematic.
        """
        if schematic is None:
            schematic = Schematic(array.shape[2], array.shape[1], array.shape[0])

        # Loop through all blocks in the schematic
        for x, y, z in schematic.iter_block_positions():
            token = array[z, y, x].item()
            try:
                block = self.block_token_mapper.token_to_block(token)
            except KeyError:
                block = Block("minecraft:air")
            schematic.set_block(x, y, z, block)

        return schematic
