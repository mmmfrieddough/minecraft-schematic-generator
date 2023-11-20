from typing import Optional

import numpy as np
from schempy import Block, Schematic

from .mapping import BlockTokenMapper


class SchematicArrayConverter:
    def __init__(self):
        self.block_token_mapper = BlockTokenMapper()

    def schematic_to_array(self, schematic: Schematic):
        """
        Convert schematic to an array.
        """
        # Create a 3D NumPy array initialized with the default token
        air_token = self.block_token_mapper.block_to_token(
            Block('minecraft:air'))
        array = np.full((schematic.length, schematic.height,
                        schematic.width), fill_value=air_token, dtype=int)

        # Loop through all blocks in the region
        for x, y, z in schematic.iter_block_positions():
            block = schematic.get_block(x, y, z)
            token = self.block_token_mapper.block_to_token(block)
            array[z, y, x] = token

        return array

    def array_to_schematic(self, array: np.ndarray, schematic: Optional[Schematic] = None):
        """
        Convert an array to a schematic.
        """
        if schematic is None:
            schematic = Schematic(
                array.shape[2], array.shape[1], array.shape[0])

        # Loop through all blocks in the schematic
        for x, y, z in schematic.iter_block_positions():
            token = array[z, y, x].item()
            block = self.block_token_mapper.token_to_block(token)
            schematic.set_block(x, y, z, block)

        return schematic
