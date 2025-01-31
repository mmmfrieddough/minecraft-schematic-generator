from typing import Optional

import numpy as np
import schempy
from schempy import Schematic

from .mapping import BlockTokenMapper


class SchematicArrayConverter:
    def __init__(self, block_token_mapper: BlockTokenMapper | None = None):
        self.block_token_mapper = block_token_mapper or BlockTokenMapper()

    def schematic_to_array(
        self, schematic: Schematic, update_mapping: bool = False
    ) -> np.ndarray:
        """
        Convert schematic to an array.
        """
        # Get block data from the schematic and create an empty copy
        original_block_data = schematic.get_raw_block_data()
        converted_block_data = np.zeros_like(original_block_data)

        # Go through each block in the palette
        for block, index in schematic.get_block_palette().items():
            # Map the block to a token
            token = self.block_token_mapper.versioned_to_token(
                str(block), update_mapping
            )

            # Replace positions in the array with the token
            converted_block_data[original_block_data == index] = token

        # Swap the dimensions of the array
        return np.swapaxes(converted_block_data, 0, 1)

    def array_to_schematic(
        self, array: np.ndarray, schematic: Optional[Schematic] = None
    ) -> Schematic:
        """
        Convert an array to a schematic.
        """
        if schematic is None:
            schematic = Schematic(array.shape[2], array.shape[1], array.shape[0])

        # Loop through all blocks in the schematic
        for x, y, z in schematic.iter_block_positions():
            token = array[z, y, x].item()
            versioned_block_str = self.block_token_mapper.token_to_versioned(token)
            block = schempy.Block.from_string(versioned_block_str)
            schematic.set_block(x, y, z, block)

        return schematic
