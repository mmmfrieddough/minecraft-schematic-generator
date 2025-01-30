from typing import Optional

import amulet
import numpy as np
import PyMCTranslate
import schempy
from schempy import Schematic

from .mapping import BlockTokenMapper


class SchematicArrayConverter:
    def __init__(self):
        self.block_token_mapper = BlockTokenMapper()
        translation_manager = PyMCTranslate.new_translation_manager()
        self.version = translation_manager.get_version("java", (1, 21, 4))

    def _block_to_universal(self, block: schempy.Block) -> str:
        block_str = str(block)
        amulet_block = amulet.Block.from_string_blockstate(block_str)
        universal_block, _, _ = self.version.block.to_universal(amulet_block)
        universal_block_str = universal_block.blockstate
        return universal_block_str

    def _universal_to_block(self, universal_block_str: str) -> schempy.Block:
        amulet_block = amulet.Block.from_string_blockstate(universal_block_str)
        versioned_block, _, _ = self.version.block.from_universal(amulet_block)
        block_str = versioned_block.blockstate
        block = schempy.Block.from_string(block_str)
        return block

    def schematic_to_array(self, schematic: Schematic, update_mapping: bool = False):
        """
        Convert schematic to an array.
        """
        # Get block data from the schematic and create an empty copy
        original_block_data = schematic.get_raw_block_data()
        converted_block_data = np.zeros_like(original_block_data)

        # Go through each block in the palette
        for block, index in schematic.get_block_palette().items():
            # Convert to amulet universal
            universal_block_str = self._block_to_universal(block)

            # Map the block to a token
            token = self.block_token_mapper.block_str_to_token(
                universal_block_str, update_mapping
            )

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
                block_str = self.block_token_mapper.token_to_block_str(token)
            except KeyError:
                block_str = "universal_minecraft:air"
            block = self._universal_to_block(block_str)
            schematic.set_block(x, y, z, block)

        return schematic
