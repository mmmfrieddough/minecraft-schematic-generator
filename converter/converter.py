import nbtlib
import numpy as np
import torch
from litemapy import BlockState, Region, Schematic

from .mapping import BlockTokenMapper


class RegionTensorConverter:
    def __init__(self):
        self.block_token_mapper = BlockTokenMapper()

    def region_to_tensor(self, region: Region):
        """
        Convert a region to a tensor.
        """
        # Create a 3D NumPy array initialized with the default token
        air_token = self.block_token_mapper.block_to_token(
            BlockState('minecraft:air'))
        blocks_np = np.full((region.length, region.height,
                            region.width), fill_value=air_token, dtype=int)

        # Loop through all blocks in the region
        for x, y, z in region.allblockpos():
            block = region.getblock(x, y, z)
            token = self.block_token_mapper.block_to_token(block)
            blocks_np[z, y, x] = token

        # Convert the NumPy array to a PyTorch tensor
        # Use torch.long for token indices
        blocks_tensor = torch.tensor(blocks_np, dtype=torch.long)

        return blocks_tensor

    def tensor_to_region(self, tensor):
        """
        Convert a tensor to a region.
        """
        # Create a region
        region = Region(
            0, 0, 0, tensor.shape[2], tensor.shape[1], tensor.shape[0])

        # Loop through all blocks in the region
        for x, y, z in region.allblockpos():
            token = tensor[z, y, x].item()
            block = self.block_token_mapper.token_to_block(token)
            region.setblock(x, y, z, block)

        return region
