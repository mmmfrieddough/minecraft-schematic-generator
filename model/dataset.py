import random

import h5py
import numpy as np
import torch
from torch.nn.functional import conv3d
from torch.utils.data import Dataset
from tqdm import tqdm

from converter import BlockTokenMapper


class MinecraftDataset(Dataset):
    def __init__(self, file_path: str, split: str, generator: str):
        natural_block_strings = [
            'minecraft:dirt',
            'minecraft:stone',
            'minecraft:grass_block[snowy=false]',
            'minecraft:grass_block[snowy=true]',
            'minecraft:water[level=0]',
            'minecraft:netherrack',
            'minecraft:bedrock',
            'minecraft:basalt[axis=y]',
            'minecraft:blackstone',
            'minecraft:gravel',
            'minecraft:dripstone_block',
            'minecraft:moss_block',
            'minecraft:deepslate[axis=y]'
            'minecraft:tuff',
            'minecraft:snow_block',
            'minecraft:ice',
            'minecraft:packed_ice',
            'minecraft:lava[level=0]',
            'minecraft:sand',
            'minecraft:sandstone',
            'minecraft:terracotta',
            'minecraft:red_terracotta',
            'minecraft:orange_terracotta',
            'minecraft:yellow_terracotta',
            'minecraft:brown_terracotta',
            'minecraft:white_terracotta',
            'minecraft:light_gray_terracotta',
            'minecraft:end_stone'
        ]
        mapper = BlockTokenMapper()
        self.natural_block_tokens = torch.tensor(
            [mapper.block_str_to_token(block) for block in natural_block_strings])

        with h5py.File(file_path, 'r') as file:
            group = file[split][generator]
            self.length = len(group['names'])
            self.structures = [torch.from_numpy(structure.astype(
                np.int64)).long() for structure in tqdm(group['structures'], desc=f'Loading {split} {generator} dataset', leave=False)]


def _create_point_noise(tensor: torch.Tensor, point: tuple[int, int, int], bias: float, min_radius: int, max_radius: int) -> torch.Tensor:
    # Get the dimensions of the tensor
    D, H, W = tensor.shape

    # Create a grid of coordinates
    d_coords, h_coords, w_coords = torch.meshgrid(
        torch.arange(D), torch.arange(H), torch.arange(W), indexing='ij')

    # Calculate the distance from the point
    distances = torch.sqrt(
        (d_coords - point[0])**2 + (h_coords - point[1])**2 + (w_coords - point[2])**2)

    # Apply min and max radius
    mask = (distances >= min_radius) & (distances <= max_radius)

    # Normalize distances to range [0, 1] within the valid radius
    max_distance = min(max_radius, torch.sqrt(
        torch.tensor(D**2 + H**2 + W**2, dtype=torch.float32)))
    normalized_distances = torch.where(
        mask, distances / max_distance, torch.ones_like(distances))

    # Apply bias to the normalized distances
    biased_distances = normalized_distances ** bias

    # Generate noise based on the biased distances, only within the valid radius
    noise = torch.where(mask, torch.bernoulli(biased_distances),
                        torch.zeros_like(biased_distances)).long()

    return noise

    def _mask_operation(structure: torch.Tensor, position: torch.Tensor, block_types: torch.Tensor, bias: float, min_radius: int, max_radius: int, require_adjacent_air: bool) -> torch.Tensor:
        # Create noise around the point
        mask = MinecraftDataset._create_point_noise(
            structure, position, bias, min_radius, max_radius)

        # Find the block types to mask
        if block_types is not None:
            mask &= torch.isin(structure, block_types)

        # Find the blocks adjacent to air if required
        if require_adjacent_air:
            kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
            kernel[0, 0, 1, 1, 1] = 0

            # Add an extra dimension to the structure tensor for batch and channel
            structure_padded = structure[None, None, ...]

            # Find blocks next to air
            air = (structure_padded == 1)
            blocks_next_to_air = conv3d(air.float(), kernel, padding=1) > 0

            # Remove the extra dimensions
            blocks_next_to_air.squeeze_()

            # Set blocks next to air to 0
            mask &= blocks_next_to_air

        # Mask the structure
        structure[mask] = 1

    def _apply_mask_type(structure: torch.Tensor, min_d: int, max_d: int, min_h: int, max_h: int, min_w: int, max_w: int, ignore_blocks: torch.Tensor) -> torch.Tensor:
        # Find structure dimensions, not counting the ignore blocks
        # Need to account for the possibility of all blocks being ignore blocks, possibly defaulting to full dimensions

        # Calculate infrequent block types and stuff

        # Mask type 1: Choose a point and randomly mask positions with increasing probability as you move away from the point
        # The point is chosen randomly, but should be within the structure and biased towards the center of the full dimensions

        # Mask type 2: Choose a point and randomly mask positions with decreasing probability as you move away from the point
        # The point is chosen randomly, but should be within the structure and biased towards the outer edges of the full dimensions
        # Possibly choose multiple points?

        # Individual mask operation function
        # Options:
        # Point position
        # Types of blocks to mask
        # Mask further away or closer
        # Radius of mask
        # Probability of masking
        # Require masked blocks to be adjacent to air

        # Full mask operation function
        # Perform multiple mask operations in sequence with random options
        # Do the same with the frequent or infrequent block types

        masked_structure = structure.clone()

        block_counts = torch.bincount(masked_structure.view(-1))
        frequent_block_types = torch.where(block_counts > 10)[0]
        infrequent_block_types = torch.where(
            (block_counts > 0) & (block_counts <= 10))[0]
        infrequent_block_types = infrequent_block_types[~torch.isin(
            infrequent_block_types, ignore_blocks)]
        # print(f'infrequent_block_types: {infrequent_block_types}')

        # Remove air and ignore blocks from the block types
        frequent_block_types = frequent_block_types[(
            frequent_block_types > 1) & ~torch.isin(frequent_block_types, ignore_blocks)]
        # print(f'frequent_block_types: {frequent_block_types}')

        structure_blocks = block_counts[2:].sum()
        percentage_structure_blocks = structure_blocks / masked_structure.numel()
        # print(f'percentage_structure_blocks: {
        #       percentage_structure_blocks:.2%}')

        mask_types = [0, 0, 1, 1]
        if len(frequent_block_types) > 1:
            mask_types.append(2)
        if len(frequent_block_types) > 0 and len(infrequent_block_types) > 0:
            mask_types.append(3)
        if percentage_structure_blocks > 0.25:
            mask_types.append(4)
        # print(f'mask_types: {mask_types}')
        mask_type = random.choice(mask_types)

        # Choose a random section to keep
        if mask_type == 0:
            # print('Choosing random section to keep')

            def get_random_positions(min_val, max_val, max_size):
                size = int(1 + (max_size - 1) * (random.random() ** 0.5))
                start = random.randint(
                    max(min_val - size + 1, 0), min(max_val, max_size - size))
                end = start + size
                return start, end

            d_start, d_end = get_random_positions(
                min_d, max_d, masked_structure.shape[0])
            h_start, h_end = get_random_positions(
                min_h, max_h, masked_structure.shape[1])
            w_start, w_end = get_random_positions(
                min_w, max_w, masked_structure.shape[2])
            # print(f'd_start: {d_start}, d_end: {d_end}')
            # print(f'h_start: {h_start}, h_end: {h_end}')
            # print(f'w_start: {w_start}, w_end: {w_end}')

            # Decide if we should leave behind all 0s or random
            if random.choice([True, False]):
                mask_prob = random.uniform(0.1, 0.9)
                # print(f'Masking a random selection with probability {
                #   mask_prob}')
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).long()
            else:
                # print('Masking full section')
                mask = torch.zeros_like(masked_structure)
            mask[d_start:d_end, h_start:h_end, w_start:w_end] = 1
            masked_structure *= mask

        # Choose a random section to mask
        elif mask_type == 1:
            # print('Choosing random section to mask')

            # Randomly choose one of the dimensions to start with
            dim = random.choice(['d', 'h', 'w'])
            # print(f'Masking dimension: {dim}')
            if dim == 'd':
                min_val, max_val = min_d, max_d
            elif dim == 'h':
                min_val, max_val = min_h, max_h
            else:
                min_val, max_val = min_w, max_w

            # Choose a side to come from
            if random.choice([True, False]):
                end = random.randint(min(min_val + 1, max_val), max_val)
                start = random.randint(0, max(end - 1, 0))
            else:
                start = random.randint(min(min_val + 1, max_val), max_val)
                end = random.randint(
                    min(start + 1, masked_structure.shape[0]), masked_structure.shape[0])

            def get_random_positions(min_val, max_val, max_size):
                size = int(1 + (max_size - 1) * (random.random() ** 0.5))
                start = random.randint(
                    max(min_val - size + 1, 0), min(max_val, max_size - size))
                end = start + size
                return start, end

            # For the other 2 dimensions, both positions could be totally random within the space
            if dim != 'd':
                d_start, d_end = get_random_positions(
                    min_d, max_d, masked_structure.shape[0])
            else:
                d_start, d_end = start, end

            if dim != 'h':
                h_start, h_end = get_random_positions(
                    min_h, max_h, masked_structure.shape[1])
            else:
                h_start, h_end = start, end

            if dim != 'w':
                w_start, w_end = get_random_positions(
                    min_w, max_w, masked_structure.shape[2])
            else:
                w_start, w_end = start, end

            # Decide if the mask should be all 0s or random
            if random.choice([True, False]):
                # print('Masking full section')
                masked_structure[d_start:d_end,
                                 h_start:h_end, w_start:w_end] = 0
            else:
                # print('Masking a random selection')
                mask_prob = random.uniform(0.1, 0.9)
                mask = torch.bernoulli(torch.full(
                    masked_structure[d_start:d_end, h_start:h_end, w_start:w_end].shape, mask_prob)).long()
                masked_structure[d_start:d_end,
                                 h_start:h_end, w_start:w_end] *= mask

        # Choose a random block type to mask
        elif mask_type == 2:
            # print('Choosing random block type to mask')

            # Choose a block type to mask
            block_type = random.choice(frequent_block_types)
            # print(f'Block type: {block_type}')

            # Decide if we should mask all blocks of the type or some
            if random.choice([True, False]):
                # Mask all blocks of the type
                # print('Masking all blocks of the type')
                masked_structure[masked_structure == block_type] = 0
            else:
                # Mask some blocks of the type
                mask_prob = random.uniform(0.1, 0.9)
                # print(f'Masking some blocks of the type with probability {
                #   mask_prob}')
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[(masked_structure == block_type) & mask] = 0

        # Remove the infrequent block types
        elif mask_type == 3:
            # print('Removing infrequent block types')

            # Decide if we should remove all infrequent block types or some
            infrequent_mask = torch.isin(
                masked_structure, infrequent_block_types)
            if random.choice([True, False]):
                # Remove all infrequent block types
                # print('Removing all infrequent block types')
                masked_structure[infrequent_mask] = 0
            else:
                # Remove some infrequent block types
                mask_prob = random.uniform(0.1, 0.9)
                # print(f'Removing some infrequent block types with probability {
                #   mask_prob}')
                random_mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[infrequent_mask & random_mask] = 0

        # Remove a layer of blocks
        else:
            # print('Removing a layer of blocks')

            # Create a 3x3x3 kernel with 1s surrounding the center
            kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
            kernel[0, 0, 1, 1, 1] = 0

            # Add an extra dimension to the structure tensor for batch and channel
            structure_padded = masked_structure[None, None, ...]

            # Find blocks next to air
            air = (structure_padded == 1)
            blocks_next_to_air = conv3d(air.float(), kernel, padding=1) > 0

            # Remove the extra dimensions
            blocks_next_to_air.squeeze_()

            # Set blocks next to air to 0
            masked_structure[blocks_next_to_air] = 0

        # Fill back in ignored blocks
        mask = torch.isin(structure, ignore_blocks)
        masked_structure[mask] = structure[mask]

        return masked_structure

    def _mask_structure(self, structure: torch.Tensor) -> torch.Tensor:
        # Find the bounding box of the object within the air
        object_coords = (structure != 1).nonzero(as_tuple=True)

        if len(object_coords[0]) == 0:
            return structure

        min_d, max_d = object_coords[0].min(), object_coords[0].max()
        min_h, max_h = object_coords[1].min(), object_coords[1].max()
        min_w, max_w = object_coords[2].min(), object_coords[2].max()
        # print(f'min_d: {min_d}, max_d: {max_d}')
        # print(f'min_h: {min_h}, max_h: {max_h}')
        # print(f'min_w: {min_w}, max_w: {max_w}')

        # Apply a mask to the structure
        for _ in range(10):
            # print('Attempting mask')

            if random.random() < 0.75:
                # print('Ignoring natural blocks')
                ignore_blocks = self.natural_block_tokens
            else:
                # print('Not ignoring natural blocks')
                ignore_blocks = torch.tensor([])

            masked_structure = MinecraftDataset._apply_mask_type(
                structure, min_d, max_d, min_h, max_h, min_w, max_w, ignore_blocks)

            # Check if there is at least one non-air block left
            has_non_air_block = (masked_structure > 1).any()
            if not has_non_air_block:
                # print('No non-air block')
                continue

            # Check if at least one block has been masked
            if torch.equal(structure, masked_structure):
                # print('No masked block')
                continue

            break

        # Remove all air that isn't adjacent to a non-air block
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
        kernel[0, 0, 1, 1, 1] = 0

        # Add extra dimensions for batch and channel
        masked_structure_padded = masked_structure[None, None, ...]

        # Find non-air blocks
        non_air = (masked_structure_padded > 1).float()

        # Find air blocks adjacent to non-air blocks
        adjacent_to_non_air = conv3d(non_air, kernel, padding=1) > 0

        # Remove extra dimensions
        adjacent_to_non_air = adjacent_to_non_air.squeeze()

        # Set air blocks not adjacent to non-air blocks to 0
        masked_structure[(masked_structure == 1) & ~adjacent_to_non_air] = 0

        # Decide if we should remove air
        if random.choice([True, False]):
            # Remove air
            # print('Removing air')
            if random.choice([True, False]):
                # Remove all air
                # print('Removing all air')
                masked_structure[masked_structure == 1] = 0
            else:
                # Remove some air
                mask_prob = random.uniform(0.1, 0.9)
                # print(f'Removing some air with probability {mask_prob}')
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[(masked_structure == 1) & mask] = 0

        return masked_structure

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        structure = self.structures[idx]
        masked_structure = self._mask_structure(structure)
        return structure, masked_structure
