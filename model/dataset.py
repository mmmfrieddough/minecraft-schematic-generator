import random
import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.functional import conv3d


class MinecraftDataset(Dataset):
    def __init__(self, file_path: str, split: str, generator: str):
        self.file_path: str = file_path
        self.split: str = split
        self.generator: str = generator
        with h5py.File(self.file_path, 'r') as file:
            self.names = list(file[split][generator].keys())
            self.length: int = len(self.names)

    def __len__(self):
        return self.length

    def _apply_mask_type(self, structure: torch.Tensor, min_d: int, max_d: int, min_h: int, max_h: int, min_w: int, max_w: int) -> torch.Tensor:
        masked_structure = structure.clone()

        block_counts = torch.bincount(masked_structure.view(-1))
        frequent_block_types = torch.where(block_counts > 10)[0]
        infrequent_block_types = torch.where(
            (block_counts > 0) & (block_counts <= 10))[0]
        frequent_block_types = frequent_block_types[frequent_block_types > 2]

        structure_blocks = block_counts[3:].sum()
        percentage_structure_blocks = structure_blocks / masked_structure.numel()

        mask_types = [0, 1]
        if len(frequent_block_types) > 1:
            mask_types.append(2)
        if len(frequent_block_types) > 0 and len(infrequent_block_types) > 0:
            mask_types.append(3)
        if percentage_structure_blocks > 0.25:
            mask_types.append(4)
        mask_type = random.choice(mask_types)

        # Choose a random section to keep
        if mask_type == 0:
            def get_random_positions(min_val, max_val, max_size):
                size = random.randint(1, max_size)
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

            # Decide if we should leave behind all 0s or random
            if random.choice([True, False]):
                mask_prob = random.random()
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).long()
            else:
                mask = torch.zeros_like(masked_structure)
            mask[d_start:d_end, h_start:h_end, w_start:w_end] = 1
            masked_structure *= mask

        # Choose a random section to mask
        elif mask_type == 1:
            # Randomly choose one of the dimensions to start with
            dim = random.choice(['d', 'h', 'w'])
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
                size = random.randint(1, max_size)
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
                masked_structure[d_start:d_end,
                                 h_start:h_end, w_start:w_end] = 0
            else:
                mask_prob = random.random()
                mask = torch.bernoulli(torch.full(
                    masked_structure[d_start:d_end, h_start:h_end, w_start:w_end].shape, mask_prob)).long()
                masked_structure[d_start:d_end,
                                 h_start:h_end, w_start:w_end] *= mask

        # Choose a random block type to mask
        elif mask_type == 2:
            # Choose a block type to mask
            block_type = random.choice(frequent_block_types)

            # Decide if we should mask all blocks of the type or some
            if random.choice([True, False]):
                # Mask all blocks of the type
                masked_structure[masked_structure == block_type] = 0
            else:
                # Mask some blocks of the type
                mask_prob = random.random()
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[(masked_structure == block_type) & mask] = 0

        # Remove the infrequent block types
        elif mask_type == 3:
            # Decide if we should remove all infrequent block types or some
            infrequent_mask = torch.isin(
                masked_structure, infrequent_block_types)
            if random.choice([True, False]):
                # Remove all infrequent block types
                masked_structure[infrequent_mask] = 0
            else:
                # Remove some infrequent block types
                mask_prob = random.random()
                random_mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[infrequent_mask & random_mask] = 0

        # Remove a layer of blocks
        else:
            # Create a 3x3x3 kernel with 1s surrounding the center
            kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
            kernel[0, 0, 1, 1, 1] = 0

            # Add an extra dimension to the structure tensor for batch and channel
            structure_padded = masked_structure[None, None, ...]

            # Find blocks next to air or water
            air_or_water = (structure_padded == 1) | (structure_padded == 2)
            blocks_next_to_air_or_water = conv3d(
                air_or_water.float(), kernel, padding=1) > 0

            # Remove the extra dimensions
            blocks_next_to_air_or_water.squeeze_()

            # Set blocks next to air or water to 0
            masked_structure[blocks_next_to_air_or_water] = 0

        return masked_structure

    def _mask_structure(self, structure: torch.Tensor) -> torch.Tensor:
        # Find the bounding box of the object within the air or water
        object_coords = ((structure != 1) & (
            structure != 2)).nonzero(as_tuple=True)
        min_d, max_d = object_coords[0].min(), object_coords[0].max()
        min_h, max_h = object_coords[1].min(), object_coords[1].max()
        min_w, max_w = object_coords[2].min(), object_coords[2].max()

        # Apply a mask to the structure
        for i in range(10):
            masked_structure = self._apply_mask_type(
                structure, min_d, max_d, min_h, max_h, min_w, max_w)

            # Check if there is at least one non-air or non-water block left
            has_non_air_water_block = (masked_structure > 2).any()

            # Check if at least one non-air or non-water block has been masked
            non_air_water_mask = (structure > 2)
            has_masked_non_air_water_block = not torch.equal(
                structure[non_air_water_mask], masked_structure[non_air_water_mask])

            if has_non_air_water_block and has_masked_non_air_water_block:
                break

        # Decide if we should remove air
        if random.choice([True, False]):
            # Remove air
            if random.choice([True, False]):
                # Remove all air
                masked_structure[masked_structure == 1] = 0
            else:
                # Remove some air
                mask_prob = random.random()
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[(masked_structure == 1) & mask] = 0

        # Decide if we should remove water
        if random.choice([True, False]):
            # Remove water
            if random.choice([True, False]):
                # Remove all water
                masked_structure[masked_structure == 2] = 0
            else:
                # Remove some water
                mask_prob = random.random()
                mask = torch.bernoulli(torch.full(
                    masked_structure.shape, mask_prob)).bool()
                masked_structure[(masked_structure == 2) & mask] = 0

        return masked_structure

    def __getitem__(self, idx):
        name = self.names[idx]
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.split][self.generator][name]
            structure = torch.from_numpy(group['structure'][:]).long()

        masked_structure = self._mask_structure(structure)

        return masked_structure, structure
