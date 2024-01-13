import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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

    def _mask_structure(self, structure: torch.Tensor) -> torch.Tensor:
        # Find the bounding box of the object
        object_coords = (structure != 1).nonzero(as_tuple=True)
        min_d, max_d = object_coords[0].min(), object_coords[0].max()
        min_h, max_h = object_coords[1].min(), object_coords[1].max()
        min_w, max_w = object_coords[2].min(), object_coords[2].max()

        if random.choice([True, False]):
            # Choose a random section to keep

            def get_random_positions(min_val, max_val, max_size):
                size = random.randint(1, max_size)
                start = random.randint(
                    max(min_val - size + 1, 0), min(max_val, max_size - size))
                end = start + size
                return start, end

            do_noise = random.choice([True, False])

            for _ in range(100):
                d_start, d_end = get_random_positions(
                    min_d, max_d, structure.shape[0])
                h_start, h_end = get_random_positions(
                    min_h, max_h, structure.shape[1])
                w_start, w_end = get_random_positions(
                    min_w, max_w, structure.shape[2])

                # Create the masked structure by copying the original and applying the mask
                masked_structure = structure.clone()
                # Decide if we should leave behind all 0s or random
                if do_noise:
                    mask_prob = random.random()
                    mask = torch.bernoulli(torch.full(
                        masked_structure.shape, mask_prob)).long()
                else:
                    mask = torch.zeros_like(structure)
                mask[d_start:d_end, h_start:h_end, w_start:w_end] = 1
                masked_structure *= mask

                # Check if the masked structure contains any value greater than 1
                if (masked_structure > 1).any():
                    break
        else:
            # Choose a random section to mask

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
                    min(start + 1, structure.shape[0]), structure.shape[0])

            def get_random_positions(min_val, max_val, max_size):
                size = random.randint(1, max_size)
                start = random.randint(
                    max(min_val - size + 1, 0), min(max_val, max_size - size))
                end = start + size
                return start, end

            # For the other 2 dimensions, both positions could be totally random within the space
            if dim != 'd':
                d_start, d_end = get_random_positions(
                    min_d, max_d, structure.shape[0])
            else:
                d_start, d_end = start, end

            if dim != 'h':
                h_start, h_end = get_random_positions(
                    min_h, max_h, structure.shape[1])
            else:
                h_start, h_end = start, end

            if dim != 'w':
                w_start, w_end = get_random_positions(
                    min_w, max_w, structure.shape[2])
            else:
                w_start, w_end = start, end

            # Create the masked structure by copying the original and applying the mask
            masked_structure = structure.clone()
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

        return masked_structure

    def __getitem__(self, idx):
        name = self.names[idx]
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.split][self.generator][name]
            structure = torch.from_numpy(group['structure'][:]).long()

        masked_structure = self._mask_structure(structure)

        return masked_structure, structure
