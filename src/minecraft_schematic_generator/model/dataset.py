import random

import h5py
import torch
from torch.nn.functional import conv3d
from torch.profiler import record_function
from torch.utils.data import Dataset

from minecraft_schematic_generator.converter import BlockTokenConverter


class MinecraftDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str,
        generator: str,
        block_token_mapper: BlockTokenConverter,
    ):
        # These should be in universal format as we are feeding them directly into the mapping
        natural_block_strings = [
            "universal_minecraft:dirt",
            "universal_minecraft:stone",
            "universal_minecraft:grass_block[snowy=false]",
            "universal_minecraft:grass_block[snowy=true]",
            "universal_minecraft:water[falling=false,flowing=false,level=0]",
            "universal_minecraft:netherrack",
            "universal_minecraft:bedrock[infiniburn=false]",
            "universal_minecraft:basalt[axis=y]",
            "universal_minecraft:blackstone",
            "universal_minecraft:gravel",
            "universal_minecraft:dripstone_block",
            "universal_minecraft:moss_block",
            "universal_minecraft:deepslate[axis=y]",
            "universal_minecraft:tuff",
            "universal_minecraft:snow_block",
            "universal_minecraft:ice",
            "universal_minecraft:packed_ice",
            "universal_minecraft:lava[falling=false,flowing=false,level=0]",
            "universal_minecraft:sand",
            "universal_minecraft:sandstone[variant=normal]",
            "universal_minecraft:terracotta",
            "universal_minecraft:stained_terracotta[color=red]",
            "universal_minecraft:stained_terracotta[color=orange]",
            "universal_minecraft:stained_terracotta[color=yellow]",
            "universal_minecraft:stained_terracotta[color=brown]",
            "universal_minecraft:stained_terracotta[color=white]",
            "universal_minecraft:stained_terracotta[color=light_gray]",
            "universal_minecraft:end_stone",
        ]
        self.natural_block_tokens = torch.tensor(
            [
                block_token_mapper.universal_to_token(block)
                for block in natural_block_strings
            ]
        )

        self.file_path = file_path
        self.split = split
        self.generator = generator

        # Get the dataset length
        with h5py.File(file_path, "r") as file:
            group = file[split][generator]
            self.length = len(group["names"])

    def _create_point_noise(
        tensor: torch.Tensor,
        point: tuple[int, int, int],
        bias: float,
        min_radius: int,
        max_radius: int,
        min_prob: float,
        max_prob: float,
    ) -> torch.Tensor:
        # Get the dimensions of the tensor
        D, H, W = tensor.shape

        # Create a grid of coordinates
        d_coords, h_coords, w_coords = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij"
        )

        # Calculate the distance from the point
        distances = torch.sqrt(
            (d_coords - point[0]) ** 2
            + (h_coords - point[1]) ** 2
            + (w_coords - point[2]) ** 2
        )

        # Apply min and max radius
        mask = (distances >= min_radius) & (distances <= max_radius)

        # Normalize distances to range [0, 1] within the valid radius
        max_distance = min(
            max_radius,
            torch.sqrt(torch.tensor(D**2 + H**2 + W**2, dtype=torch.float32)),
        )
        normalized_distances = torch.where(
            mask, distances / max_distance, torch.ones_like(distances)
        )

        # Apply bias to the normalized distances
        if bias >= 0:
            biased_distances = normalized_distances**bias
        else:
            biased_distances = 1 - (1 - normalized_distances) ** abs(bias)

        # Calculate probabilities based on biased distances
        probabilities = min_prob + (max_prob - min_prob) * (1 - biased_distances)

        # Clamp probabilities to [0, 1] range
        probabilities = torch.clamp(probabilities, 0, 1)

        # Additional safeguard: replace any NaN or inf values with 0.5
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=0.5, neginf=0.5)

        # Generate noise based on the biased distances, only within the valid radius
        try:
            noise = torch.where(
                mask, torch.bernoulli(probabilities), torch.zeros_like(probabilities)
            ).bool()
        except RuntimeError as e:
            print(f"Error in torch.bernoulli: {e}")
            print(f"probabilities shape: {probabilities.shape}")
            print(f"probabilities unique values: {torch.unique(probabilities)}")
            raise

        # Set values less than the min radius to True if bias is negative, or greater than the max radius if bias is positive
        if bias < 0:
            noise[distances < min_radius] = True
        else:
            noise[distances > max_radius] = True

        return noise

    def _mask_operation(
        structure: torch.Tensor,
        position: torch.Tensor,
        block_types: torch.Tensor,
        bias: float,
        min_radius: int,
        max_radius: int,
        min_prob: float,
        max_prob: float,
        require_adjacent_air: bool,
    ) -> torch.Tensor:
        # Create noise around the point
        mask = MinecraftDataset._create_point_noise(
            structure, position, bias, min_radius, max_radius, min_prob, max_prob
        )

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
            air = structure_padded == 1
            blocks_next_to_air = conv3d(air.float(), kernel, padding=1) > 0

            # Remove the extra dimensions
            blocks_next_to_air.squeeze_()

            # Set blocks next to air to 0
            mask &= blocks_next_to_air

        # Mask the structure
        structure[mask] = 0

        return structure

    def _mask_blocks(
        structure: torch.Tensor,
        block_types: torch.Tensor,
        min_d: int,
        max_d: int,
        min_h: int,
        max_h: int,
        min_w: int,
        max_w: int,
        max_dim: int,
    ) -> torch.Tensor:
        # print(f'block_types: {block_types}')

        # Randomize parameters
        if random.choice([True, False]):
            bias = random.uniform(-2.0, -0.001)
        else:
            bias = random.uniform(0.001, 2.0)
        position = (
            random.randint(min_d, max_d),
            random.randint(min_h, max_h),
            random.randint(min_w, max_w),
        )
        min_radius = random.randint(0, max_dim)
        max_radius = random.randint(min_radius, max_dim)
        min_prob = random.uniform(0.0, 0.3)
        max_prob = random.uniform(0.7, 1.0)
        adj_air = random.random() < 0.5

        # print(f'position: {position}')
        # print(f'bias: {bias}')
        # print(f'min_radius: {min_radius}')
        # print(f'max_radius: {max_radius}')
        # print(f'min_prob: {min_prob}')
        # print(f'max_prob: {max_prob}')
        # print(f'adj_air: {adj_air}')

        # Apply a mask to the structure
        masked_structure = MinecraftDataset._mask_operation(
            structure,
            position,
            block_types,
            bias,
            min_radius,
            max_radius,
            min_prob,
            max_prob,
            adj_air,
        )

        return masked_structure

    def _find_structure_bounds(
        structure: torch.Tensor, ignore_blocks: torch.Tensor
    ) -> tuple[int, int, int, int, int, int]:
        # Find the bounding box of the structure we're interested in
        object_coords = (
            (structure != 1) & ~torch.isin(structure, ignore_blocks)
        ).nonzero(as_tuple=True)

        # No object is found
        if len(object_coords[0]) == 0:
            # Return the full size of the structure
            return (
                0,
                structure.shape[0] - 1,
                0,
                structure.shape[1] - 1,
                0,
                structure.shape[2] - 1,
            )

        min_d, max_d = object_coords[0].min(), object_coords[0].max()
        min_h, max_h = object_coords[1].min(), object_coords[1].max()
        min_w, max_w = object_coords[2].min(), object_coords[2].max()

        return min_d, max_d, min_h, max_h, min_w, max_w

    def _mask_structure(self, structure: torch.Tensor) -> torch.Tensor:
        masked_structure = structure.clone()

        # Decide if we want to remove any solid blocks
        if random.random() < 0.9:
            # print('Removing solid blocks')
            # Decide if we should ignore natural blocks
            if random.random() < 0.75:
                # print('Ignoring natural blocks')
                ignore_blocks = self.natural_block_tokens
            else:
                # print('Not ignoring natural blocks')
                ignore_blocks = torch.tensor([])

            # Find the bounds of the structure
            min_d, max_d, min_h, max_h, min_w, max_w = (
                MinecraftDataset._find_structure_bounds(masked_structure, ignore_blocks)
            )
            # print(f'min_d: {min_d}, max_d: {max_d}')
            # print(f'min_h: {min_h}, max_h: {max_h}')
            # print(f'min_w: {min_w}, max_w: {max_w}')

            # Find the largest dimension
            max_dim = max(max_d - min_d, max_h - min_h, max_w - min_w)

            # Get unique block types ignore_blocks
            unique_blocks = torch.unique(masked_structure)
            unique_blocks = unique_blocks[~torch.isin(unique_blocks, ignore_blocks)]
            # print(f'unique_blocks: {unique_blocks}')

            # Apply for all blocks
            masked_structure = MinecraftDataset._mask_blocks(
                masked_structure,
                unique_blocks,
                min_d,
                max_d,
                min_h,
                max_h,
                min_w,
                max_w,
                max_dim,
            )

            # Apply again for sampling of block types
            for block in random.sample(
                unique_blocks.tolist(), random.randint(0, min(3, len(unique_blocks)))
            ):
                # print(f'block: {block}')

                # Decide whether to apply the mask or remove the block entirely
                if random.choice([True, False]):
                    masked_structure = MinecraftDataset._mask_blocks(
                        masked_structure,
                        torch.tensor([block]),
                        min_d,
                        max_d,
                        min_h,
                        max_h,
                        min_w,
                        max_w,
                        max_dim,
                    )
                else:
                    # print('Removing block entirely')
                    masked_structure[masked_structure == block] = 0

            # Decide whether to apply for air
            if random.random() < 0.3:
                # print('Removing some air')
                masked_structure = MinecraftDataset._mask_blocks(
                    masked_structure,
                    torch.tensor([1]),
                    min_d,
                    max_d,
                    min_h,
                    max_h,
                    min_w,
                    max_w,
                    max_dim,
                )

            # Fill back in middle block
            mid_d = masked_structure.shape[0] // 2
            mid_h = masked_structure.shape[1] // 2
            mid_w = masked_structure.shape[2] // 2
            masked_structure[mid_d, mid_h, mid_w] = structure[mid_d, mid_h, mid_w]

        # Remove air
        if random.random() < 0.1:
            # Remove all air
            # print('Removing all air')
            masked_structure[masked_structure == 1] = 0
        else:
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

        return masked_structure

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with record_function("getitem_total"):
            with record_function("read_structure"):
                # Load single structure from disk when needed
                with h5py.File(self.file_path, "r") as file:
                    # Read directly into a torch tensor with the correct dtype
                    structure = torch.from_numpy(
                        file[self.split][self.generator]["structures"][idx][()]
                    ).long()

            with record_function("mask_structure"):
                masked_structure = self._mask_structure(structure)

            return structure, masked_structure
