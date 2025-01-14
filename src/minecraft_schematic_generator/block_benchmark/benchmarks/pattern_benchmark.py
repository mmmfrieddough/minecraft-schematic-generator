import random
from enum import Enum, auto
from itertools import product
from typing import Optional, Tuple

from schempy import Block, Schematic

from .structure_benchmark import StructureBenchmark


class PatternType(Enum):
    PLANE = auto()
    CROSS = auto()


class PatternBenchmark(StructureBenchmark):
    # Basic blocks that can be used for patterns
    BASIC_BLOCKS = [
        "stone",
        "cobblestone",
        "stone_bricks",
        "mossy_stone_bricks",
        "cracked_stone_bricks",
        "deepslate",
        "cobbled_deepslate",
        "polished_deepslate",
        "deepslate_bricks",
        "deepslate_tiles",
        "dirt",
        "white_wool",
        "orange_wool",
        "magenta_wool",
        "light_blue_wool",
        "yellow_wool",
        "lime_wool",
        "pink_wool",
        "gray_wool",
        "light_gray_wool",
        "cyan_wool",
        "purple_wool",
        "blue_wool",
        "brown_wool",
        "green_wool",
        "red_wool",
        "black_wool",
        "glass",
        "white_stained_glass",
        "orange_stained_glass",
        "magenta_stained_glass",
        "light_blue_stained_glass",
        "yellow_stained_glass",
        "lime_stained_glass",
        "pink_stained_glass",
        "gray_stained_glass",
        "light_gray_stained_glass",
        "cyan_stained_glass",
        "purple_stained_glass",
        "blue_stained_glass",
        "brown_stained_glass",
        "green_stained_glass",
        "red_stained_glass",
        "black_stained_glass",
        "terracotta",
        "white_terracotta",
        "orange_terracotta",
        "magenta_terracotta",
        "light_blue_terracotta",
        "yellow_terracotta",
        "lime_terracotta",
        "pink_terracotta",
        "gray_terracotta",
        "light_gray_terracotta",
        "cyan_terracotta",
        "purple_terracotta",
        "blue_terracotta",
        "brown_terracotta",
        "green_terracotta",
        "red_terracotta",
        "black_terracotta",
        "white_concrete",
        "orange_concrete",
        "magenta_concrete",
        "light_blue_concrete",
        "yellow_concrete",
        "lime_concrete",
        "pink_concrete",
        "gray_concrete",
        "light_gray_concrete",
        "cyan_concrete",
        "purple_concrete",
        "blue_concrete",
        "brown_concrete",
        "green_concrete",
        "red_concrete",
        "black_concrete",
    ]

    def __init__(
        self,
        name: str,
        pattern_type: PatternType,
        checkerboard: bool = False,
        removal_chance: float = 0.3,
        save_debug_schematics=False,
        debug_output_dir="debug_schematics",
    ):
        super().__init__(name, save_debug_schematics, debug_output_dir)
        self.pattern_type = pattern_type
        self.checkerboard = checkerboard
        self.removal_chance = removal_chance

    def get_random_blocks(self, seed: int) -> Tuple[str, Optional[str]]:
        """Get one or two random blocks depending on checkerboard setting"""
        random.seed(seed)
        block1 = f"minecraft:{random.choice(self.BASIC_BLOCKS)}"
        block2 = None

        if self.checkerboard:
            # Ensure second block is different from the first
            remaining_blocks = [
                b for b in self.BASIC_BLOCKS if f"minecraft:{b}" != block1
            ]
            block2 = f"minecraft:{random.choice(remaining_blocks)}"

        return block1, block2

    def build_structure(
        self, complete_schematic: Schematic, partial_schematic: Schematic, seed: int
    ) -> set:
        random.seed(seed)
        block1, block2 = self.get_random_blocks(seed)
        placed_positions = set()

        if self.pattern_type == PatternType.PLANE:
            # Randomly choose which axis to fix
            fixed_axis = random.choice(["x", "y", "z"])
            fixed_value = self.SCHEMATIC_MIDDLE

            # Place blocks in the plane
            for pos in product(range(self.SCHEMATIC_SIZE), repeat=2):
                x, y, z = [0] * 3
                if fixed_axis == "x":
                    x, y, z = fixed_value, pos[0], pos[1]
                elif fixed_axis == "y":
                    x, y, z = pos[0], fixed_value, pos[1]
                else:  # z
                    x, y, z = pos[0], pos[1], fixed_value

                # Determine which block to use for checkerboard pattern
                block = block1
                if self.checkerboard and (pos[0] + pos[1]) % 2 == 0:
                    block = block2

                complete_schematic.set_block(x, y, z, Block(block))
                placed_positions.add((x, y, z))

        else:  # CROSS
            # Place blocks along each axis through the middle
            for i in range(self.SCHEMATIC_SIZE):
                positions = [
                    (i, self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE),
                    (self.SCHEMATIC_MIDDLE, i, self.SCHEMATIC_MIDDLE),
                    (self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE, i),
                ]

                for x, y, z in positions:
                    # Determine which block to use for checkerboard pattern
                    block = block1
                    if self.checkerboard and i % 2 == 0:
                        block = block2

                    complete_schematic.set_block(x, y, z, Block(block))
                    placed_positions.add((x, y, z))

        # Copy blocks to partial schematic
        for pos in placed_positions:
            block = complete_schematic.get_block(pos[0], pos[1], pos[2])
            partial_schematic.set_block(pos[0], pos[1], pos[2], block)

        # Get valid removable positions (those with adjacent blocks)
        removed_positions = set()
        removable_positions = []

        for pos in placed_positions:
            # Check all 6 adjacent positions
            adjacent_positions = [
                (pos[0] + 1, pos[1], pos[2]),
                (pos[0] - 1, pos[1], pos[2]),
                (pos[0], pos[1] + 1, pos[2]),
                (pos[0], pos[1] - 1, pos[2]),
                (pos[0], pos[1], pos[2] + 1),
                (pos[0], pos[1], pos[2] - 1),
            ]

            # Count how many adjacent positions have blocks
            adjacent_blocks = sum(
                1 for adj_pos in adjacent_positions if adj_pos in placed_positions
            )

            # Only add to removable if it would leave at least one adjacent block
            if (
                adjacent_blocks > 1
            ):  # Need more than 1 so removing this block leaves at least 1
                removable_positions.append(pos)

        # Randomly remove blocks from valid positions
        random.shuffle(removable_positions)
        target_removals = int(len(placed_positions) * self.removal_chance)

        for pos in removable_positions:
            if len(removed_positions) >= target_removals:
                break

            # Check if removing this block would isolate any neighbors
            adjacent_positions = [
                (pos[0] + 1, pos[1], pos[2]),
                (pos[0] - 1, pos[1], pos[2]),
                (pos[0], pos[1] + 1, pos[2]),
                (pos[0], pos[1] - 1, pos[2]),
                (pos[0], pos[1], pos[2] + 1),
                (pos[0], pos[1], pos[2] - 1),
            ]

            can_remove = True
            for adj_pos in adjacent_positions:
                if adj_pos in placed_positions and adj_pos not in removed_positions:
                    # Check if this neighbor would still have at least one other neighbor
                    adj_neighbors = [
                        (adj_pos[0] + 1, adj_pos[1], adj_pos[2]),
                        (adj_pos[0] - 1, adj_pos[1], adj_pos[2]),
                        (adj_pos[0], adj_pos[1] + 1, adj_pos[2]),
                        (adj_pos[0], adj_pos[1] - 1, adj_pos[2]),
                        (adj_pos[0], adj_pos[1], adj_pos[2] + 1),
                        (adj_pos[0], adj_pos[1], adj_pos[2] - 1),
                    ]
                    remaining_neighbors = sum(
                        1
                        for n in adj_neighbors
                        if n in placed_positions
                        and n not in removed_positions
                        and n != pos
                    )
                    if remaining_neighbors == 0:
                        can_remove = False
                        break

            if can_remove:
                partial_schematic.set_block(
                    pos[0], pos[1], pos[2], Block("minecraft:air")
                )
                removed_positions.add(pos)

        return removed_positions
