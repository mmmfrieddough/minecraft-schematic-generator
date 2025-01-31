import random

from schempy import Block, Schematic

from minecraft_schematic_generator.converter import SchematicArrayConverter

from .structure_benchmark import StructureBenchmark


class StairsBenchmark(StructureBenchmark):
    # Available stair types
    STAIR_BLOCKS = [
        "oak_stairs",
        "brick_stairs",
        "birch_stairs",
        "stone_stairs",
        "sandstone_stairs",
        "purpur_stairs",
        "spruce_stairs",
        "jungle_stairs",
        "quartz_stairs",
        "acacia_stairs",
        "warped_stairs",
        "blackstone_stairs",
        "cobblestone_stairs",
        "dark_oak_stairs",
        "granite_stairs",
        "diorite_stairs",
        "crimson_stairs",
        "red_sandstone_stairs",
        "andesite_stairs",
        "cut_copper_stairs",
        "stone_brick_stairs",
        "prismarine_stairs",
        "smooth_sandstone_stairs",
        "mossy_cobblestone_stairs",
        "nether_brick_stairs",
        "smooth_quartz_stairs",
        "smooth_red_sandstone_stairs",
        "polished_blackstone_stairs",
        "end_stone_brick_stairs",
        "deepslate_tile_stairs",
        "dark_prismarine_stairs",
        "red_nether_brick_stairs",
        "deepslate_brick_stairs",
        "waxed_cut_copper_stairs",
        "prismarine_brick_stairs",
        "polished_granite_stairs",
        "mossy_stone_brick_stairs",
        "polished_diorite_stairs",
        "polished_andesite_stairs",
        "cobbled_deepslate_stairs",
        "exposed_cut_copper_stairs",
        "polished_blackstone_brick_stairs",
        "oxidized_cut_copper_stairs",
        "polished_deepslate_stairs",
        "weathered_cut_copper_stairs",
        "waxed_exposed_cut_copper_stairs",
        "waxed_oxidized_cut_copper_stairs",
        "waxed_weathered_cut_copper_stairs",
    ]

    def __init__(
        self,
        name: str,
        schematic_array_converter: SchematicArrayConverter,
        removal_chance: float = 0.3,
        min_width: int = 1,
        max_width: int = 3,
        save_debug_schematics=False,
        debug_output_dir="debug_schematics",
    ):
        super().__init__(
            name, schematic_array_converter, save_debug_schematics, debug_output_dir
        )
        self.removal_chance = removal_chance
        self.min_width = min_width
        self.max_width = max_width

    def build_structure(
        self, complete_schematic: Schematic, partial_schematic: Schematic, seed: int
    ) -> set:
        random.seed(seed)
        placed_positions = set()

        # Choose random parameters
        stair_type = random.choice(self.STAIR_BLOCKS)
        stair_width = random.randint(self.min_width, self.max_width)
        facing = random.choice(["north", "south", "east", "west"])

        # Calculate opposite direction for upside-down stairs
        opposite_directions = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
        }
        upside_down_facing = opposite_directions[facing]

        # Place stairs along the diagonal, adjusting for direction
        for i in range(self.SCHEMATIC_SIZE):
            # Place stairs across width
            for j in range(
                self.SCHEMATIC_MIDDLE - stair_width,
                self.SCHEMATIC_MIDDLE + stair_width + 1,
            ):
                # Adjust coordinates based on facing direction
                if facing == "east":
                    x, y, z = i, i, j
                elif facing == "west":
                    x, y, z = self.SCHEMATIC_SIZE - 1 - i, i, j
                elif facing == "north":
                    x, y, z = j, i, self.SCHEMATIC_SIZE - 1 - i
                else:  # south
                    x, y, z = j, i, i

                # Place normal stair
                normal_stair = f"minecraft:{stair_type}[facing={facing},half=bottom,shape=straight,waterlogged=false]"
                complete_schematic.set_block(x, y, z, Block(normal_stair))
                placed_positions.add((x, y, z))

                # Place upside-down stair below
                if y > 0:
                    upside_down_stair = f"minecraft:{stair_type}[facing={upside_down_facing},half=top,shape=straight,waterlogged=false]"
                    complete_schematic.set_block(x, y - 1, z, Block(upside_down_stair))
                    placed_positions.add((x, y - 1, z))

        # Copy blocks to partial schematic
        for pos in placed_positions:
            block = complete_schematic.get_block(pos[0], pos[1], pos[2])
            partial_schematic.set_block(pos[0], pos[1], pos[2], block)

        # Randomly remove blocks while ensuring connectivity
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

            # Count adjacent blocks
            adjacent_blocks = sum(
                1 for adj_pos in adjacent_positions if adj_pos in placed_positions
            )

            # Only consider removing if it has multiple neighbors
            if adjacent_blocks > 1:
                removable_positions.append(pos)

        # Randomly remove blocks while maintaining connectivity
        random.shuffle(removable_positions)
        target_removals = int(len(placed_positions) * self.removal_chance)

        for pos in removable_positions:
            if len(removed_positions) >= target_removals:
                break

            # Check if removing would isolate any neighbors
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
