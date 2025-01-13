import random
from itertools import product

from schempy import Block

from .structure_benchmark import StructureBenchmark


class TallPlantBenchmark(StructureBenchmark):
    SCHEMATIC_SIZE = 11
    SCHEMATIC_MIDDLE = SCHEMATIC_SIZE // 2

    @staticmethod
    def get_plant_states():
        plant_types = [
            "lilac",
            "peony",
            "pitcher_plant",
            "rose_bush",
            "sunflower",
            "large_fern",
            "tall_grass",
        ]

        for plant_type in plant_types:
            lower_state = f"minecraft:{plant_type}[half=lower]"
            upper_state = f"minecraft:{plant_type}[half=upper]"
            yield {
                "type": plant_type,
                "lower": lower_state,
                "upper": upper_state,
            }

    def build_structure(self, complete_schematic, partial_schematic, seed):
        random.seed(seed)
        plant_states = list(self.get_plant_states())
        used_positions = set()
        plant_positions = {}

        # Place middle plant
        complete_schematic.set_block(
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE,
            Block("minecraft:grass_block"),
        )
        plant_state = random.choice(plant_states)
        complete_schematic.set_block(
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE + 1,
            self.SCHEMATIC_MIDDLE,
            Block(plant_state["lower"]),
        )
        complete_schematic.set_block(
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE + 2,
            self.SCHEMATIC_MIDDLE,
            Block(plant_state["upper"]),
        )

        # Track positions
        used_positions.update(
            {
                (self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE),
                (
                    self.SCHEMATIC_MIDDLE,
                    self.SCHEMATIC_MIDDLE + 1,
                    self.SCHEMATIC_MIDDLE,
                ),
                (
                    self.SCHEMATIC_MIDDLE,
                    self.SCHEMATIC_MIDDLE + 2,
                    self.SCHEMATIC_MIDDLE,
                ),
            }
        )
        plant_positions.update(
            {
                (
                    self.SCHEMATIC_MIDDLE,
                    self.SCHEMATIC_MIDDLE + 1,
                    self.SCHEMATIC_MIDDLE,
                ): plant_state["lower"],
                (
                    self.SCHEMATIC_MIDDLE,
                    self.SCHEMATIC_MIDDLE + 2,
                    self.SCHEMATIC_MIDDLE,
                ): plant_state["upper"],
            }
        )

        # Generate remaining valid positions
        valid_positions = [
            (x, y, z)
            for x, y, z in product(
                range(self.SCHEMATIC_SIZE),
                range(self.SCHEMATIC_SIZE - 2),
                range(self.SCHEMATIC_SIZE),
            )
            if (x, y, z) not in used_positions
        ]

        # Randomly place remaining plants
        random.shuffle(valid_positions)

        for x, y, z in valid_positions:
            if (
                (x, y, z) not in used_positions
                and (x, y + 1, z) not in used_positions
                and (x, y + 2, z) not in used_positions
                and random.random() > 0.5
            ):
                # Place in complete schematic
                complete_schematic.set_block(x, y, z, Block("minecraft:grass_block"))
                plant_state = random.choice(plant_states)
                complete_schematic.set_block(x, y + 1, z, Block(plant_state["lower"]))
                complete_schematic.set_block(x, y + 2, z, Block(plant_state["upper"]))

                # Copy to partial schematic
                partial_schematic.set_block(x, y, z, Block("minecraft:grass_block"))
                partial_schematic.set_block(x, y + 1, z, Block(plant_state["lower"]))
                partial_schematic.set_block(x, y + 2, z, Block(plant_state["upper"]))

                # Track positions
                used_positions.update({(x, y, z), (x, y + 1, z), (x, y + 2, z)})
                plant_positions[(x, y + 1, z)] = plant_state["lower"]
                plant_positions[(x, y + 2, z)] = plant_state["upper"]

        # Remove random halves and return removed positions
        removed_positions = self.remove_plant_halves(partial_schematic, plant_positions)

        return removed_positions

    def remove_plant_halves(self, partial_schematic, plant_positions):
        removed_positions = set()
        # Group plant positions by their base block position
        plant_groups = {}
        for pos, state in plant_positions.items():
            base_y = pos[1] - (2 if "upper" in state else 1)
            base_pos = (pos[0], base_y, pos[2])

            if base_pos not in plant_groups:
                plant_groups[base_pos] = []
            plant_groups[base_pos].append((pos, state))

        # Remove one half of each plant
        for plant_parts in plant_groups.values():
            plant_parts.sort(key=lambda p: p[0][1])
            pos_to_remove, _ = random.choice(plant_parts)
            partial_schematic.set_block(
                pos_to_remove[0],
                pos_to_remove[1],
                pos_to_remove[2],
                Block("minecraft:air"),
            )
            removed_positions.add(pos_to_remove)

        return removed_positions
