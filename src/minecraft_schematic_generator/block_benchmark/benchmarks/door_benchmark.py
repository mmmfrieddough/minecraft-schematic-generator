import random
from itertools import product

from schempy import Block

from .structure_benchmark import StructureBenchmark


class DoorBenchmark(StructureBenchmark):
    SCHEMATIC_SIZE = 11
    SCHEMATIC_MIDDLE = SCHEMATIC_SIZE // 2

    @staticmethod
    def get_door_states():
        door_types = [
            "oak",
            "spruce",
            "birch",
            "jungle",
            "acacia",
            "dark_oak",
            "mangrove",
            "cherry",
            "pale_oak",
            "bamboo",
            "crimson",
            "warped",
        ]
        facings = ["north", "south", "east", "west"]
        hinges = ["left", "right"]
        opens = [True, False]

        for door_type, facing, hinge, open_state in product(
            door_types, facings, hinges, opens
        ):
            lower_state = f"minecraft:{door_type}_door[facing={facing},half=lower,hinge={hinge},open={str(open_state).lower()}]"
            upper_state = f"minecraft:{door_type}_door[facing={facing},half=upper,hinge={hinge},open={str(open_state).lower()}]"
            yield {
                "type": door_type,
                "facing": facing,
                "hinge": hinge,
                "open": open_state,
                "lower": lower_state,
                "upper": upper_state,
            }

    def build_structure(self, complete_schematic, partial_schematic, seed):
        random.seed(seed)
        door_states = list(self.get_door_states())
        used_positions = set()
        door_positions = {}

        # Place middle door
        complete_schematic.set_block(
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE,
            Block("minecraft:stone"),
        )
        door_state = random.choice(door_states)
        complete_schematic.set_block(
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE + 1,
            self.SCHEMATIC_MIDDLE,
            Block(door_state["lower"]),
        )
        complete_schematic.set_block(
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE + 2,
            self.SCHEMATIC_MIDDLE,
            Block(door_state["upper"]),
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
        door_positions.update(
            {
                (
                    self.SCHEMATIC_MIDDLE,
                    self.SCHEMATIC_MIDDLE + 1,
                    self.SCHEMATIC_MIDDLE,
                ): door_state["lower"],
                (
                    self.SCHEMATIC_MIDDLE,
                    self.SCHEMATIC_MIDDLE + 2,
                    self.SCHEMATIC_MIDDLE,
                ): door_state["upper"],
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

        # Randomly place remaining doors
        random.shuffle(valid_positions)

        for x, y, z in valid_positions:
            if (
                (x, y, z) not in used_positions
                and (x, y + 1, z) not in used_positions
                and (x, y + 2, z) not in used_positions
                and random.random() > 0.5
            ):
                # Place in complete schematic
                complete_schematic.set_block(x, y, z, Block("minecraft:stone"))
                door_state = random.choice(door_states)
                complete_schematic.set_block(x, y + 1, z, Block(door_state["lower"]))
                complete_schematic.set_block(x, y + 2, z, Block(door_state["upper"]))

                # Copy to partial schematic
                partial_schematic.set_block(x, y, z, Block("minecraft:stone"))
                partial_schematic.set_block(x, y + 1, z, Block(door_state["lower"]))
                partial_schematic.set_block(x, y + 2, z, Block(door_state["upper"]))

                # Track positions
                used_positions.update({(x, y, z), (x, y + 1, z), (x, y + 2, z)})
                door_positions[(x, y + 1, z)] = door_state["lower"]
                door_positions[(x, y + 2, z)] = door_state["upper"]

        # Copy blocks to partial schematic
        for pos in used_positions:
            block = complete_schematic.get_block(pos[0], pos[1], pos[2])
            partial_schematic.set_block(pos[0], pos[1], pos[2], block)

        # Remove random halves and return removed positions
        removed_positions = self.remove_door_halves(partial_schematic, door_positions)

        return removed_positions

    def remove_door_halves(self, partial_schematic, door_positions):
        removed_positions = set()
        # Group door positions by their base block position
        door_groups = {}
        for pos, state in door_positions.items():
            base_y = pos[1] - (2 if "upper" in state else 1)
            base_pos = (pos[0], base_y, pos[2])

            if base_pos not in door_groups:
                door_groups[base_pos] = []
            door_groups[base_pos].append((pos, state))

        # Remove one half of each door
        for door_parts in door_groups.values():
            if random.random() > 0.75:
                continue
            door_parts.sort(key=lambda p: p[0][1])
            pos_to_remove, _ = random.choice(door_parts)
            partial_schematic.set_block(
                pos_to_remove[0],
                pos_to_remove[1],
                pos_to_remove[2],
                Block("minecraft:air"),
            )
            removed_positions.add(pos_to_remove)

        return removed_positions
