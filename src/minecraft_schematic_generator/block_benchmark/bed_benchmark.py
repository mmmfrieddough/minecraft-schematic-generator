import random
from itertools import product

from schempy import Block

from .structure_benchmark import StructureBenchmark


class BedBenchmark(StructureBenchmark):
    SCHEMATIC_SIZE = 11
    SCHEMATIC_MIDDLE = SCHEMATIC_SIZE // 2

    @staticmethod
    def get_bed_states():
        bed_colors = [
            "white",
            "orange",
            "magenta",
            "light_blue",
            "yellow",
            "lime",
            "pink",
            "gray",
            "light_gray",
            "cyan",
            "purple",
            "blue",
            "brown",
            "green",
            "red",
            "black",
        ]
        facings = ["north", "south", "east", "west"]

        for color, facing in product(bed_colors, facings):
            head_state = (
                f"minecraft:{color}_bed[facing={facing},occupied=false,part=head]"
            )
            foot_state = (
                f"minecraft:{color}_bed[facing={facing},occupied=false,part=foot]"
            )

            # Define the offset for the head part based on facing
            if facing == "north":
                offset = (0, 0, -1)
            elif facing == "south":
                offset = (0, 0, 1)
            elif facing == "east":
                offset = (1, 0, 0)
            else:  # west
                offset = (-1, 0, 0)

            yield {
                "color": color,
                "facing": facing,
                "head": head_state,
                "foot": foot_state,
                "head_offset": offset,
            }

    def build_structure(self, complete_schematic, partial_schematic, seed):
        random.seed(seed)
        bed_states = list(self.get_bed_states())
        used_positions = set()
        bed_positions = {}

        # Place middle bed
        middle_x, middle_y, middle_z = (
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE,
            self.SCHEMATIC_MIDDLE,
        )
        bed_state = random.choice(bed_states)

        # Place support blocks
        complete_schematic.set_block(
            middle_x, middle_y, middle_z, Block("minecraft:stone")
        )
        head_x = middle_x + bed_state["head_offset"][0]
        head_z = middle_z + bed_state["head_offset"][2]
        complete_schematic.set_block(head_x, middle_y, head_z, Block("minecraft:stone"))

        # Place bed blocks
        complete_schematic.set_block(
            middle_x, middle_y + 1, middle_z, Block(bed_state["foot"])
        )
        complete_schematic.set_block(
            head_x, middle_y + 1, head_z, Block(bed_state["head"])
        )

        # Track positions
        used_positions.update(
            {
                (middle_x, middle_y, middle_z),
                (head_x, middle_y, head_z),
                (middle_x, middle_y + 1, middle_z),
                (head_x, middle_y + 1, head_z),
            }
        )
        bed_positions.update(
            {
                (middle_x, middle_y + 1, middle_z): bed_state["foot"],
                (head_x, middle_y + 1, head_z): bed_state["head"],
            }
        )

        # Generate remaining valid positions
        valid_positions = [
            (x, y, z)
            for x, y, z in product(
                range(self.SCHEMATIC_SIZE),
                range(self.SCHEMATIC_SIZE - 1),  # -1 because beds are only 1 block high
                range(self.SCHEMATIC_SIZE),
            )
            if (x, y, z) not in used_positions
        ]

        # Randomly place remaining beds
        random.shuffle(valid_positions)

        for x, y, z in valid_positions:
            if random.random() > 0.5:
                bed_state = random.choice(bed_states)
                head_x = x + bed_state["head_offset"][0]
                head_z = z + bed_state["head_offset"][2]

                # Check if all required positions are available
                required_positions = {
                    (x, y, z),
                    (head_x, y, head_z),
                    (x, y + 1, z),
                    (head_x, y + 1, head_z),
                }

                if all(
                    0 <= pos[0] < self.SCHEMATIC_SIZE
                    and 0 <= pos[2] < self.SCHEMATIC_SIZE
                    for pos in required_positions
                ) and not (required_positions & used_positions):
                    # Place support blocks
                    complete_schematic.set_block(x, y, z, Block("minecraft:stone"))
                    complete_schematic.set_block(
                        head_x, y, head_z, Block("minecraft:stone")
                    )

                    # Place bed blocks
                    complete_schematic.set_block(x, y + 1, z, Block(bed_state["foot"]))
                    complete_schematic.set_block(
                        head_x, y + 1, head_z, Block(bed_state["head"])
                    )

                    # Copy to partial schematic
                    partial_schematic.set_block(x, y, z, Block("minecraft:stone"))
                    partial_schematic.set_block(
                        head_x, y, head_z, Block("minecraft:stone")
                    )
                    partial_schematic.set_block(x, y + 1, z, Block(bed_state["foot"]))
                    partial_schematic.set_block(
                        head_x, y + 1, head_z, Block(bed_state["head"])
                    )

                    # Track positions
                    used_positions.update(required_positions)
                    bed_positions[(x, y + 1, z)] = bed_state["foot"]
                    bed_positions[(head_x, y + 1, head_z)] = bed_state["head"]

        # Remove random halves and return removed positions
        removed_positions = self.remove_bed_halves(partial_schematic, bed_positions)

        return removed_positions

    def remove_bed_halves(self, partial_schematic, bed_positions):
        removed_positions = set()

        # Group bed positions by their connected pairs
        bed_groups = {}
        for pos, state in bed_positions.items():
            if "foot" in state:
                facing = state.split("facing=")[1].split(",")[0]
                if facing == "north":
                    pair_pos = (pos[0], pos[1], pos[2] - 1)
                elif facing == "south":
                    pair_pos = (pos[0], pos[1], pos[2] + 1)
                elif facing == "east":
                    pair_pos = (pos[0] + 1, pos[1], pos[2])
                else:  # west
                    pair_pos = (pos[0] - 1, pos[1], pos[2])

                bed_groups[pos] = (pos, pair_pos)

        # Remove one half of each bed
        for foot_pos, head_pos in bed_groups.values():
            pos_to_remove = random.choice([foot_pos, head_pos])
            partial_schematic.set_block(
                pos_to_remove[0],
                pos_to_remove[1],
                pos_to_remove[2],
                Block("minecraft:air"),
            )
            removed_positions.add(pos_to_remove)

        return removed_positions
