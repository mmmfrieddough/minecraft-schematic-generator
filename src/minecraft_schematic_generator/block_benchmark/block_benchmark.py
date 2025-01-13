import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
from schempy import Block, Schematic

from minecraft_schematic_generator.converter import SchematicArrayConverter


class BlockBenchmark:
    SCHEMATIC_SIZE = 11
    SCHEMATIC_MIDDLE = SCHEMATIC_SIZE // 2

    def __init__(self, random_seed=0):
        self.random_seed = random_seed

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

    def run(self, model):
        random.seed(self.random_seed)

        # Test #1 - Doors
        door_states = list(self.get_door_states())

        # Create two schematics
        complete_schematic = Schematic(
            self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE
        )
        partial_schematic = Schematic(
            self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE
        )

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

        # Track used positions including middle door
        used_positions = {
            (self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE),
            (self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE + 1, self.SCHEMATIC_MIDDLE),
            (self.SCHEMATIC_MIDDLE, self.SCHEMATIC_MIDDLE + 2, self.SCHEMATIC_MIDDLE),
        }

        # Track door positions and their states for later removal
        door_positions = {
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

        # Copy blocks to partial schematic
        for x, y, z in used_positions:
            block = complete_schematic.get_block(x, y, z)
            partial_schematic.set_block(x, y, z, block)

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

        # Randomly remove door halves from partial schematic
        removed_positions = set()
        # Group door positions by their base block position
        door_groups = {}
        for pos, state in door_positions.items():
            # Find the base position (2 blocks below upper door, 1 block below lower door)
            base_y = pos[1] - (2 if "upper" in state else 1)
            base_pos = (pos[0], base_y, pos[2])

            if base_pos not in door_groups:
                door_groups[base_pos] = []
            door_groups[base_pos].append((pos, state))

        # Remove one half of each door
        for door_parts in door_groups.values():
            # Sort by Y coordinate to ensure consistent order (lower then upper)
            door_parts.sort(key=lambda p: p[0][1])
            # Randomly choose which half to remove
            pos_to_remove, _ = random.choice(door_parts)
            partial_schematic.set_block(
                pos_to_remove[0],
                pos_to_remove[1],
                pos_to_remove[2],
                Block("minecraft:air"),
            )
            removed_positions.add(pos_to_remove)

        # Save schematics
        complete_schematic.save_to_file(Path("random_doors_complete.schem"), 2)
        partial_schematic.save_to_file(Path("random_doors_partial.schem"), 2)

        # Convert schematics to arrays
        schematic_array_converter = SchematicArrayConverter()
        complete_structure = schematic_array_converter.schematic_to_array(
            complete_schematic
        )
        partial_structure = schematic_array_converter.schematic_to_array(
            partial_schematic
        )

        partial_structure = (
            torch.from_numpy(partial_structure.astype(np.int64)).long().contiguous()
        )

        # for pos in removed_positions:
        #     partial_structure[pos[2], pos[1], pos[0]] = 597

        # # Convert array back to schematic
        # test = schematic_array_converter.array_to_schematic(partial_structure)

        # # Save generated schematic
        # test.save_to_file(Path("test.schem"), 2)

        # return

        # Run model
        # Remove air
        partial_structure[partial_structure == 1] = 0

        print(f"Running model for {partial_structure.shape} structure")

        # Generate a sample using the model
        generated_structure = model.one_shot_inference(partial_structure, 0.7)

        # Convert array back to schematic
        generated_schematic = schematic_array_converter.array_to_schematic(
            generated_structure
        )

        # Save generated schematic
        generated_schematic.save_to_file(Path("random_doors_generated.schem"), 2)

        # Compare original and generated structures at removed positions
        correct_predictions = 0
        total_predictions = len(removed_positions)

        for pos in removed_positions:
            original_value = complete_structure[pos[2], pos[1], pos[0]]
            generated_value = generated_structure[pos[2], pos[1], pos[0]]

            if original_value == generated_value:
                correct_predictions += 1

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        print(
            f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)"
        )

        return accuracy
