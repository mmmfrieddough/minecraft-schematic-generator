import random
from enum import Enum, auto
from typing import List, Optional, Tuple

from schempy import Block, Schematic

from .structure_benchmark import StructureBenchmark


class RedstoneComponentType(Enum):
    LAMP = auto()
    DOOR = auto()
    PISTON = auto()


class PowerSourceType(Enum):
    LEVER_BLOCK_ABOVE = auto()
    LEVER_BLOCK_BELOW = auto()
    LEVER_DIRECT_BELOW = auto()
    TORCH_BELOW = auto()
    REDSTONE_BLOCK_ABOVE = auto()
    REDSTONE_BLOCK_BELOW = auto()


class RedstonePowerBenchmark(StructureBenchmark):
    # Define valid combinations of components and power sources
    VALID_COMBINATIONS = {
        RedstoneComponentType.LAMP: [
            PowerSourceType.LEVER_BLOCK_ABOVE,
            PowerSourceType.LEVER_BLOCK_BELOW,
            PowerSourceType.REDSTONE_BLOCK_ABOVE,
            PowerSourceType.REDSTONE_BLOCK_BELOW,
        ],
        RedstoneComponentType.DOOR: [
            PowerSourceType.LEVER_BLOCK_ABOVE,
            PowerSourceType.LEVER_DIRECT_BELOW,
            PowerSourceType.TORCH_BELOW,
            PowerSourceType.REDSTONE_BLOCK_ABOVE,
        ],
        RedstoneComponentType.PISTON: [
            PowerSourceType.LEVER_BLOCK_BELOW,
            PowerSourceType.TORCH_BELOW,
            PowerSourceType.REDSTONE_BLOCK_BELOW,
        ],
    }

    def __init__(
        self,
        name: str,
        component_type: RedstoneComponentType,
        save_debug_schematics=False,
        debug_output_dir="debug_schematics",
    ):
        super().__init__(name, save_debug_schematics, debug_output_dir)
        self.component_type = component_type

    def get_component_states(
        self,
        powered: bool,
        *,
        is_sticky: bool = None,
        door_facing: str = None,
        door_hinge: str = None,
    ) -> Tuple[str, Optional[str]]:
        """Get the block state(s) for a component based on power state"""
        if self.component_type == RedstoneComponentType.LAMP:
            return (
                "minecraft:redstone_lamp[lit=true]"
                if powered
                else "minecraft:redstone_lamp[lit=false]"
            ), None
        elif self.component_type == RedstoneComponentType.DOOR:
            return (
                (
                    f"minecraft:iron_door[facing={door_facing},half=lower,hinge={door_hinge},open=true]",
                    f"minecraft:iron_door[facing={door_facing},half=upper,hinge={door_hinge},open=true]",
                )
                if powered
                else (
                    f"minecraft:iron_door[facing={door_facing},half=lower,hinge={door_hinge},open=false]",
                    f"minecraft:iron_door[facing={door_facing},half=upper,hinge={door_hinge},open=false]",
                )
            )
        else:  # PISTON
            piston_type = "minecraft:sticky_piston" if is_sticky else "minecraft:piston"
            head_type = "sticky" if is_sticky else "normal"

            if powered:
                return (
                    f"{piston_type}[extended=true,facing=up]",
                    f"minecraft:piston_head[facing=up,short=false,type={head_type}]",
                )
            else:
                return (
                    f"{piston_type}[extended=false,facing=up]",
                    "minecraft:air",
                )

    def place_power_source(
        self,
        schematic: Schematic,
        x: int,
        z: int,
        source_type: PowerSourceType,
        top: int,
        bottom: int,
    ) -> List[Tuple[int, int, int]]:
        """Place a power source and return all positions used"""
        positions = []
        directions = ["north", "south", "east", "west"]

        if source_type == PowerSourceType.LEVER_BLOCK_ABOVE:
            direction = random.choice(directions)
            powered = random.random() > 0.5
            state = "true" if powered else "false"
            schematic.set_block(x, top, z, Block("minecraft:stone"))
            schematic.set_block(
                x,
                top + 1,
                z,
                Block(
                    f"minecraft:lever[face=floor,facing={direction},powered={state}]"
                ),
            )
            positions.extend([(x, top, z), (x, top + 1, z)])
            return positions, powered

        elif source_type == PowerSourceType.LEVER_BLOCK_BELOW:
            direction = random.choice(directions)
            powered = random.random() > 0.5
            state = "true" if powered else "false"
            schematic.set_block(x, bottom, z, Block("minecraft:stone"))
            schematic.set_block(
                x,
                bottom - 1,
                z,
                Block(
                    f"minecraft:lever[face=ceiling,facing={direction},powered={state}]"
                ),
            )
            positions.extend([(x, bottom, z), (x, bottom - 1, z)])
            return positions, powered

        elif source_type == PowerSourceType.LEVER_DIRECT_BELOW:
            direction = random.choice(directions)
            powered = random.random() > 0.5
            state = "true" if powered else "false"
            schematic.set_block(
                x,
                bottom,
                z,
                Block(
                    f"minecraft:lever[face=ceiling,facing={direction},powered={state}]"
                ),
            )
            positions.append((x, bottom, z))
            return positions, powered

        elif source_type == PowerSourceType.TORCH_BELOW:
            schematic.set_block(x, bottom - 1, z, Block("minecraft:stone"))
            schematic.set_block(
                x, bottom, z, Block("minecraft:redstone_torch[lit=true]")
            )
            positions.extend([(x, bottom - 1, z), (x, bottom, z)])

        elif source_type == PowerSourceType.REDSTONE_BLOCK_ABOVE:
            schematic.set_block(x, top, z, Block("minecraft:redstone_block"))
            positions.append((x, top, z))

        elif source_type == PowerSourceType.REDSTONE_BLOCK_BELOW:
            schematic.set_block(x, bottom, z, Block("minecraft:redstone_block"))
            positions.append((x, bottom, z))

        return positions, True

    def has_adjacent_solid_block(
        self, schematic: Schematic, x: int, y: int, z: int
    ) -> bool:
        """Check if position has at least one adjacent solid block in partial schematic"""
        adjacents = [
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ]

        for ax, ay, az in adjacents:
            # Check if coordinates are within schematic bounds
            if (
                0 <= ax < self.SCHEMATIC_SIZE
                and 0 <= ay < self.SCHEMATIC_SIZE
                and 0 <= az < self.SCHEMATIC_SIZE
            ):
                block = schematic.get_block(ax, ay, az)
                if block and "minecraft:air" not in str(block):
                    return True
        return False

    def build_structure(
        self, complete_schematic: Schematic, partial_schematic: Schematic, seed: int
    ) -> set:
        random.seed(seed)
        used_positions = set()
        removed_positions = set()

        # Make component-specific decisions once for the entire schematic
        is_sticky = (
            random.choice([True, False])
            if self.component_type == RedstoneComponentType.PISTON
            else None
        )
        door_facing = (
            random.choice(["north", "south", "east", "west"])
            if self.component_type == RedstoneComponentType.DOOR
            else None
        )
        door_hinge = (
            random.choice(["left", "right"])
            if self.component_type == RedstoneComponentType.DOOR
            else None
        )

        # Get valid power sources for this component
        valid_sources = self.VALID_COMBINATIONS[self.component_type]

        # Place components in a grid pattern across the entire middle layer
        for x in range(0, self.SCHEMATIC_SIZE):
            for z in range(0, self.SCHEMATIC_SIZE):
                y = self.SCHEMATIC_MIDDLE

                # Place base block if needed
                if self.component_type == RedstoneComponentType.DOOR:
                    complete_schematic.set_block(x, y, z, Block("minecraft:stone"))
                    partial_schematic.set_block(x, y, z, Block("minecraft:stone"))
                    used_positions.add((x, y, z))
                    y += 1

                # Randomly decide if we should place a power source
                place_power_source = random.random() > 0.5

                # Randomly choose and place power source
                powered = False
                if place_power_source:
                    top = self.SCHEMATIC_MIDDLE + 1
                    bottom = self.SCHEMATIC_MIDDLE - 1
                    if self.component_type == RedstoneComponentType.DOOR:
                        top = self.SCHEMATIC_MIDDLE + 3
                    source_type = random.choice(valid_sources)
                    # Try to place power source, respecting height constraints
                    source_positions, powered = self.place_power_source(
                        complete_schematic, x, z, source_type, top, bottom
                    )
                    used_positions.update(source_positions)

                # Get component block state(s)
                main_state, extra_state = self.get_component_states(
                    powered,
                    is_sticky=is_sticky,
                    door_facing=door_facing,
                    door_hinge=door_hinge,
                )

                # Place component
                complete_schematic.set_block(x, y, z, Block(main_state))
                used_positions.add((x, y, z))

                if extra_state:  # For doors that need two blocks
                    complete_schematic.set_block(x, y + 1, z, Block(extra_state))
                    used_positions.add((x, y + 1, z))

                # Randomly decide whether to remove this component
                if random.random() < 0.5:
                    # Only remove if there will be an adjacent solid block
                    if self.has_adjacent_solid_block(partial_schematic, x, y, z):
                        removed_positions.add((x, y, z))
                        if extra_state:
                            removed_positions.add((x, y + 1, z))
                    else:
                        # If no adjacent blocks, keep the component
                        partial_schematic.set_block(x, y, z, Block(main_state))
                        if extra_state:
                            partial_schematic.set_block(x, y + 1, z, Block(extra_state))
                else:
                    partial_schematic.set_block(x, y, z, Block(main_state))
                    if extra_state:
                        partial_schematic.set_block(x, y + 1, z, Block(extra_state))

        # Copy power sources to partial schematic
        for pos in used_positions - removed_positions:
            if pos not in removed_positions:
                block = complete_schematic.get_block(pos[0], pos[1], pos[2])
                if block is not None:  # Check if block exists at position
                    partial_schematic.set_block(pos[0], pos[1], pos[2], block)

        return removed_positions
