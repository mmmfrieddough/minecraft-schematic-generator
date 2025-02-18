import random
from itertools import product
from typing import Dict, Set, Tuple

from schempy import Block, Schematic

from minecraft_schematic_generator.converter import SchematicArrayConverter

from .structure_benchmark import StructureBenchmark


class PortalBenchmark(StructureBenchmark):
    MIN_PORTAL_WIDTH = 2
    MAX_PORTAL_WIDTH = 5
    MIN_PORTAL_HEIGHT = 3
    MAX_PORTAL_HEIGHT = 6

    def __init__(
        self,
        name: str,
        schematic_array_converter: SchematicArrayConverter,
        schematic_size: int,
        save_debug_schematics: bool = False,
        debug_output_dir: str = "debug_schematics",
    ):
        super().__init__(
            name,
            schematic_array_converter,
            schematic_size,
            save_debug_schematics,
            debug_output_dir,
        )
        self.portal_groups: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}
        self.portal_frames: Dict[Tuple[int, int, int], Set[Tuple[int, int, int]]] = {}

    @staticmethod
    def get_portal_states():
        return [
            {"axis": "x", "state": "minecraft:nether_portal[axis=x]"},
            {"axis": "z", "state": "minecraft:nether_portal[axis=z]"},
        ]

    def build_portal(
        self,
        complete_schematic: any,
        partial_schematic: any,
        start_pos: Tuple[int, int, int],
        portal_state: Dict[str, str],
        size: Tuple[int, int] = (2, 3),  # (width, height)
    ) -> Tuple[Set[Tuple[int, int, int]], Set[Tuple[int, int, int]]]:
        """
        Build a single portal structure and return sets of portal and frame block positions.
        """
        x, y, z = start_pos
        width, height = size
        portal_blocks = set()
        frame_blocks = set()
        is_x_axis = portal_state["axis"] == "x"

        # Calculate portal dimensions
        if is_x_axis:
            portal_range = range(x + 1, x + 1 + width)
            portal_coords = product(portal_range, range(y + 1, y + 1 + height), [z])
            frame_x_range = range(x, x + width + 2)

            # Build frame
            # Bottom and top
            for fx in frame_x_range:
                frame_blocks.add((fx, y, z))  # Bottom
                frame_blocks.add((fx, y + height + 1, z))  # Top

            # Sides
            for fy in range(y + 1, y + height + 2):
                frame_blocks.add((x, fy, z))  # Left
                frame_blocks.add((x + width + 1, fy, z))  # Right
        else:
            portal_range = range(z + 1, z + 1 + width)
            portal_coords = product([x], range(y + 1, y + 1 + height), portal_range)
            frame_z_range = range(z, z + width + 2)

            # Build frame
            # Bottom and top
            for fz in frame_z_range:
                frame_blocks.add((x, y, fz))  # Bottom
                frame_blocks.add((x, y + height + 1, fz))  # Top

            # Sides
            for fy in range(y + 1, y + height + 2):
                frame_blocks.add((x, fy, z))  # Left
                frame_blocks.add((x, fy, z + width + 1))  # Right

        # Add portal blocks
        portal_blocks.update(portal_coords)

        # Place blocks in schematics
        for pos in frame_blocks:
            complete_schematic.set_block(*pos, Block("minecraft:obsidian"))
            partial_schematic.set_block(*pos, Block("minecraft:obsidian"))

        for pos in portal_blocks:
            complete_schematic.set_block(*pos, Block(portal_state["state"]))
            partial_schematic.set_block(*pos, Block(portal_state["state"]))

        return portal_blocks, frame_blocks

    def can_place_portal(
        self,
        start_pos: Tuple[int, int, int],
        size: Tuple[int, int],
        is_x_axis: bool,
        used_positions: Set[Tuple[int, int, int]],
    ) -> bool:
        """Check if a portal can be placed at the given position"""
        x, y, z = start_pos
        width, height = size

        # Calculate the area needed for the portal including buffer
        if is_x_axis:
            area_coords = product(
                range(x, x + width + 2),
                range(y, y + height + 2),
                range(z, z + 1),
            )
        else:
            area_coords = product(
                range(x, x + 1),
                range(y, y + height + 2),
                range(z, z + width + 2),
            )

        # Check if any position is already used or out of bounds
        for px, py, pz in area_coords:
            if (
                (px, py, pz) in used_positions
                or not (0 <= px < self._schematic_size)
                or not (0 <= py < self._schematic_size)
                or not (0 <= pz < self._schematic_size)
            ):
                return False
        return True

    def build_structure(self, complete_schematic, partial_schematic, seed):
        random.seed(seed)
        portal_states = self.get_portal_states()
        used_positions = set()
        self.portal_groups.clear()
        self.portal_frames.clear()

        # Place middle portal
        middle_pos = (
            self._schematic_middle,
            self._schematic_middle,
            self._schematic_middle,
        )
        portal_state = random.choice(portal_states)
        portal_blocks, frame_blocks = self.build_portal(
            complete_schematic, partial_schematic, middle_pos, portal_state
        )
        used_positions.update(portal_blocks)
        used_positions.update(frame_blocks)
        self.portal_groups[middle_pos] = portal_blocks
        self.portal_frames[middle_pos] = frame_blocks

        # Try to place additional portals
        attempts = 50  # Limit attempts to avoid infinite loops
        while attempts > 0:
            portal_state = random.choice(portal_states)
            is_x_axis = portal_state["axis"] == "x"

            width = random.randint(self.MIN_PORTAL_WIDTH, self.MAX_PORTAL_WIDTH)
            height = random.randint(self.MIN_PORTAL_HEIGHT, self.MAX_PORTAL_HEIGHT)

            x = random.randint(
                0,
                (self._schematic_size - width - 2)
                if is_x_axis
                else self._schematic_size,
            )
            y = random.randint(0, self._schematic_size - height - 2)
            z = random.randint(
                0,
                (self._schematic_size - width - 2)
                if not is_x_axis
                else self._schematic_size,
            )

            if self.can_place_portal(
                (x, y, z), (width, height), is_x_axis, used_positions
            ):
                portal_blocks, frame_blocks = self.build_portal(
                    complete_schematic,
                    partial_schematic,
                    (x, y, z),
                    portal_state,
                    (width, height),
                )
                used_positions.update(portal_blocks)
                used_positions.update(frame_blocks)
                self.portal_groups[(x, y, z)] = portal_blocks
                self.portal_frames[(x, y, z)] = frame_blocks

            attempts -= 1

        # Remove random blocks and return removed positions
        removed_positions = self.remove_portal_blocks(
            partial_schematic, complete_schematic
        )
        return removed_positions

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
                0 <= ax < self._schematic_size
                and 0 <= ay < self._schematic_size
                and 0 <= az < self._schematic_size
            ):
                block = schematic.get_block(ax, ay, az)
                if block and "minecraft:air" not in str(block):
                    return True
        return False

    def remove_portal_blocks(self, partial_schematic, complete_schematic):
        removed_positions = set()
        blocks_to_remove = []  # Track all blocks we want to remove

        # For each portal structure
        for base_pos, portal_blocks in self.portal_groups.items():
            if random.random() > 0.75:
                continue

            frame_blocks = self.portal_frames[base_pos]

            # Select portal blocks to remove (ensuring at least one remains)
            portal_blocks_list = list(portal_blocks)
            num_portal_blocks_to_remove = random.randint(1, len(portal_blocks) - 1)
            portal_blocks_to_remove = random.sample(
                portal_blocks_list, num_portal_blocks_to_remove
            )

            # Select frame blocks to remove
            frame_blocks_list = list(frame_blocks)
            num_frame_blocks_to_remove = random.randint(0, len(frame_blocks) // 2)
            frame_blocks_to_remove = random.sample(
                frame_blocks_list, num_frame_blocks_to_remove
            )

            blocks_to_remove.extend(portal_blocks_to_remove)
            blocks_to_remove.extend(frame_blocks_to_remove)

        # Remove all selected blocks first
        for pos in blocks_to_remove:
            if not self.is_corner_block(pos, base_pos):
                partial_schematic.set_block(*pos, Block("minecraft:air"))
                removed_positions.add(pos)

        # Check all removed positions and restore any that don't have adjacent solid blocks
        restored_positions = set()
        for pos in removed_positions:
            if not self.has_adjacent_solid_block(partial_schematic, *pos):
                x, y, z = pos
                # Get the original block type based on whether it was a portal or frame block
                block = complete_schematic.get_block(x, y, z)
                partial_schematic.set_block(x, y, z, block)
                restored_positions.add(pos)

        # Return only the positions that remained removed
        return removed_positions - restored_positions

    def is_corner_block(
        self, pos: Tuple[int, int, int], base_pos: Tuple[int, int, int]
    ) -> bool:
        """Determine if a frame block is a corner block"""
        x, y, z = pos
        portal_blocks = self.portal_groups[base_pos]

        # Get the bounds of the portal
        min_x = min(p[0] for p in portal_blocks)
        max_x = max(p[0] for p in portal_blocks)
        min_z = min(p[2] for p in portal_blocks)
        max_z = max(p[2] for p in portal_blocks)
        min_y = min(p[1] for p in portal_blocks)
        max_y = max(p[1] for p in portal_blocks)

        # Check if the position is at a corner
        is_corner = (
            (x in (min_x - 1, max_x + 1) and z in (min_z - 1, max_z + 1))
            or (x in (min_x - 1, max_x + 1) and y in (min_y - 1, max_y + 1))
            or (z in (min_z - 1, max_z + 1) and y in (min_y - 1, max_y + 1))
        )
        return is_corner
