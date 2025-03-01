import random

import torch
from amulet import Block
from amulet_nbt import StringTag

from minecraft_schematic_generator.converter import BlockTokenConverter


class StructureTransformer:
    def setup(self, block_token_converter: BlockTokenConverter):
        blocks = {
            i: block_token_converter.token_to_universal_block(i)
            for i in range(1, block_token_converter.get_unused_token())
        }
        self._rotation_luts = []
        for rotation in range(1, 4):

            def func(block: Block) -> Block:
                return StructureTransformer.rotate_block_properties(block, rotation)

            self._rotation_luts.append(
                StructureTransformer._create_lut(block_token_converter, blocks, func)
            )
        self._flip_lut = StructureTransformer._create_lut(
            block_token_converter, blocks, StructureTransformer.flip_block_properties
        )

    @staticmethod
    def _create_lut(
        block_token_converter: BlockTokenConverter,
        blocks: dict[int, Block],
        func: callable,
    ) -> dict[int, int]:
        lut = torch.arange(len(blocks) + 1)
        for token, block in blocks.items():
            if not block.properties:
                continue
            rotated_block = func(block)
            rotated_token = block_token_converter.universal_block_to_token(
                rotated_block
            )
            lut[token] = rotated_token
        return lut

    @staticmethod
    def rotate_block_properties(block: Block, rotation: int) -> Block:
        """
        Rotates the block-state properties around the vertical (y) axis.
        `rotations` is the number of 90-degree clockwise rotations (0..3).
        """
        props: dict = block.properties

        # Rotate 'facing' (e.g. 'north' -> 'east', etc.)
        if "facing" in props:
            direction = str(props["facing"])
            rotated_direction = StructureTransformer._rotate_direction(
                direction, rotation
            )
            props["facing"] = StringTag(rotated_direction)

        # Rotate 'axis' (e.g. logs or pillars)
        if "axis" in props:
            axis = str(props["axis"])
            rotated_axis = StructureTransformer._rotate_axis(axis, rotation)
            props["axis"] = StringTag(rotated_axis)

        if "rail" in block.base_name:
            shape = str(props["shape"])
            rotated_shape = StructureTransformer._rotate_rail_shape(shape, rotation)
            props["shape"] = StringTag(rotated_shape)

        if "rotation" in props:
            rotation_value = int(str(props["rotation"]))
            new_rotation = (rotation_value + 4 * rotation) % 16
            props["rotation"] = StringTag(str(new_rotation))

        # Rotate side-connection flags (e.g. fences, glass panes)
        side_keys = ["north", "east", "south", "west"]
        side_connections = {}
        for side in side_keys:
            # Pop them off if they exist
            if side in props:
                new_side = StructureTransformer._rotate_direction(side, rotation)
                side_connections[new_side] = props.pop(side)

        # Put them back in the properties
        props.update(side_connections)

        rotated_block = Block(
            namespace=block.namespace,
            base_name=block.base_name,
            properties=props,
        )

        return rotated_block

    @staticmethod
    def _rotate_direction(direction: str, k: int) -> str:
        """Rotates 'north', 'east', 'south', 'west' by k*90° clockwise around y."""
        directions = ["north", "east", "south", "west"]
        if direction not in directions:
            return direction  # e.g. 'up', 'down', or an unsupported value
        idx = directions.index(direction)
        return directions[(idx + k) % 4]

    @staticmethod
    def _rotate_axis(axis: str, k: int) -> str:
        """
        Rotates the axis property (x/y/z) around the vertical y-axis.
        - If axis = y, it stays y (because rotating around y doesn't change it).
        - x <-> z are swapped for 90° or 270°, remain the same for 180°, etc.
        """
        if axis == "y":
            return "y"
        elif axis == "x":
            # 90 or 270 swaps x <-> z; 180 leaves x
            return "z" if k % 2 == 1 else "x"
        elif axis == "z":
            # 90 or 270 swaps z <-> x; 180 leaves z
            return "x" if k % 2 == 1 else "z"
        else:
            return axis  # unknown axis, just return as is

    @staticmethod
    def _rotate_rail_shape(shape: str, k: int) -> str:
        """
        Rotates the rail shape property (north_south, east_west, etc.) around the vertical y-axis.
        """
        straight_shapes = ["north_south", "east_west"]
        if shape in straight_shapes:
            idx = straight_shapes.index(shape)
            return straight_shapes[(idx + k) % 2]
        ascending_shapes = [
            "ascending_north",
            "ascending_east",
            "ascending_south",
            "ascending_west",
        ]
        if shape in ascending_shapes:
            idx = ascending_shapes.index(shape)
            return ascending_shapes[(idx + k) % 4]
        curved_shapes = ["north_west", "north_east", "south_east", "south_west"]
        if shape in curved_shapes:
            idx = curved_shapes.index(shape)
            return curved_shapes[(idx + k) % 4]
        return shape

    @staticmethod
    def flip_block_properties(block: Block) -> Block:
        """
        Flips the block-state properties along the z-axis.
        """
        props: dict = block.properties

        # Flip 'facing' (e.g. 'north' -> 'south', etc.)
        if "facing" in props:
            direction = str(props["facing"])
            flipped_direction = StructureTransformer._flip_direction(direction)
            props["facing"] = StringTag(flipped_direction)

        if "rail" in block.base_name:
            shape = str(props["shape"])
            flipped_shape = StructureTransformer._flip_rail_shape(shape)
            props["shape"] = StringTag(flipped_shape)

        if block.base_name == "door":
            hinge = str(props["hinge"])
            hinge = "right" if hinge == "left" else "left"
            props["hinge"] = StringTag(hinge)

        if block.base_name == "stairs":
            shape = str(props["shape"])
            if "left" in shape:
                shape = shape.replace("left", "right")
            elif "right" in shape:
                shape = shape.replace("right", "left")
            props["shape"] = StringTag(shape)

        if "rotation" in props:
            rotation_value = int(str(props["rotation"]))
            flipped_rotation = (8 - rotation_value) % 16
            props["rotation"] = StringTag(str(flipped_rotation))

        if block.base_name == "chest" or block.base_name == "trapped_chest":
            connection = str(props["connection"])
            if connection == "left":
                connection = "right"
            elif connection == "right":
                connection = "left"
            props["connection"] = StringTag(connection)

        # Flip side-connection flags (e.g. fences, glass panes)
        side_keys = ["north", "south"]
        side_connections = {}
        for side in side_keys:
            # Pop them off if they exist
            if side in props:
                new_side = StructureTransformer._flip_direction(side)
                side_connections[new_side] = props.pop(side)

        # Put them back in the properties
        props.update(side_connections)

        flipped_block = Block(
            namespace=block.namespace,
            base_name=block.base_name,
            properties=props,
        )

        return flipped_block

    @staticmethod
    def _flip_direction(direction: str) -> str:
        """Flips 'north' -> 'south', etc."""
        if direction == "north":
            return "south"
        if direction == "south":
            return "north"
        return direction

    @staticmethod
    def _flip_rail_shape(shape: str) -> str:
        """
        Flips the rail shape property along the z-axis.
        """
        if "north" in shape:
            return shape.replace("north", "south")
        elif "south" in shape:
            return shape.replace("south", "north")
        return shape

    def _rotate_structure(self, structure: torch.Tensor, rotation: int) -> torch.Tensor:
        rotation = rotation % 4
        if rotation == 0:
            return structure

        # Perform the rotation
        structure = torch.rot90(structure, rotation, (2, 0))

        # Update the block tokens
        structure = self._rotation_luts[rotation - 1][structure]

        return structure

    def _flip_structure(self, structure: torch.Tensor) -> torch.Tensor:
        # Perform the flip
        structure = torch.flip(structure, (0,))

        # Update the block tokens
        structure = self._flip_lut[structure]

        return structure

    def transform_structure(self, structure: torch.Tensor) -> torch.Tensor:
        rotation = random.randint(1, 3)
        structure = self._rotate_structure(structure, rotation)

        if random.random() > 0.5:
            # Only flip along one axis since with the rotation it makes every variation
            structure = self._flip_structure(structure)

        return structure
