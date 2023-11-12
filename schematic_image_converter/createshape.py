from litemapy import Schematic, Region, BlockState


def create_shape(shape_type, radius, block_type, position, region_size, schematic_path):
    """
    Create a shape in a Minecraft schematic.

    :param shape_type: The type of shape ('sphere' or 'cube').
    :param radius: The radius of the shape. For cubes, this is half the side length.
    :param block_type: The type of block (e.g., 'minecraft:stone').
    :param position: The center position of the shape (tuple of x, y, z).
    :param region_size: The size of the region (tuple of width, height, depth).
    :param schematic_path: Path to save the schematic.
    """
    # Unpack region size and position
    width, height, depth = region_size
    pos_x, pos_y, pos_z = position

    # Shortcut to create a schematic with a single region
    reg = Region(0, 0, 0, width, height, depth)
    schem = reg.as_schematic(
        name="CustomShape", author="SmylerMC", description="Made with litemapy")

    # Create the block state we are going to use
    block = BlockState(block_type)

    # Build the shape
    for x, y, z in reg.allblockpos():
        if shape_type == 'sphere':
            if round(((x - pos_x) ** 2 + (y - pos_y) ** 2 + (z - pos_z) ** 2) ** .5) <= radius:
                reg.setblock(x, y, z, block)
        elif shape_type == 'cube':
            if (pos_x - radius) <= x <= (pos_x + radius) and \
               (pos_y - radius) <= y <= (pos_y + radius) and \
               (pos_z - radius) <= z <= (pos_z + radius):
                reg.setblock(x, y, z, block)

    # Save the schematic
    schem.save(schematic_path)


# Example usage
create_shape('sphere', 31, 'minecraft:light_blue_concrete',
             (32, 32, 32), (64, 64, 64), "sphere.litematic")
create_shape('cube', 15, 'minecraft:gold_block',
             (32, 32, 32), (64, 64, 64), "cube.litematic")
