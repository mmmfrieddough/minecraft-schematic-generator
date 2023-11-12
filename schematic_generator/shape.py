import random
from litemapy import Region, BlockState

# Constants for magic strings
MINECRAFT_AIR = 'minecraft:air'
MINECRAFT_BLOCK_START = 'minecraft:'
RANDOM = 'random'
SPHERE = 'sphere'
CUBE = 'cube'

def block_id_to_name(block_id):
    if block_id.startswith(MINECRAFT_BLOCK_START):
        block_id = block_id[len(MINECRAFT_BLOCK_START):]
    return block_id.replace('_', ' ')

def block_ids_to_names(block_ids):
    names = [block_id_to_name(id) for id in block_ids]
    if len(names) > 2:
        return f'{", ".join(names[:-1])}, and {names[-1]}'
    elif len(names) > 1:
        return f'{names[0]} and {names[1]}'
    else:
        return names[0]

def get_structure_size_description(properties):
    shape_type = properties.get('shape_type')
    if shape_type == SPHERE:
        radius = properties.get('radius')
        return f'a radius of {radius} block{"s" if radius > 1 else ""}'
    elif shape_type == CUBE:
        side_length = properties.get('side_length')
        return f'a side length of {side_length} block{"s" if side_length > 1 else ""}'
    else:
        raise ValueError(f"Invalid shape type: {shape_type}")
    
def get_composition_description(block_types):
    block_types = list(set(block_types))
    ending = ''
    if 'minecraft:air' in block_types:
        block_types.remove('minecraft:air')
        ending = ', interspersed with pockets of air'
    return block_ids_to_names(block_types) + ending

def generate_description(properties):
    # It is considered hollow only if it is filled with air, otherwise it is solid
    structure_block_types = properties.get('structure_block_types')
    structure_fill_block_types = properties.get('structure_fill_block_types')
    if structure_fill_block_types and len(structure_fill_block_types) == 1 and structure_fill_block_types[0] == MINECRAFT_AIR:
        hollow = 'hollow'
    else:
        hollow = 'solid'

    shape_type = properties.get('shape_type')
    dimensions = get_structure_size_description(properties)

    # We only describe the layer if it is different from the structure blocks
    thickness = properties.get('thickness')
    if structure_fill_block_types and structure_fill_block_types != structure_block_types:
        layer = f'a {thickness} block thick layer of '
    else:
        layer = ''

    structure_composition = get_composition_description(structure_block_types)

    # We only describe the filling if it is different from the structure blocks and not air
    if structure_fill_block_types and structure_fill_block_types != structure_block_types and structure_fill_block_types != [MINECRAFT_AIR]:
        filling = f' and filled with {block_ids_to_names(structure_fill_block_types)}'
    else:
        filling = ''

    # It is considered floating only if it is floating in air, otherwise it is embedded
    background_block_types = properties.get('background_block_types')
    if len(background_block_types) == 1 and background_block_types[0] == MINECRAFT_AIR:
        position = 'floating'
        background_composition = 'an empty void'
    else:
        position = 'embedded'
        background_composition = get_composition_description(background_block_types)

    description = f'A {hollow} {shape_type} with {dimensions}. It is composed of {layer}{structure_composition}{filling}. It is {position} within {background_composition}.'
    return description

def calculate_start_position(region_size, shape_dimensions, position_percentages):
    """
    Calculate the start position for a shape within a region based on percentage offsets.
    This function assumes that the shape is centered on the start position.

    :param region_size: Tuple of (width, height, depth) of the region.
    :param shape_dimensions: Tuple of dimensions of the shape.
    :param position_percentages: Tuple of percentages (x_percent, y_percent, z_percent).
    :return: Tuple of (x, y, z) start position for the shape.
    """
    start_position = []
    for region_dim, shape_dim, percent in zip(region_size, shape_dimensions, position_percentages):
        # Calculate the maximum offset to keep the shape within the region bounds
        max_offset = (region_dim - shape_dim) // 2 - 1

        # Account for even/odd dimensions
        if region_dim % 2 != shape_dim % 2 and percent < 0:
            max_offset += 1

        # Apply the percentage to calculate the actual offset
        offset = int(max_offset * (percent / 100.0))

        # Calculate the start position
        start_pos = (region_dim // 2) + offset
        start_position.append(start_pos)

    return tuple(start_position)

def generate_shape(reg, start_pos, shape_type, size, block_types, is_inner=False):
    pos_x, pos_y, pos_z = start_pos
    blocks = [BlockState(block_type) for block_type in block_types]
    if shape_type == SPHERE:
        radius = size
        for x, y, z in reg.allblockpos():
            distance = round(((x - pos_x) ** 2 + (y - pos_y) ** 2 + (z - pos_z) ** 2) ** .5)
            if (is_inner and distance <= radius) or (not is_inner and distance <= radius):
                reg.setblock(x, y, z, random.choice(blocks))
    elif shape_type == CUBE:
        half_side = size / 2
        for x, y, z in reg.allblockpos():
            if (pos_x - half_side) <= x < (pos_x + half_side) and \
            (pos_y - half_side) <= y < (pos_y + half_side) and \
            (pos_z - half_side) <= z < (pos_z + half_side):
                reg.setblock(x, y, z, random.choice(blocks))

def generate_schematic(properties):
    # Unpack properties
    shape_type = properties.get('shape_type')
    region_size = properties.get('region_size')
    random_seed = properties.get('random_seed')
    radius = properties.get('radius')
    side_length = properties.get('side_length')
    structure_block_types = properties.get('structure_block_types')
    fill_block_types = properties.get('structure_fill_block_types')
    thickness = properties.get('thickness')

    # Set random seed
    random.seed(random_seed)
    
    # Unpack region size and position
    width, height, depth = region_size

    # Shortcut to create a schematic with a single region
    reg = Region(0, 0, 0, width, height, depth)
    schem = reg.as_schematic(name="SimpleShape", author="mmmfrieddough", description="Simple shape intended for training ML models")

    # Determine the background type and create the appropriate blocks
    background_block_types = properties.get('background_block_types', [])
    background_blocks = [BlockState(block_type) for block_type in background_block_types]

    # Fill the region with the background blocks if not air
    if len(background_block_types) != 1 or background_block_types[0] != MINECRAFT_AIR:
        for x, y, z in reg.allblockpos():
            reg.setblock(x, y, z, random.choice(background_blocks))

    # Get the user-defined position percentages with default to center if not provided
    position_percentages = properties.get('position_offset')

    # Calculate the dimensions of the shape
    if shape_type == SPHERE:
        shape_dimensions = (radius * 2 + 1, radius * 2 + 1, radius * 2 + 1)
    elif shape_type == CUBE:
        shape_dimensions = (side_length, side_length, side_length)

    # Calculate the start position for the shape
    start_pos = calculate_start_position(region_size, shape_dimensions, position_percentages)

    # Generate outer shape
    generate_shape(reg, start_pos, shape_type, radius if shape_type == SPHERE else side_length, structure_block_types)

    # Build the inner shape if fill block types are provided
    if fill_block_types and thickness > 0:
        inner_size = radius - thickness if shape_type == SPHERE else side_length - 2 * thickness
        generate_shape(reg, start_pos, shape_type, inner_size, fill_block_types, is_inner=True)

    return schem
