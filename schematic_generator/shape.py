import itertools
import random

from schempy import Block, Schematic

# Constants for magic strings
MINECRAFT_AIR = 'minecraft:air'
MINECRAFT_BLOCK_START = 'minecraft:'
RANDOM = 'random'
SPHERE = 'sphere'
CUBE = 'cube'


def blocks_to_names(blocks, plural: bool = False) -> str:
    names = [block['names']['plural' if plural else 'singular'][0]
             for block in blocks]
    if len(names) > 2:
        return f'{", ".join(names[:-1])}, and {names[-1]}'
    elif len(names) > 1:
        return f'{names[0]} and {names[1]}'
    else:
        return names[0]


def get_dimensions_description_simple(properties):
    shape_type = properties.get('shape_type')
    length = properties.get(
        'radius') if shape_type == SPHERE else properties.get('side_length')
    return f'{length} block{"s" if length > 1 else ""}'


def get_dimensions_description(properties, description_simple):
    shape_type = properties.get('shape_type')
    if shape_type == SPHERE:
        return f'a radius of {description_simple}'
    elif shape_type == CUBE:
        return f'a side length of {description_simple}'
    else:
        raise ValueError(f"Invalid shape type: {shape_type}")


def get_composition_descriptions(block_types: list[dict], suffix: bool = False) -> list[str]:
    ending = ''
    for block_type in block_types:
        if block_type['id'] == MINECRAFT_AIR:
            block_types.remove(block_type)
            ending = ', interspersed with pockets of air'
            break
    if suffix:
        names = blocks_to_names(block_types, True)
        descriptions = [
            f'made of {names}' + ending,
            f'composed of {names}' + ending,
            f'constructed of {names}' + ending
        ]
        if len(block_types) == 1:
            descriptions.extend([
                f'made entirely of {names}' + ending,
                f'composed entirely of {names}' + ending,
                f'constructed entirely of {names}' + ending
            ])
        else:
            descriptions.extend([
                f'made of a mixture of {names}' + ending,
                f'composed of a mixture of {names}' + ending,
                f'constructed of a mixture of {names}' + ending
            ])
    else:
        names = blocks_to_names(block_types, False)
        descriptions = [
            f'{names}' + ending,
            f'{names}' + ending,
            f'{names}' + ending
        ]
    return descriptions


def generate_permutations(pattern):
    for r in range(0, len(pattern) + 1):
        for perm in itertools.permutations(pattern, r):
            yield perm


def insert_permutations(sequence, elements):
    sequence = list(sequence)
    for num_elements in range(len(elements) + 1):
        for element_subset in itertools.permutations(elements, num_elements):
            for positions in itertools.combinations(range(len(sequence) + num_elements), num_elements):
                temp_sequence = sequence[:]
                for element, pos in sorted(zip(element_subset, positions), key=lambda x: x[1]):
                    temp_sequence.insert(pos, element)
                yield tuple(temp_sequence)


def generate_prefix_combinations(element_list):
    element_list[0] = ([''], element_list[0][1])
    transformed_elements = []
    for element in element_list:
        transformed_elements.append([' '.join(item)
                                    for item in itertools.product(*element)])
    for combination in itertools.product(*transformed_elements):
        yield ''.join(combination)


def generate_combinations(pattern):
    for prefix_perm in generate_permutations(pattern['prefixes']):
        for postfix_perm in generate_permutations(pattern['suffixes']):
            base_postfix_tuple = prefix_perm + \
                ((pattern['base'],)) + postfix_perm
            for ambifix_combination in insert_permutations(base_postfix_tuple, pattern['ambifixes']):
                cleaned_elements = []
                base_index = ambifix_combination.index(pattern['base'])
                for index, element in enumerate(ambifix_combination):
                    if isinstance(element, tuple):
                        if len(element) == 0:
                            continue
                        if isinstance(element[0], tuple):
                            element = element[index > base_index]
                        cleaned_elements.append(element)
                    else:
                        cleaned_elements.append(([element], ['']))
                for prefix_combination in generate_prefix_combinations(cleaned_elements):
                    start = pattern['start']
                    start = start[1] if prefix_combination[1].lower(
                    ) in 'aeiou8' else start[0]
                    yield start + prefix_combination + '.'


def generate_descriptions(properties: dict) -> list[str]:
    num_descriptions = properties['descriptions']
    if num_descriptions <= 0:
        return []

    structure_block_types = properties['structure_block_types']
    structure_fill_block_types = properties.get(
        'structure_fill_block_types', [])
    background_block_types = properties.get('background_block_types', [])
    shape_type = properties['shape_type']

    is_hollow = len(
        structure_fill_block_types) == 1 and structure_fill_block_types[0]['id'] == MINECRAFT_AIR
    hollow = 'hollow' if is_hollow else 'solid'
    composition_descriptions_prefix = get_composition_descriptions(
        structure_block_types, False)
    composition_descriptions_suffix = get_composition_descriptions(
        structure_block_types, True)
    contains_air = any(
        block['id'] == MINECRAFT_AIR for block in structure_block_types)
    background_contains_air = any(
        block['id'] == MINECRAFT_AIR for block in background_block_types)
    perfection = ['imperfect', 'flawed', 'imperfectly constructed', 'flawed in its construction'] if contains_air else [
        'perfect', 'immaculate', 'flawless', 'pristine', 'unblemished', 'impeccably constructed', 'meticulously crafted', 'perfect in its construction']
    geometry_prefix = [
        'imperfectly geometric' if contains_air else 'perfectly geometric']
    geometry_suffix = ['imperfect in geometry', 'imperfectly geometric'] if contains_air else [
        'perfect in geometry', 'perfectly geometric']

    if shape_type == CUBE:
        length = properties['radius'] if shape_type == SPHERE else properties['side_length']
        dimensions_prefix = [f'{length} block wide', f'{length} meter wide', f'{length} block long', f'{length} meter long', f'{length} block tall', f'{length} meter tall', f'{length} block high',
                             f'{length} meter high', f'{length} block thick', f'{length} meter thick', f'{length} block deep', f'{length} meter deep', f'{length}x{length}x{length}', f'{length} by {length} by {length}']
        dimensions_suffix = [f'a side length of {length} blocks', f'a side length of {length} meters',
                             f'{length} blocks per side', f'{length} meters per side']
    elif shape_type == SPHERE:
        radius = properties['radius']
        diameter = radius * 2 + 1
        dimensions_prefix = [f'{radius} block radius', f'{radius} meter radius', f'{diameter} block diameter',
                             f'{radius} meter diameter', f'{diameter} block wide', f'{diameter} meter wide']
        dimensions_suffix = [f'a radius of {radius} block' if radius == 1 else f'a radius of {radius} blocks', f'a radius of {radius} meter' if radius ==
                             1 else f'a radius of {radius} meters', f'a diameter of {diameter} blocks', f'a diameter of {diameter} meters']
    else:
        raise ValueError(
            f"Dimensions not defined for shape type: {shape_type}")

    if background_contains_air:
        background = ['floating within an empty void', 'floating immaculately in a void', 'floating in a void', 'floating in an empty void',
                      'set against the backdrop of an empty void', 'ethereally floating in an expansive void', 'floating in the air']
    else:
        raise ValueError('Non-air background not implemented yet')

    base_pattern = {
        'start': ('A', 'An'),
        'base': ([''], [shape_type]),
        'prefixes': [
            ([','], perfection[:1]),
            ([','], [hollow])
        ],
        'ambifixes': [
            (([','], dimensions_prefix[:1]),
             ([', with', '. It has'], dimensions_suffix[:1])),
            (([','], composition_descriptions_prefix[:1]),
             ([',', '. It is'], composition_descriptions_suffix[:1]))
        ],
        'suffixes': [
            ([',', '. It is'], background[:1])
        ]
    }

    condition_pattern = {
        'start': ('A', 'An'),
        'base': ([''], [shape_type]),
        'prefixes': [
            ([','], perfection),
            ([','], [hollow])
        ],
        'ambifixes': [
            (([','], composition_descriptions_prefix[:1]),
             ([','], composition_descriptions_suffix[:1])),
            (([','], geometry_prefix),
             ([','], geometry_suffix))
        ],
        'suffixes': [
            ([','], background[:1])
        ]
    }

    size_pattern = {
        'start': ('A', 'An'),
        'base': ([''], [shape_type]),
        'prefixes': [
            ([','], [hollow])
        ],
        'ambifixes': [
            (([','], dimensions_prefix),
             ([', with', '. It has'], dimensions_suffix)),
            (([','], composition_descriptions_prefix[:1]),
             ([',', '. It is'], composition_descriptions_suffix[:1])),
        ],
        'suffixes': [
            ([',', '. It is'], background[:1])
        ]
    }

    background_pattern = {
        'start': ('A', 'An'),
        'base': ([''], [shape_type]),
        'prefixes': [
            ([','], [hollow])
        ],
        'ambifixes': [
            (([','], dimensions_prefix[:1]),
             ([', with', '. It has'], dimensions_suffix[:1])),
            (([','], composition_descriptions_prefix[:1]),
             ([',', '. It is'], composition_descriptions_suffix[:1])),
        ],
        'suffixes': [
            ([',', '. It is'], background)
        ]
    }

    composition_pattern = {
        'start': ('A', 'An'),
        'base': ([''], [shape_type]),
        'prefixes': [
            ([','], [hollow])
        ],
        'ambifixes': [
            (([','], dimensions_prefix[:1]),
             ([', with', '. It has'], dimensions_suffix[:1])),
            (([','], composition_descriptions_prefix),
             ([',', '. It is'], composition_descriptions_suffix)),
        ],
        'suffixes': [
            ([',', '. It is'], background[:1])
        ]
    }

    patterns = [base_pattern, condition_pattern,
                size_pattern, background_pattern, composition_pattern]
    descriptions = []
    for pattern in patterns:
        combinations = list(generate_combinations(pattern))
        combinations = random.sample(combinations, min(
            num_descriptions // len(patterns), len(combinations)))
        descriptions.extend(combinations)
    return descriptions


def calculate_start_position(region_size: tuple[int], position_offset: tuple[int]) -> tuple[int]:
    """
    Calculate the start position for a shape within a region.
    This function assumes that the shape is centered on the start position.

    :param region_size: Tuple of (width, height, depth) of the region.
    :param position_offset: Tuple of offsets (x_offset, y_offset, z_offset).
    :return: Tuple of (x, y, z) start position for the shape.
    """
    start_position = []
    for region_dim, offset in zip(region_size, position_offset):
        # Calculate the start position
        start_pos = (region_dim // 2) + offset
        start_position.append(start_pos)

    return tuple(start_position)


def generate_shape(schematic: Schematic, start_pos: tuple[int], shape_type: str, size: int, block_types: list[str], is_inner: bool = False) -> None:
    pos_x, pos_y, pos_z = start_pos
    blocks = [Block(block_type) for block_type in block_types]
    if shape_type == SPHERE:
        radius = size
        for x, y, z in schematic.iter_block_positions():
            distance = round(((x - pos_x) ** 2 + (y - pos_y)
                             ** 2 + (z - pos_z) ** 2) ** .5)
            if (is_inner and distance <= radius) or (not is_inner and distance <= radius):
                schematic.set_block(x, y, z, random.choice(blocks))
    elif shape_type == CUBE:
        half_side = size / 2
        for x, y, z in schematic.iter_block_positions():
            if (pos_x - half_side) <= x < (pos_x + half_side) and \
                (pos_y - half_side) <= y < (pos_y + half_side) and \
                    (pos_z - half_side) <= z < (pos_z + half_side):
                schematic.set_block(x, y, z, random.choice(blocks))


def generate_schematic(properties):
    # Unpack properties
    shape_type = properties.get('shape_type')
    region_size = properties.get('region_size')
    random_seed = properties.get('random_seed')
    radius = properties.get('radius')
    side_length = properties.get('side_length')
    structure_block_types = [block_type['id']
                             for block_type in properties['structure_block_types']]
    fill_block_types = [block_type['id'] for block_type in properties.get(
        'structure_fill_block_types', [])]
    thickness = properties.get('thickness')

    # Set random seed
    random.seed(random_seed)

    # Unpack region size and position
    width, height, depth = region_size

    # Create schematic
    schematic = Schematic(width, height, depth)

    # Determine the background type and create the appropriate blocks
    background_block_types = properties.get('background_block_types', [])
    background_blocks = [Block(block_type['id'])
                         for block_type in background_block_types]

    # Fill the region with the background blocks if not air
    if len(background_block_types) != 1 or background_block_types[0] != MINECRAFT_AIR:
        for x, y, z in schematic.iter_block_positions():
            schematic.set_block(x, y, z, random.choice(background_blocks))

    # Calculate the start position for the shape
    start_pos = calculate_start_position(
        region_size, properties['position_offset'])

    # Generate outer shape
    generate_shape(schematic, start_pos, shape_type, radius if shape_type ==
                   SPHERE else side_length, structure_block_types)

    # Build the inner shape if fill block types are provided
    if fill_block_types and thickness > 0:
        inner_size = radius - thickness if shape_type == SPHERE else side_length - 2 * thickness
        generate_shape(schematic, start_pos, shape_type,
                       inner_size, fill_block_types, is_inner=True)

    return schematic
