import hashlib
import logging
import random

import amulet
from tqdm import tqdm
from amulet.api.errors import ChunkDoesNotExist
from amulet.api.selection import SelectionBox, SelectionGroup
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper

logging.getLogger('amulet').setLevel(logging.WARNING)

chunk_size = 8
save_selection = SelectionGroup(SelectionBox(
    (0, 0, 0), (chunk_size, chunk_size, chunk_size)))
world_path = 'data/New World (12)'
world = amulet.load_level(world_path)
interested_blocks = set([
    'log',
    'grass_block',
    'leaves'
])
not_interested_blocks = set([
    'mushroom',
    'water',
    'ore',
    'sugar_cane',
    'lava',
    'pumpkin',
    'sand',
    'gravel',
    'diorite',
    'andesite',
    'granite',
    'fire',
    'ice'
])

processed_selections = set()


def get_random_selection():
    while True:
        start_positions = (random.randint(154, 512), random.randint(
            60, 70), random.randint(-222, 398))
        if start_positions not in processed_selections:
            break
    processed_selections.add(start_positions)
    end_positions = (start_positions[0] + chunk_size,
                     start_positions[1] + chunk_size, start_positions[2] + chunk_size)
    selection = SelectionBox(start_positions, end_positions)
    return selection


def check_selection(selection):
    found_blocks = set()
    # Iterate over the blocks in the selection
    for x, y, z in selection.blocks:
        try:
            block = world.get_block(x, y, z, 'minecraft:overworld')
        except ChunkDoesNotExist:
            return False
        for not_interested_block in not_interested_blocks:
            if not_interested_block in block.namespaced_name:
                return False
        for interested_block in interested_blocks:
            if interested_block in block.namespaced_name:
                found_blocks.add(interested_block)
                break

    # Check if we have found at least one of each interested block
    return found_blocks == interested_blocks


files_bar = tqdm(range(10000), desc='Collecting samples')
total_attempts = 0
rejections = 0


def get_random_structure():
    global total_attempts
    global rejections
    while True:
        total_attempts += 1
        selection = get_random_selection()
        if check_selection(selection):
            return world.extract_structure(selection, 'minecraft:overworld')
        else:
            rejections += 1


for i in files_bar:
    structure = get_random_structure()
    filename = f'selection__{i}'
    file_hash = hashlib.sha256(filename.encode()).hexdigest()
    wrapper = SpongeSchemFormatWrapper(
        f'data/schematics/natural_gen_trees/{file_hash}.schem')
    wrapper.create_and_open(
        'java', 3578, bounds=SelectionGroup(structure.bounds('minecraft:overworld')), overwrite=True)
    structure.save(wrapper)
    wrapper.close()
    rejection_rate = (rejections / total_attempts) * 100
    files_bar.set_postfix_str(f"Rejection Rate: {rejection_rate:.2f}%")

# Remember to close the world when done
world.close()
