import logging

logging.getLogger('amulet').setLevel(logging.WARNING)  # noqa
logging.getLogger('PyMCTranslate').setLevel(logging.WARNING)  # noqa

import hashlib
import json
import os
import shutil
import time
from multiprocessing import Process, Queue
from queue import Empty

import amulet
import numpy as np
from amulet.api.chunk import Chunk
from amulet.api.level import World
from amulet.api.selection import SelectionBox, SelectionGroup
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper
from amulet.utils.world_utils import chunk_coords_to_block_coords
from tqdm import tqdm


class WorldSampler:
    glass = [
        'white_stained_glass|',
        'orange_stained_glass|',
        'magenta_stained_glass',
        'light_blue_stained_glass',
        'yellow_stained_glass|',
        'lime_stained_glass',
        'pink_stained_glass',
        'gray_stained_glass',
        'light_gray_stained_glass',
        'cyan_stained_glass',
        'purple_stained_glass',
        'blue_stained_glass',
        'brown_stained_glass|',
        'green_stained_glass',
        'red_stained_glass|',
        'black_stained_glass'
    ]
    wood = [
        'mangrove_planks',
        'cherry_planks',
        'bamboo_planks',
        'mangrove_fence',
        'cherry_fence',
        'bamboo_fence',
        'birch_fence_gate',
        'mangrove_fence_gate',
        'cherry_fence_gate',
        'bamboo_fence_gate',
        'stripped_birch_log',
        'stripped_mangrove_log',
        'stripped_cherry_log',
        'stripped_acacia_wood',
        'birch_wood',
        'jungle_wood',
        'dark_oak_wood',
        'mangrove_wood',
        'cherry_wood'
    ]
    redstone = [
        'hopper',
        'observer',
        'dropper',
        'light_weighted_pressure_plate',
        'heavy_weighted_pressure_plate',
        'polished_blackstone_pressure_plate',
        'birch_pressure_plate',
        'jungle_pressure_plate',
        'dark_oak_pressure_plate',
        'mangrove_pressure_plate',
        'cherry_pressure_plate',
        'bamboo_pressure_plate'
    ]
    slabs = [
        'mangrove_slab',
        'cherry_slab',
        'bamboo_slab',
        'bamboo_mosiac_slab',
        'granite_slab',
        'diorite_slab',
        'andesite_slab',
        'nether_brick_slab',
        'cut_sandstone_slab',
        'red_sandstone_slab',
        'prismarine_slab',
    ]
    stairs = [
        'mangrove_stairs',
        'cherry_stairs',
        'bamboo_stairs',
        'bamboo_mosiac_stairs',
        ':stone_stairs',
        'granite_stairs',
        'diorite_stairs',
        'andesite_stairs',
        'red_sandstone_stairs',
        'prismarine_stairs',
        'prismarine_brick_stairs'
    ]
    wool = [
        'magenta_wool',
        'pink_wool',
        'purple_wool',
    ]
    anvil = [
        ':anvil',
        'chipped'
    ]
    copper = [
        'weathered',
        'exposed',
        'oxidized',
        'waxed',
        ':copper_block'
    ]
    universal_man_made_blocks = [
        'beacon',
        'scaffolding',
        'concrete',
        'slime',
        'rod',
        ':iron_block',
        'honey',
        'enchant',
        'daylight_detector',
        'cake',
        'soul_campfire',
        'shulker',
    ] + glass + wood + redstone + slabs + stairs + wool + anvil + copper
    overworld_man_made_blocks = [
        'nether_brick',
        'glowstone',
        'soulsand',
        'warped',
        'crimson',
        'quartz',
        'purpur',
        'end_stone',
        'shroomlight',
        'blackstone',
        'chorus',
        'wart'
    ]
    nether_man_made_blocks = ['campfire']
    chunk_target_blocks = universal_man_made_blocks + overworld_man_made_blocks
    overworld_possible_man_made_blocks = [
        'glass',
        'farmland',
        'door',
        'bricks',
        'chest',
        'crafting_table',
        'furnace',
        'ladder',
        'bed|',
        'stairs',
        'slab',
        'wool',
        'anvil',
        'hay_block',
        'redstone_wire',
        'redstone_torch',
        'repeater',
        'comparator',
        'piston',
        'glazed',
        'prismarine',
        'lantern',
        'sign',
        'lever',
        'button',
        'smooth_stone',
        'dirt_path'
    ]
    sample_target_blocks = chunk_target_blocks + overworld_possible_man_made_blocks

    def __init__(self,
                 schematic_directory,
                 temp_directory,
                 chunk_progress_save_interval,
                 chunk_mark_radius,
                 sample_offset,
                 sample_size,
                 sample_interested_block_threshold,
                 sample_progress_save_interval,
                 sampling_purge_interval,
                 num_workers=os.cpu_count()):
        self.schematic_directory = schematic_directory
        self.temp_directory = temp_directory
        self.chunk_progress_save_interval = chunk_progress_save_interval
        self.chunk_mark_radius = chunk_mark_radius
        self.sample_offset = sample_offset
        self.sample_size = sample_size
        self.sample_interested_block_threshold = sample_interested_block_threshold
        self.sample_progress_save_interval = sample_progress_save_interval
        self.sampling_purge_interval = sampling_purge_interval
        self.num_workers = num_workers

        self.worker_directories = None

    def setup_worker_directories(self, src_directory: str) -> None:
        """Sets up worker directories for a given source directory"""
        if self.worker_directories:
            return
        self.worker_directories = {}

        # Get the parent directory and the name of the source directory
        _, src_dir_name = os.path.split(src_directory)

        # Create a new directory next to the source directory to hold the copies
        copies_directory = os.path.join(self.temp_directory, src_dir_name)
        os.makedirs(copies_directory, exist_ok=True)

        # Create a copy of the source directory for each worker
        pbar = tqdm(range(self.num_workers),
                    desc="Setting up worker directories")
        for i in pbar:
            worker_directory_name = f"worker_{i}"
            worker_directory = os.path.join(
                copies_directory, worker_directory_name)
            if not os.path.exists(worker_directory):
                shutil.copytree(src_directory, worker_directory)
            self.worker_directories[i] = worker_directory

    def _check_block(self, block, target_blocks, translator):
        """Returns True if the block is one of the target blocks"""
        block, _, _ = translator.block.from_universal(block)
        name = block.namespaced_name + '|'
        for target_block in target_blocks:
            if target_block in name:
                return True
        return False

    def _chunk_contains_target_blocks(self, chunk, target_blocks, translator):
        """Returns True if the chunk contains any of the target blocks"""
        for block in chunk.block_palette:
            if self._check_block(block, target_blocks, translator):
                return True
        return False

    def _save_chunk_progress(self, directory, visited_chunks, relevant_chunks):
        """Save the current chunk progress to a file"""
        temp_file_path = os.path.join(directory, 'chunk_progress_temp.json')
        final_file_path = os.path.join(directory, 'chunk_progress.json')
        with open(temp_file_path, 'w') as file:
            json.dump({'visited_chunks': list(visited_chunks),
                       'relevant_chunks': list(relevant_chunks)}, file)
        os.replace(temp_file_path, final_file_path)

    def _load_chunk_progress(self, directory):
        """Load the current chunk progress from a file"""
        path = os.path.join(directory, 'chunk_progress.json')
        if os.path.exists(path):
            with open(path, 'r') as file:
                data = json.load(file)
            visited_chunks = set(tuple(c) for c in data['visited_chunks'])
            relevant_chunks = set(tuple(c) for c in data['relevant_chunks'])
        else:
            visited_chunks = set()
            relevant_chunks = set()
        return visited_chunks, relevant_chunks

    def _save_sample_progress(self, directory, sampled_chunks, sample_positions):
        """Save the current sample progress to a file"""
        with open(os.path.join(directory, 'sample_progress.json'), 'w') as file:
            json.dump({'sampled_chunks': list(sampled_chunks),
                      'sample_positions': list(sample_positions)}, file)

    def _load_sample_progress(self, directory):
        """Load the current sample progress from a file"""
        path = os.path.join(directory, 'sample_progress.json')
        if os.path.exists(path):
            with open(path, 'r') as file:
                data = json.load(file)
            sampled_chunks = set(tuple(c) for c in data['sampled_chunks'])
            sample_positions = set(tuple(c) for c in data['sample_positions'])
        else:
            sampled_chunks = set()
            sample_positions = set()
        return sampled_chunks, sample_positions

    def _mark_chunks_worker(self, directory: str, all_chunks_queue: Queue, visited_chunks_queue: Queue, relevant_chunks_queue: Queue) -> None:
        """Worker function for marking chunks"""
        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)
            translator = world.translation_manager.get_version(
                'java', (1, 20, 4))

            while True:
                chunk_coords = all_chunks_queue.get()
                if chunk_coords is None:
                    all_chunks_queue.put(None)
                    break

                chunk = world.level_wrapper.load_chunk(
                    *chunk_coords, 'minecraft:overworld')
                if self._chunk_contains_target_blocks(chunk, self.chunk_target_blocks, translator):
                    # Add this chunk and its neighbors
                    for dx in range(-self.chunk_mark_radius, self.chunk_mark_radius + 1):
                        for dz in range(-self.chunk_mark_radius, self.chunk_mark_radius + 1):
                            relevant_chunks_queue.put(
                                (chunk_coords[0] + dx, chunk_coords[1] + dz))
                visited_chunks_queue.put(chunk_coords)
        finally:
            world.close()

    def _mark_chunks(self, directory: str) -> None:
        """Looks through chunks in a world and marks them as relevant or not relevant"""

        # Load progress
        visited_chunks, relevant_chunks = self._load_chunk_progress(directory)

        # Get all chunk coordinates and convert to a list
        world = amulet.load_level(directory)
        all_chunk_coords = world.all_chunk_coords('minecraft:overworld')
        world.close()

        # Remove visited chunks from the list
        remaining_chunk_coords = all_chunk_coords - visited_chunks
        if len(remaining_chunk_coords) == 0:
            print(
                f"All {len(all_chunk_coords)} chunks have already been visited")
            return

        # Set up worker directories
        self.setup_worker_directories(directory)

        # Create a tqdm progress bar
        pbar = tqdm(
            total=len(all_chunk_coords),
            initial=len(visited_chunks),
            desc="Marking chunks"
        )
        pbar.set_postfix({"relevant_chunks": len(relevant_chunks)})

        # Create a queue and add all chunk coordinates to it
        all_chunks_queue = Queue()
        for chunk_coords in remaining_chunk_coords:
            all_chunks_queue.put(chunk_coords)
        all_chunks_queue.put(None)

        # Create output queues
        visited_chunks_queue = Queue()
        relevant_chunks_queue = Queue()

        processes = []
        try:
            # Create and start worker processes
            for i in range(self.num_workers):
                process = Process(target=self._mark_chunks_worker, args=(
                    self.worker_directories[i], all_chunks_queue, visited_chunks_queue, relevant_chunks_queue))
                process.start()
                processes.append(process)

            # Update the progress bar based on the progress queue
            while any(p.is_alive() for p in processes):
                while not visited_chunks_queue.empty():
                    pbar.update(1)
                    visited_chunks.add(visited_chunks_queue.get())
                    if len(visited_chunks) % self.chunk_progress_save_interval == 0:
                        self._save_chunk_progress(
                            directory, visited_chunks, relevant_chunks)
                while not relevant_chunks_queue.empty():
                    relevant_chunks.add(relevant_chunks_queue.get())
                    pbar.set_postfix({"relevant_chunks": len(relevant_chunks)})
                time.sleep(0.1)

            # Wait for all processes to finish
            for process in processes:
                process.join()
        finally:
            # Empty the queue
            while True:
                try:
                    all_chunks_queue.get_nowait()
                except Empty:
                    break

        # Final save
        self._save_chunk_progress(directory, all_chunk_coords, relevant_chunks)

    def _get_interested_palette_indices(self, chunk, translator):
        """Returns a set of indices of blocks from the chunk palette that we are interested in"""
        interested_indices = set()
        for i, block in enumerate(chunk.block_palette):
            if self._check_block(block, self.sample_target_blocks, translator):
                interested_indices.add(i)
        return interested_indices

    def _identify_samples_in_chunk(self, world: World, translator, chunk_coords) -> set:
        """Collects samples from a chunk"""

        sample_positions = set()

        # Load all chunks that a selection starting in this chunk could possibly intersect
        num_chunks = self.sample_size // 16 + 2
        chunk_blocks = [[None for _ in range(
            num_chunks)] for _ in range(num_chunks)]
        interested_indices = [[None for _ in range(
            num_chunks)] for _ in range(num_chunks)]
        for dx in range(num_chunks):
            for dz in range(num_chunks):
                inner_chunk_coords = (
                    chunk_coords[0] + dx, chunk_coords[1] + dz)
                if world.has_chunk(*inner_chunk_coords, 'minecraft:overworld'):
                    chunk = world.level_wrapper.load_chunk(
                        *inner_chunk_coords, 'minecraft:overworld')
                else:
                    chunk = Chunk(*inner_chunk_coords)
                chunk_blocks[dx][dz] = np.asarray(chunk.blocks[:, -64:320, :])
                interested_indices[dx][dz] = self._get_interested_palette_indices(
                    chunk, translator)

        # Precompute interesting block positions
        max_pos = 15 // self.sample_offset * self.sample_offset + self.sample_size
        y_size = 384
        x_start, z_start = chunk_coords_to_block_coords(*chunk_coords)
        block = np.zeros((max_pos, y_size, max_pos))
        for j in range(y_size):
            for i in range(max_pos):
                for k in range(max_pos):
                    chunk_x = i // 16
                    chunk_z = k // 16
                    blocks = chunk_blocks[chunk_x][chunk_z]
                    x = i % 16
                    y = j
                    z = k % 16
                    block_value = blocks[x, y, z]
                    chunk_interesting_indices = interested_indices[chunk_x][chunk_z]
                    block[i, j, k] = 1 if block_value in chunk_interesting_indices else 0
        marked_count = np.cumsum(
            np.cumsum(np.cumsum(block, axis=0), axis=1), axis=2)

        # Iterate through grid of possible selection start positions
        y_offset = (chunk_coords[0] + chunk_coords[1]) % self.sample_offset
        m = self.sample_size
        y_limit = y_size - m
        for i in range(0, 16, self.sample_offset):
            for j in range(y_offset, y_limit, self.sample_offset):
                for k in range(0, 16, self.sample_offset):
                    total_marked = marked_count[i +
                                                m - 1][j + m - 1][k + m - 1]

                    if i > 0:
                        total_marked -= marked_count[i -
                                                     1][j + m - 1][k + m - 1]
                    if j > 0:
                        total_marked -= marked_count[i +
                                                     m - 1][j - 1][k + m - 1]
                    if k > 0:
                        total_marked -= marked_count[i +
                                                     m - 1][j + m - 1][k - 1]

                    if i > 0 and j > 0:
                        total_marked += marked_count[i - 1][j - 1][k + m - 1]
                    if i > 0 and k > 0:
                        total_marked += marked_count[i - 1][j + m - 1][k - 1]
                    if j > 0 and k > 0:
                        total_marked += marked_count[i + m - 1][j - 1][k - 1]

                    if i > 0 and j > 0 and k > 0:
                        total_marked -= marked_count[i - 1][j - 1][k - 1]

                    if total_marked > self.sample_interested_block_threshold:
                        x = x_start + i
                        y = j - 64
                        z = z_start + k
                        sample_positions.add((x, y, z))

        return sample_positions

    def _identify_samples_worker(self, directory: str, relevant_chunks_queue: Queue, sampled_chunks_queue: Queue, sample_positions_queue: Queue) -> None:
        """Worker function for identifying samples"""

        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)
            translator = world.translation_manager.get_version(
                'java', (1, 20, 4))

            while True:
                chunk_coords = relevant_chunks_queue.get()
                if chunk_coords is None:
                    relevant_chunks_queue.put(None)
                    break

                if world.has_chunk(*chunk_coords, 'minecraft:overworld'):
                    samples = self._identify_samples_in_chunk(
                        world, translator, chunk_coords)
                    sample_positions_queue.put(samples)

                sampled_chunks_queue.put(chunk_coords)
        finally:
            world.close()

    def _identify_samples(self, directory: str) -> None:
        """Identifies samples from the marked chunks"""

        # Load progress
        _, relevant_chunks = self._load_chunk_progress(directory)
        sampled_chunks, sample_positions = self._load_sample_progress(
            directory)

        # Get all relevant chunks that have not been sampled
        remaining_relevant_chunks = relevant_chunks - sampled_chunks
        if len(remaining_relevant_chunks) == 0:
            print(
                f"All {len(relevant_chunks)} relevant chunks have already been sampled")
            return

        # Set up worker directories
        self.setup_worker_directories(directory)

        # Create a tqdm progress bar
        pbar = tqdm(
            total=len(relevant_chunks),
            initial=len(sampled_chunks),
            desc="Identifying samples from chunks"
        )
        pbar.set_postfix({"sample_positions": len(sample_positions)})

        # Create a queue and add all chunk coordinates to it
        relevant_chunks_queue = Queue()
        for chunk_coords in remaining_relevant_chunks:
            relevant_chunks_queue.put(chunk_coords)
        relevant_chunks_queue.put(None)

        # Create a progress queue
        sampled_chunks_queue = Queue()
        sample_positions_queue = Queue()

        processes = []
        try:
            # Create and start worker processes
            for i in range(self.num_workers):
                process = Process(target=self._identify_samples_worker, args=(
                    self.worker_directories[i], relevant_chunks_queue, sampled_chunks_queue, sample_positions_queue))
                process.start()
                processes.append(process)

            # Update the progress bar based on the progress queue
            while any(p.is_alive() for p in processes):
                while not sampled_chunks_queue.empty():
                    pbar.update(1)
                    sampled_chunks.add(sampled_chunks_queue.get())
                    if len(sampled_chunks) % self.sample_progress_save_interval == 0:
                        self._save_sample_progress(
                            directory, sampled_chunks, sample_positions)
                while not sample_positions_queue.empty():
                    sample_positions.update(sample_positions_queue.get())
                    pbar.set_postfix(
                        {"sample_positions": len(sample_positions)})
                time.sleep(0.1)

            # Wait for all processes to finish
            for process in processes:
                process.join()
        finally:
            # Empty the queue
            while True:
                try:
                    relevant_chunks_queue.get_nowait()
                except Empty:
                    break

        # Final save
        self._save_sample_progress(directory, sampled_chunks, sample_positions)

    def _get_schematic_path(self, world_name: str, position: tuple) -> str:
        """Returns the path to the schematic file for the given position"""
        filename = world_name + "_" + str(position)
        file_hash = hashlib.sha256(filename.encode()).hexdigest()
        path = os.path.join(self.schematic_directory,
                            world_name, file_hash + '.schem')
        return path

    def _collect_samples_worker(self, directory: str, world_name: str, sample_positions_queue: Queue, sampled_positions_queue: Queue) -> None:
        """Worker function for collecting samples"""

        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)

            purge_counter = 0
            while True:
                position = sample_positions_queue.get()
                if position is None:
                    sample_positions_queue.put(None)
                    break

                path = self._get_schematic_path(world_name, position)
                x, y, z = position

                selection = SelectionBox(
                    (x, y, z), (x + self.sample_size, y + self.sample_size, z + self.sample_size))
                structure = world.extract_structure(
                    selection, 'minecraft:overworld')

                try:
                    wrapper = SpongeSchemFormatWrapper(path)
                    wrapper.create_and_open(
                        'java', 3578, bounds=SelectionGroup(structure.bounds('minecraft:overworld')), overwrite=True)
                    structure.save(wrapper)
                finally:
                    wrapper.close()

                purge_counter += 1
                if purge_counter >= self.sampling_purge_interval:
                    purge_counter = 0
                    world.close()
                    world = amulet.load_level(directory)

                sampled_positions_queue.put(position)
        except KeyboardInterrupt:
            pass
        finally:
            world.close()

    def _collect_samples(self, directory: str) -> None:
        """Collects samples from the world at the identified positions"""

        # Clear the schematics directory
        for file in os.listdir(self.schematic_directory):
            file_path = os.path.join(self.schematic_directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        # Load progress
        _, all_sample_positions = self._load_sample_progress(directory)

        # Filter out positions that already have a schematic
        world_name = os.path.basename(directory)
        sample_positions = [p for p in all_sample_positions if not os.path.exists(
            self._get_schematic_path(world_name, p))]
        if len(sample_positions) == 0:
            print(
                f"All {len(all_sample_positions)} samples have already been collected")
            return

        # Set up worker directories
        self.setup_worker_directories(directory)

        # Create a tqdm progress bar
        pbar = tqdm(
            total=len(all_sample_positions),
            initial=(len(all_sample_positions) - len(sample_positions)),
            desc="Collecting samples"
        )

        # Create a queue and add all positions to it
        sample_positions_queue = Queue()
        for position in sample_positions:
            sample_positions_queue.put(position)
        sample_positions_queue.put(None)

        # Create a progress queue
        sampled_positions_queue = Queue()

        processes = []
        try:
            # Create and start worker processes
            for i in range(self.num_workers):
                process = Process(target=self._collect_samples_worker, args=(
                    self.worker_directories[i], world_name, sample_positions_queue, sampled_positions_queue))
                process.start()
                processes.append(process)

            start_time = time.time()

            # Update the progress bar based on the progress queue
            while any(p.is_alive() for p in processes):
                while not sampled_positions_queue.empty():
                    sampled_positions_queue.get()
                    pbar.update(1)

                # Check if a minute has passed
                if time.time() - start_time >= 60:
                    print(
                        f"Progress after 1 minute: {pbar.n} samples collected")
                    return

                time.sleep(0.1)

            # Wait for all processes to finish
            for process in processes:
                process.join()
        finally:
            # Empty the queue
            while True:
                try:
                    sample_positions_queue.get_nowait()
                except Empty:
                    break

    def sample_directory(self, directory: str) -> None:
        """Samples a directory of worlds"""
        try:
            print(f"Sampling directory: {directory}")
            for subdir in os.listdir(directory):
                if os.path.isdir(os.path.join(directory, subdir)):
                    self.sample_world(os.path.join(directory, subdir))
            print("Done sampling directory")
        except KeyboardInterrupt:
            pass

    def sample_world(self, directory: str) -> None:
        """Samples a world"""
        try:
            print(f"Sampling world: {directory}")
            self._mark_chunks(directory)
            self._identify_samples(directory)
            self._collect_samples(directory)
            print("Done sampling world")
        except KeyboardInterrupt:
            pass
