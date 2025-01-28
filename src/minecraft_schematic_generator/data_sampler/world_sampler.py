import hashlib
import json
import logging
import os
import random
import shutil
import time
from collections import Counter
from multiprocessing import Process, Queue
from queue import Empty

logging.getLogger("amulet").setLevel(logging.CRITICAL)
logging.getLogger("PyMCTranslate").setLevel(logging.CRITICAL)

import amulet  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from amulet.api.chunk import Chunk  # noqa: E402
from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError  # noqa: E402
from amulet.api.level import World  # noqa: E402
from amulet.api.selection import SelectionBox, SelectionGroup  # noqa: E402
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper  # noqa: E402
from amulet.utils.world_utils import chunk_coords_to_block_coords  # noqa: E402
from tqdm import tqdm  # noqa: E402


class WorldSampler:
    def __init__(
        self,
        schematic_directory,
        temp_directory,
        chunk_progress_save_interval,
        chunk_mark_radius,
        sample_offset,
        sample_size,
        sample_interested_block_threshold,
        sample_minimum_air_threshold,
        sample_progress_save_interval,
        sampling_purge_interval,
        clear_worker_directories=True,
        chunk_search_limit=None,
        sample_search_limit=None,
        sample_limit=None,
        num_workers=os.cpu_count(),
    ):
        self.schematic_directory = schematic_directory
        self.temp_directory = temp_directory
        self.chunk_progress_save_interval = chunk_progress_save_interval
        self.chunk_mark_radius = chunk_mark_radius
        self.sample_offset = sample_offset
        self.sample_size = sample_size
        self.sample_interested_block_threshold = sample_interested_block_threshold
        self.sample_minimum_air_threshold = sample_minimum_air_threshold
        self.sample_progress_save_interval = sample_progress_save_interval
        self.sampling_purge_interval = sampling_purge_interval
        self.num_workers = num_workers
        self.clear_worker_directories = clear_worker_directories
        self.chunk_search_limit = chunk_search_limit
        self.sample_search_limit = sample_search_limit
        self.sample_limit = sample_limit

        self.MC_VERSION = (1, 21, 4)

    def load_interested_blocks(self, directory: str) -> None:
        """Loads the interested blocks from the file"""
        # Check if a custom interested blocks file exists
        filename = "interested_blocks.json"
        if os.path.exists(os.path.join(directory, filename)):
            config_path = os.path.join(directory, filename)
            print(f"Using custom interested blocks file: {config_path}")
        else:
            config_path = os.path.join(os.path.dirname(__file__), filename)

        # Load the interested blocks from the file
        with open(config_path, "r") as file:
            interested_blocks = json.load(file)

        # Set the interested blocks
        self.chunk_target_blocks = {
            "minecraft:overworld": set(interested_blocks["chunk"]["overworld"]),
            "minecraft:the_nether": set(interested_blocks["chunk"]["nether"]),
            "minecraft:the_end": set(interested_blocks["chunk"]["end"]),
        }
        self.sample_target_blocks = {
            "minecraft:overworld": set(interested_blocks["sample"]["overworld"]),
            "minecraft:the_nether": set(interested_blocks["sample"]["nether"]),
            "minecraft:the_end": set(interested_blocks["sample"]["end"]),
        }

    def get_worker_directory(
        self, root_directory: str, src_directory: str, worker_id: int
    ) -> str:
        """Returns the directory for a given worker"""
        rel_path = os.path.relpath(src_directory, root_directory)
        copies_directory = os.path.join(self.temp_directory, rel_path)
        worker_directory_name = f"worker_{worker_id}"
        worker_directory = os.path.join(copies_directory, worker_directory_name)
        return worker_directory

    def setup_worker_directories(self, root_directory: str, src_directory: str) -> None:
        """Sets up worker directories for a given source directory"""

        # Check if the worker directories have already been set up
        for i in range(self.num_workers):
            worker_directory = self.get_worker_directory(
                root_directory, src_directory, i
            )
            if not os.path.exists(worker_directory):
                break
        else:
            return

        # Create a copy of the source directory for each worker
        pbar = tqdm(range(self.num_workers), desc="Setting up worker directories")
        for i in pbar:
            worker_directory = self.get_worker_directory(
                root_directory, src_directory, i
            )
            if not os.path.exists(worker_directory):
                shutil.copytree(src_directory, worker_directory)

    def _clear_worker_directories(
        self, root_directory: str, src_directory: str
    ) -> None:
        """Clears the worker directories for a given source directory"""

        rel_path = os.path.relpath(src_directory, root_directory)

        # Clear the worker directories if they exist
        copies_directory = os.path.join(self.temp_directory, rel_path)
        if os.path.exists(copies_directory):
            shutil.rmtree(copies_directory)

    def _check_block(self, block, target_blocks, translator):
        """Returns True if the block is one of the target blocks"""
        block, _, _ = translator.block.from_universal(block)
        if "universal" in block.namespaced_name:
            print(f"Conversion failed for {block.namespaced_name}")
            return False
        name = block.namespaced_name + "|"
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

    def _save_chunk_progress(
        self, directory: str, dimension: str, visited_chunks: set, relevant_chunks: set
    ) -> None:
        """Save the current chunk progress to a file"""
        dimension = dimension.replace(":", "_")
        temp_file_path = os.path.join(
            directory, f"{dimension}_chunk_progress_temp.json"
        )
        final_file_path = os.path.join(directory, f"{dimension}_chunk_progress.json")
        with open(temp_file_path, "w") as file:
            json.dump(
                {
                    "visited_chunks": list(visited_chunks),
                    "relevant_chunks": list(relevant_chunks),
                },
                file,
            )
        os.replace(temp_file_path, final_file_path)

    def _load_chunk_progress(self, directory: str, dimension: str) -> tuple:
        """Load the current chunk progress from a file"""
        dimension = dimension.replace(":", "_")
        path = os.path.join(directory, f"{dimension}_chunk_progress.json")
        if os.path.exists(path):
            with open(path, "r") as file:
                data = json.load(file)
            visited_chunks = set(tuple(c) for c in data["visited_chunks"])
            relevant_chunks = set(tuple(c) for c in data["relevant_chunks"])
        else:
            visited_chunks = set()
            relevant_chunks = set()
        return visited_chunks, relevant_chunks

    def _save_sample_progress(
        self, directory: str, dimension: str, sampled_chunks: set, sample_positions: set
    ) -> None:
        """Save the current sample progress to a file"""
        dimension = dimension.replace(":", "_")
        temp_file_path = os.path.join(
            directory, f"{dimension}_sample_progress_temp.json"
        )
        final_file_path = os.path.join(directory, f"{dimension}_sample_progress.json")
        with open(temp_file_path, "w") as file:
            json.dump(
                {
                    "sampled_chunks": list(sampled_chunks),
                    "sample_positions": list(sample_positions),
                },
                file,
            )
        os.replace(temp_file_path, final_file_path)

    def _load_sample_progress(self, directory: str, dimension: str) -> tuple:
        """Load the current sample progress from a file"""
        dimension = dimension.replace(":", "_")
        path = os.path.join(directory, f"{dimension}_sample_progress.json")
        if os.path.exists(path):
            with open(path, "r") as file:
                data = json.load(file)
            sampled_chunks = set(tuple(c) for c in data["sampled_chunks"])
            sample_positions = set(tuple(c) for c in data["sample_positions"])
        else:
            sampled_chunks = set()
            sample_positions = set()
        return sampled_chunks, sample_positions

    def _mark_chunks_worker(
        self,
        directory: str,
        dimension: str,
        all_chunks_queue: Queue,
        visited_chunks_queue: Queue,
        relevant_chunks_queue: Queue,
    ) -> None:
        """Worker function for marking chunks"""
        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)
            translator = world.translation_manager.get_version("java", self.MC_VERSION)

            while True:
                chunk_coords = all_chunks_queue.get()
                if chunk_coords is None:
                    all_chunks_queue.put(None)
                    break

                try:
                    chunk = world.level_wrapper.load_chunk(*chunk_coords, dimension)
                    if self._chunk_contains_target_blocks(
                        chunk, self.chunk_target_blocks[dimension], translator
                    ):
                        # Add this chunk and its neighbors
                        for dx in range(
                            -self.chunk_mark_radius, self.chunk_mark_radius + 1
                        ):
                            for dz in range(
                                -self.chunk_mark_radius, self.chunk_mark_radius + 1
                            ):
                                relevant_chunks_queue.put(
                                    (chunk_coords[0] + dx, chunk_coords[1] + dz)
                                )
                    successful = True
                except (ChunkDoesNotExist, ChunkLoadError):
                    successful = False
                except Exception as e:
                    print(f"Unknown error loading chunk {chunk_coords}: {e}")
                    successful = False
                visited_chunks_queue.put((chunk_coords, successful))
        except KeyboardInterrupt:
            pass
        finally:
            world.close()

    def _create_visualization(
        self,
        directory: str,
        dimension: str,
        x_coords,
        z_coords,
        intensities,
        title: str,
        colorbar: str = None,
    ) -> None:
        # Set the background for dark mode
        plt.style.use("dark_background")

        # Plotting
        plt.figure(figsize=(10, 10))
        # s is the size of the point
        plt.scatter(
            x_coords, z_coords, c=intensities, cmap="cool", edgecolor="none", s=10
        )
        plt.colorbar(label=colorbar)
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.title(title)

        # Adjust axes limits to have the same range for a square aspect ratio
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(z_coords), max(z_coords)
        xy_min = min(x_min, y_min)
        xy_max = max(x_max, y_max)
        plt.xlim(xy_min, xy_max)
        plt.ylim(xy_min, xy_max)

        plt.gca().invert_yaxis()

        plt.gca().set_aspect("equal", adjustable="box")

        # Save the plot as an image
        dimension = dimension.replace(":", "_")
        plt.savefig(os.path.join(directory, f"{dimension}_{title}.png"))
        plt.close()

    def _visualize_marked_chunks(
        self, directory: str, dimension: str, relevant_chunks: set
    ) -> None:
        if len(relevant_chunks) == 0:
            return

        # Extract x and z coordinates
        x_coords, z_coords = zip(*relevant_chunks)

        # Use 1's to indicate presence uniformly
        intensities = [1] * len(relevant_chunks)

        # Create visualization
        self._create_visualization(
            directory, dimension, x_coords, z_coords, intensities, "selected_chunks"
        )

    def _visualize_sample_positions(
        self, directory: str, dimension: str, sample_positions: set
    ) -> None:
        if len(sample_positions) == 0:
            return

        # Function to convert world coordinates to chunk coordinates
        def world_to_chunk(x, z):
            return x // 16, z // 16

        # Convert sample positions to chunk coordinates and count samples per chunk
        chunk_coordinates = [world_to_chunk(x, z) for x, _, z in sample_positions]
        samples_per_chunk = Counter(chunk_coordinates)

        # Prepare data for plotting
        x_coords, z_coords, intensities = zip(
            *[(x, z, samples_per_chunk[(x, z)]) for x, z in samples_per_chunk]
        )

        # Create visualization for sample positions with intensity based on sample count
        self._create_visualization(
            directory,
            dimension,
            x_coords,
            z_coords,
            intensities,
            "sample_positions",
            "Sample Count",
        )

    def _mark_chunks(self, root_directory: str, directory: str, dimension: str) -> set:
        """Looks through chunks in a world and marks them as relevant or not relevant"""

        # Get all chunk coordinates
        world = amulet.load_level(directory)
        all_chunk_coords = world.all_chunk_coords(dimension)
        world.close()

        if len(all_chunk_coords) == 0:
            print("No chunks found in the dimension")
            return set()

        # Load progress
        visited_chunks, relevant_chunks = self._load_chunk_progress(
            directory, dimension
        )

        # Remove visited chunks from the list
        remaining_chunk_coords = all_chunk_coords - visited_chunks
        if len(remaining_chunk_coords) == 0:
            print(f"All {len(all_chunk_coords)} chunks have already been visited")
            return relevant_chunks

        # Set up worker directories
        self.setup_worker_directories(root_directory, directory)

        # Create a tqdm progress bar
        pbar = tqdm(
            total=len(all_chunk_coords),
            initial=len(visited_chunks),
            desc="Marking chunks",
        )
        pbar.set_postfix({"relevant_chunks": len(relevant_chunks)})

        # Create a queue and add all chunk coordinates to it
        all_chunks_queue = Queue()
        if (
            self.chunk_search_limit
            and len(remaining_chunk_coords) > self.chunk_search_limit
        ):
            remaining_chunk_coords = set(
                random.sample(list(remaining_chunk_coords), self.chunk_search_limit)
            )
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
                process = Process(
                    target=self._mark_chunks_worker,
                    args=(
                        self.get_worker_directory(root_directory, directory, i),
                        dimension,
                        all_chunks_queue,
                        visited_chunks_queue,
                        relevant_chunks_queue,
                    ),
                )
                process.start()
                processes.append(process)

            # Update the progress bar based on the visited chunks
            unsuccessful_chunks_count = 0
            while any(p.is_alive() for p in processes):
                while not visited_chunks_queue.empty():
                    pbar.update(1)
                    chunk_coords, successful = visited_chunks_queue.get()
                    visited_chunks.add(chunk_coords)
                    if not successful:
                        unsuccessful_chunks_count += 1
                    if len(visited_chunks) % self.chunk_progress_save_interval == 0:
                        self._save_chunk_progress(
                            directory, dimension, visited_chunks, relevant_chunks
                        )
                while not relevant_chunks_queue.empty():
                    relevant_chunks.add(relevant_chunks_queue.get())
                pbar.set_postfix(
                    {
                        "relevant_chunks": len(relevant_chunks),
                        "unsuccessful_chunks": unsuccessful_chunks_count,
                    }
                )
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
        self._save_chunk_progress(
            directory, dimension, all_chunk_coords, relevant_chunks
        )

        return relevant_chunks

    def _get_interested_palette_indices(
        self, chunk: Chunk, translator, dimension: str
    ) -> set:
        """Returns a set of indices of blocks from the chunk palette that we are interested in"""
        interested_indices = set()
        for i, block in enumerate(chunk.block_palette):
            if self._check_block(
                block, self.sample_target_blocks[dimension], translator
            ):
                interested_indices.add(i)
        return interested_indices

    def _get_deterministic_random_offsets(self, chunk_coords):
        # Convert the chunk coordinates to a string
        coord_str = f"{chunk_coords[0]}_{chunk_coords[1]}"
        # Use a hash function, e.g., SHA-256
        hash_obj = hashlib.sha256(coord_str.encode())
        # Convert the hash to an integer
        hash_int = int(hash_obj.hexdigest(), base=16)
        # Generate three offsets using different ranges of the hash
        x_offset = hash_int % self.sample_offset
        y_offset = (hash_int // self.sample_offset) % self.sample_offset
        z_offset = (hash_int // (self.sample_offset**2)) % self.sample_offset
        return x_offset, y_offset, z_offset

    def _identify_samples_in_chunk(
        self, world: World, dimension: str, translator, chunk_coords
    ) -> set:
        """Identifies samples from a chunk"""

        sample_positions = set()
        min_height = world.bounds(dimension).min_y
        max_height = world.bounds(dimension).max_y
        world.get_block

        # Load all chunks that a selection starting in this chunk could possibly intersect
        num_chunks = self.sample_size // 16 + 2
        chunk_blocks = [[None for _ in range(num_chunks)] for _ in range(num_chunks)]
        interested_indices = [
            [None for _ in range(num_chunks)] for _ in range(num_chunks)
        ]
        for dx in range(num_chunks):
            for dz in range(num_chunks):
                inner_chunk_coords = (chunk_coords[0] + dx, chunk_coords[1] + dz)
                if world.has_chunk(*inner_chunk_coords, dimension):
                    chunk = world.get_chunk(*inner_chunk_coords, dimension)
                    world.purge()
                else:
                    chunk = Chunk(*inner_chunk_coords)
                chunk_blocks[dx][dz] = np.asarray(
                    chunk.blocks[:, min_height:max_height, :]
                )
                interested_indices[dx][dz] = self._get_interested_palette_indices(
                    chunk, translator, dimension
                )

        # Precompute interesting block and air block positions
        max_pos = 15 + self.sample_size
        y_size = max_height - min_height
        x_start, z_start = chunk_coords_to_block_coords(*chunk_coords)
        interested_block = np.zeros((max_pos, y_size, max_pos))
        air_block = np.zeros((max_pos, y_size, max_pos))
        original_block = np.zeros((max_pos, y_size, max_pos))
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
                    original_block[i, j, k] = block_value
                    chunk_interesting_indices = interested_indices[chunk_x][chunk_z]
                    interested_block[i, j, k] = (
                        1 if block_value in chunk_interesting_indices else 0
                    )
                    air_block[i, j, k] = 1 if block_value == 0 else 0

        # Calculate cumulative sums for both interested blocks and air blocks
        marked_count = np.cumsum(
            np.cumsum(np.cumsum(interested_block, axis=0), axis=1), axis=2
        )
        air_count = np.cumsum(np.cumsum(np.cumsum(air_block, axis=0), axis=1), axis=2)

        # Iterate through grid of possible selection start positions
        x_offset, y_offset, z_offset = self._get_deterministic_random_offsets(
            chunk_coords
        )
        m = self.sample_size
        y_limit = y_size - m
        middle_offset = m // 2
        i = x_offset
        while i < 16:
            found_valid_position_x = False
            j = y_offset
            while j < y_limit:
                found_valid_position_y = False
                k = z_offset
                while k < 16:
                    if (
                        original_block[
                            i + middle_offset, j + middle_offset, k + middle_offset
                        ]
                        == 0
                    ):
                        k += 1
                        continue
                    found_valid_position_y = True

                    # Calculate total marked (interested) blocks
                    total_marked = marked_count[i + m - 1][j + m - 1][k + m - 1]
                    # Calculate total air blocks
                    total_air = air_count[i + m - 1][j + m - 1][k + m - 1]

                    # Subtract previous slices for both counts
                    if i > 0:
                        total_marked -= marked_count[i - 1][j + m - 1][k + m - 1]
                        total_air -= air_count[i - 1][j + m - 1][k + m - 1]
                    if j > 0:
                        total_marked -= marked_count[i + m - 1][j - 1][k + m - 1]
                        total_air -= air_count[i + m - 1][j - 1][k + m - 1]
                    if k > 0:
                        total_marked -= marked_count[i + m - 1][j + m - 1][k - 1]
                        total_air -= air_count[i + m - 1][j + m - 1][k - 1]

                    # Add back double-subtracted regions
                    if i > 0 and j > 0:
                        total_marked += marked_count[i - 1][j - 1][k + m - 1]
                        total_air += air_count[i - 1][j - 1][k + m - 1]
                    if i > 0 and k > 0:
                        total_marked += marked_count[i - 1][j + m - 1][k - 1]
                        total_air += air_count[i - 1][j + m - 1][k - 1]
                    if j > 0 and k > 0:
                        total_marked += marked_count[i + m - 1][j - 1][k - 1]
                        total_air += air_count[i + m - 1][j - 1][k - 1]

                    # Subtract back triple-subtracted regions
                    if i > 0 and j > 0 and k > 0:
                        total_marked -= marked_count[i - 1][j - 1][k - 1]
                        total_air -= air_count[i - 1][j - 1][k - 1]

                    # Check both conditions
                    if (
                        total_marked > self.sample_interested_block_threshold
                        and total_air > self.sample_minimum_air_threshold
                    ):
                        x = x_start + i
                        y = j + min_height
                        z = z_start + k
                        sample_positions.add((x, y, z))

                    k += self.sample_offset

                if found_valid_position_y:
                    j += self.sample_offset
                    found_valid_position_x = True
                else:
                    j += 1

            if found_valid_position_x:
                i += self.sample_offset
            else:
                i += 1

        return sample_positions

    def _identify_samples_worker(
        self,
        directory: str,
        dimension: str,
        relevant_chunks_queue: Queue,
        sampled_chunks_queue: Queue,
        sample_positions_queue: Queue,
    ) -> None:
        """Worker function for identifying samples"""

        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)
            translator = world.translation_manager.get_version("java", self.MC_VERSION)

            while True:
                chunk_coords = relevant_chunks_queue.get()
                if chunk_coords is None:
                    relevant_chunks_queue.put(None)
                    break

                try:
                    if world.has_chunk(*chunk_coords, dimension):
                        samples = self._identify_samples_in_chunk(
                            world, dimension, translator, chunk_coords
                        )
                        sample_positions_queue.put(samples)
                    successful = True
                except ChunkLoadError:
                    successful = False
                except Exception as e:
                    print(
                        f"Unknown error identifying samples in chunk {chunk_coords}: {e}"
                    )
                    successful = False
                sampled_chunks_queue.put((chunk_coords, successful))
        except KeyboardInterrupt:
            pass
        finally:
            world.close()

    def _identify_samples(
        self, root_directory: str, directory: str, dimension: str, relevant_chunks: set
    ) -> set:
        """Identifies samples from the marked chunks"""

        # Load progress
        sampled_chunks, sample_positions = self._load_sample_progress(
            directory, dimension
        )

        # Get all relevant chunks that have not been sampled
        remaining_relevant_chunks = relevant_chunks - sampled_chunks
        if len(remaining_relevant_chunks) == 0:
            print(
                f"All {len(relevant_chunks)} relevant chunks have already been sampled"
            )
            return sample_positions

        # Set up worker directories
        self.setup_worker_directories(root_directory, directory)

        # Create a tqdm progress bar
        pbar = tqdm(
            total=len(relevant_chunks),
            initial=len(sampled_chunks),
            desc="Identifying samples from chunks",
        )
        pbar.set_postfix({"sample_positions": len(sample_positions)})

        # Create a queue and add all chunk coordinates to it
        relevant_chunks_queue = Queue()
        if (
            self.sample_search_limit
            and len(remaining_relevant_chunks) > self.sample_search_limit
        ):
            remaining_relevant_chunks = set(
                random.sample(list(remaining_relevant_chunks), self.sample_search_limit)
            )
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
                process = Process(
                    target=self._identify_samples_worker,
                    args=(
                        self.get_worker_directory(root_directory, directory, i),
                        dimension,
                        relevant_chunks_queue,
                        sampled_chunks_queue,
                        sample_positions_queue,
                    ),
                )
                process.start()
                processes.append(process)

            # Update the progress bar based on the progress queue
            unsuccessful_chunks_count = 0
            while any(p.is_alive() for p in processes):
                while not sampled_chunks_queue.empty():
                    pbar.update(1)
                    chunk_coords, successful = sampled_chunks_queue.get()
                    sampled_chunks.add(chunk_coords)
                    if not successful:
                        unsuccessful_chunks_count += 1
                    if len(sampled_chunks) % self.sample_progress_save_interval == 0:
                        self._save_sample_progress(
                            directory, dimension, sampled_chunks, sample_positions
                        )
                while not sample_positions_queue.empty():
                    sample_positions.update(sample_positions_queue.get())
                pbar.set_postfix(
                    {
                        "sample_positions": len(sample_positions),
                        "unsuccessful_chunks": unsuccessful_chunks_count,
                    }
                )
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
        self._save_sample_progress(
            directory, dimension, sampled_chunks, sample_positions
        )

        return sample_positions

    def _get_schematic_path(
        self, world_name: str, dimension: str, position: tuple
    ) -> str:
        """Returns the path to the schematic file for the given position"""
        filename = world_name + dimension + str(position)
        file_hash = hashlib.sha256(filename.encode()).hexdigest()
        path = os.path.join(self.schematic_directory, world_name, file_hash + ".schem")
        return path

    def _collect_samples_worker(
        self,
        directory: str,
        dimension: str,
        world_name: str,
        sample_positions_queue: Queue,
        sampled_positions_queue: Queue,
    ) -> None:
        """Worker function for collecting samples"""

        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)

            purge_counter = 0
            while True:
                position = sample_positions_queue.get()

                # Check if the worker should stop
                if position is None:
                    # Put None back into the queue for the next worker
                    sample_positions_queue.put(None)
                    break

                path = self._get_schematic_path(world_name, dimension, position)
                x, y, z = position

                selection = SelectionBox(
                    (x, y, z),
                    (x + self.sample_size, y + self.sample_size, z + self.sample_size),
                )
                structure = world.extract_structure(selection, dimension)

                tmp_path = f"{directory}/tmp.schem"

                try:
                    # Save the schematic to a temporary file
                    wrapper = SpongeSchemFormatWrapper(tmp_path)
                    wrapper.create_and_open(
                        "java",
                        self.MC_VERSION,
                        bounds=SelectionGroup(structure.bounds(dimension)),
                        overwrite=True,
                    )
                    structure.save(wrapper)
                finally:
                    wrapper.close()

                # Move the temporary file to the final location
                os.replace(tmp_path, path)

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

    def _collect_samples(
        self,
        root_directory: str,
        directory: str,
        dimension: str,
        all_sample_positions: set,
    ) -> None:
        """Collects samples from the world at the identified positions"""

        # Filter out positions that already have a schematic
        world_name = os.path.relpath(directory, root_directory)
        sample_positions = set(
            [
                p
                for p in all_sample_positions
                if not os.path.exists(
                    self._get_schematic_path(world_name, dimension, p)
                )
            ]
        )
        if len(sample_positions) == 0:
            print(
                f"All {len(all_sample_positions)} samples have already been collected"
            )
            return

        # Create the schematic directory if it doesn't exist
        os.makedirs(os.path.join(self.schematic_directory, world_name), exist_ok=True)

        # Set up worker directories
        self.setup_worker_directories(root_directory, directory)

        # Create a tqdm progress bar
        pbar = tqdm(
            total=len(all_sample_positions),
            initial=(len(all_sample_positions) - len(sample_positions)),
            desc="Collecting samples",
        )

        # Create a queue and add all positions to it
        if self.sample_limit and len(sample_positions) > self.sample_limit:
            sample_positions = set(
                random.sample(list(sample_positions), self.sample_limit)
            )
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
                process = Process(
                    target=self._collect_samples_worker,
                    args=(
                        self.get_worker_directory(root_directory, directory, i),
                        dimension,
                        world_name,
                        sample_positions_queue,
                        sampled_positions_queue,
                    ),
                )
                process.start()
                processes.append(process)

            # Update the progress bar based on the progress queue
            while any(p.is_alive() for p in processes):
                while not sampled_positions_queue.empty():
                    sampled_positions_queue.get()
                    pbar.update(1)
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
        """Samples a directory of worlds recursively."""
        try:
            print(f"Sampling directory: {directory}")
            for root, dirs, files in os.walk(directory):
                if "level.dat" in files:
                    self.sample_world(directory, root)
                    # Prevent further traversal into subdirectories of a world
                    dirs[:] = []
            schematic_count = sum(
                len(files) for _, _, files in os.walk(self.schematic_directory)
            )
            print(f"Total schematics: {schematic_count}")
            print("Done sampling directory")
        except KeyboardInterrupt:
            pass

    def sample_world(self, root_directory: str, directory: str) -> None:
        """Samples a world"""
        print(f"Sampling {directory}")
        self.load_interested_blocks(directory)
        for dimension in [
            "minecraft:overworld",
            "minecraft:the_nether",
            "minecraft:the_end",
        ]:
            print(f"Sampling dimension: {dimension}")
            relevant_chunks = self._mark_chunks(root_directory, directory, dimension)
            if len(relevant_chunks) == 0:
                print("No relevant chunks found")
                continue
            self._visualize_marked_chunks(directory, dimension, relevant_chunks)
            sample_positions = self._identify_samples(
                root_directory, directory, dimension, relevant_chunks
            )
            if len(sample_positions) == 0:
                print("No sample positions found")
                continue
            self._visualize_sample_positions(directory, dimension, sample_positions)
            self._collect_samples(
                root_directory, directory, dimension, sample_positions
            )
        if self.clear_worker_directories:
            self._clear_worker_directories(root_directory, directory)
        print(f"Done sampling {directory}")

    def clear_directory(self, directory: str) -> None:
        """Clears all progress and samples from a directory"""
        try:
            print(f"Clearing directory: {directory}")
            for root, dirs, files in os.walk(directory):
                if "level.dat" in files:
                    self.clear_world(root)
                    # Prevent further traversal into subdirectories of a world
                    dirs[:] = []
            print("Done clearing directory")
        except KeyboardInterrupt:
            pass

    def clear_world(self, directory: str) -> None:
        """Clears all progress and samples from a world"""
        print(f"Clearing {directory}")
        for dimension in [
            "minecraft:overworld",
            "minecraft:the_nether",
            "minecraft:the_end",
        ]:
            dimension = dimension.replace(":", "_")
            chunk_progress_path = os.path.join(
                directory, f"{dimension}_chunk_progress.json"
            )
            sample_progress_path = os.path.join(
                directory, f"{dimension}_sample_progress.json"
            )
            if os.path.exists(chunk_progress_path):
                os.remove(chunk_progress_path)
            if os.path.exists(sample_progress_path):
                os.remove(sample_progress_path)
        shutil.rmtree(
            os.path.join(self.schematic_directory, os.path.basename(directory)),
            ignore_errors=True,
        )
        print(f"Done clearing {directory}")
