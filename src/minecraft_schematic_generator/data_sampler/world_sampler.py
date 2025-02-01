import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import time
from collections import Counter
from multiprocessing import Process, Queue
from queue import Empty

from tqdm import tqdm

from minecraft_schematic_generator.constants import (
    MINECRAFT_PLATFORM,
    MINECRAFT_VERSION,
)

logging.getLogger("amulet").setLevel(logging.WARNING)
logging.getLogger("PyMCTranslate").setLevel(logging.CRITICAL)

import amulet  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from amulet.api.block import Block  # noqa: E402
from amulet.api.chunk import Chunk  # noqa: E402
from amulet.api.errors import ChunkDoesNotExist, ChunkLoadError  # noqa: E402
from amulet.api.level import World  # noqa: E402
from amulet.api.selection import SelectionBox, SelectionGroup  # noqa: E402
from amulet.level.formats.sponge_schem import SpongeSchemFormatWrapper  # noqa: E402
from amulet.utils.world_utils import chunk_coords_to_block_coords  # noqa: E402
from PyMCTranslate.py3.api.version.translators import BlockTranslator  # noqa: E402


class WorldSampler:
    # Define dimensions as a class variable
    DIMENSIONS = {
        "minecraft:overworld": "overworld",
        "minecraft:the_nether": "nether",
        "minecraft:the_end": "end",
    }

    def __init__(
        self,
        schematic_directory,
        temp_directory,
        chunk_progress_save_interval,
        chunk_mark_radius,
        sample_offset,
        sample_size,
        sample_target_block_threshold,
        sample_minimum_air_threshold,
        sample_progress_save_interval,
        sampling_purge_interval,
        clear_worker_directories=True,
        chunk_search_limit=None,
        sample_search_limit=None,
        sample_limit=None,
        num_mark_chunks_workers=1,
        num_identify_samples_workers=1,
        num_collect_samples_workers=1,
    ):
        self.schematic_directory = schematic_directory
        self.temp_directory = temp_directory
        self.chunk_progress_save_interval = chunk_progress_save_interval
        self.chunk_mark_radius = chunk_mark_radius
        self.sample_offset = sample_offset
        self.sample_size = sample_size
        self.sample_target_block_threshold = sample_target_block_threshold
        self.sample_minimum_air_threshold = sample_minimum_air_threshold
        self.sample_progress_save_interval = sample_progress_save_interval
        self.sampling_purge_interval = sampling_purge_interval
        self.num_mark_chunks_workers = num_mark_chunks_workers
        self.num_identify_samples_workers = num_identify_samples_workers
        self.num_collect_samples_workers = num_collect_samples_workers
        self.clear_worker_directories = clear_worker_directories
        self.chunk_search_limit = chunk_search_limit
        self.sample_search_limit = sample_search_limit
        self.sample_limit = sample_limit

    def _get_data_directory(self, directory: str) -> str:
        """Returns the path to the data directory for a world"""
        return os.path.join(directory, ".minecraft_schematic_generator")

    def _get_world_version(self, directory: str) -> tuple:
        """Gets the version of the Minecraft world"""
        world = amulet.load_level(directory)
        try:
            version = world.translation_manager._get_version_number(
                world.level_wrapper.platform, world.level_wrapper.version
            )
            return version
        finally:
            world.close()

    def _get_available_target_block_versions(self) -> dict:
        """Returns a list of available target block versions"""
        dir = os.path.join(os.path.dirname(__file__), "target_blocks")

        # Get all available version directories
        version_dirs = {}
        for item in os.listdir(dir):
            version_path = os.path.join(dir, item)
            if os.path.isdir(version_path):
                try:
                    # Split version string and pad with zeros
                    version_parts = item.split(".")
                    version = tuple(int(x) for x in version_parts) + (0,) * (
                        3 - len(version_parts)
                    )
                    version_dirs[version] = version_path
                except ValueError:
                    continue

        return version_dirs

    def _get_version_specific_target_block_path(self, directory: str) -> tuple:
        """Gets the path to the version-specific file that's closest to but not older than the world version"""
        world_version = self._get_world_version(directory)
        versions = self._get_available_target_block_versions()

        # Find the oldest version that's not older than the world version
        valid_versions = [v for v in versions if v >= world_version]
        if not valid_versions:
            raise ValueError("No compatible target block files found")
        closest_version = min(valid_versions)

        # Return path
        return versions[closest_version], closest_version

    def load_target_blocks(self, directory: str) -> tuple[dict, dict]:
        """Loads the target blocks from the files and returns them"""
        # Initialize dictionaries
        chunk_target_blocks = {}
        sample_target_blocks = {}

        version_path = None
        data_dir = self._get_data_directory(directory)

        # Load each dimension's blocks from separate files
        for block_type in ["chunk", "sample"]:
            for dimension in self.DIMENSIONS:
                prefix = self.DIMENSIONS[dimension]
                filename = f"{prefix}_{block_type}_blocks.json"

                # Check if custom files exist first
                custom_path = os.path.join(data_dir, filename)
                if os.path.exists(custom_path):
                    config_path = custom_path
                    print(f"Using custom {block_type} target blocks for {dimension}")
                else:
                    if version_path is None:
                        version_path, version = (
                            self._get_version_specific_target_block_path(directory)
                        )
                        print(
                            f"Using target blocks from {'.'.join(str(x) for x in version)}"
                        )
                    config_path = os.path.join(version_path, filename)

                # Load the block file
                with open(config_path, "r") as file:
                    target_blocks = json.load(file)

                # Store the target blocks
                if block_type == "chunk":
                    chunk_target_blocks[dimension] = target_blocks
                else:
                    sample_target_blocks[dimension] = target_blocks

        return chunk_target_blocks, sample_target_blocks

    def get_worker_directory(
        self, root_directory: str, src_directory: str, worker_id: int
    ) -> str:
        """Returns the directory for a given worker"""
        rel_path = os.path.relpath(src_directory, root_directory)
        copies_directory = os.path.join(self.temp_directory, rel_path)
        worker_directory_name = f"worker_{worker_id}"
        worker_directory = os.path.join(copies_directory, worker_directory_name)
        return worker_directory

    def setup_worker_directories(
        self, root_directory: str, src_directory: str, num_workers: int
    ) -> None:
        """Sets up worker directories for a given source directory"""
        COPY_LIST = {
            "level.dat",  # Essential world data
            "region",  # Block data
            "DIM-1",  # Nether
            "DIM1",  # End
        }

        # Check if the worker directories have already been set up
        for i in range(num_workers):
            worker_directory = self.get_worker_directory(
                root_directory, src_directory, i
            )
            if not os.path.exists(worker_directory):
                break
        else:
            return

        # Create a copy of only the necessary files for each worker
        pbar = tqdm(range(num_workers), desc="Setting up worker directories")
        for i in pbar:
            worker_directory = self.get_worker_directory(
                root_directory, src_directory, i
            )
            if not os.path.exists(worker_directory):
                os.makedirs(worker_directory)

                # Copy only the files/directories we need
                for item in os.listdir(src_directory):
                    if item in COPY_LIST:
                        src_path = os.path.join(src_directory, item)
                        dst_path = os.path.join(worker_directory, item)
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path)

    def _clear_worker_directories(
        self, root_directory: str, src_directory: str
    ) -> None:
        """Clears the worker directories for a given source directory"""

        rel_path = os.path.relpath(src_directory, root_directory)
        copies_directory = os.path.join(self.temp_directory, rel_path)

        # Clear the worker directories if they exist
        if os.path.exists(copies_directory):
            shutil.rmtree(copies_directory)

            # Clean up empty parent directories
            parent = os.path.dirname(copies_directory)
            while parent >= self.temp_directory:
                if len(os.listdir(parent)) == 0:
                    os.rmdir(parent)
                    parent = os.path.dirname(parent)
                else:
                    break

    def _check_block(
        self,
        block: Block,
        target_blocks: set,
        translator: BlockTranslator,
        block_cache: dict,
    ) -> bool:
        """Returns True if the block is one of the target blocks"""
        cache_key = block.namespaced_name
        if cache_key in block_cache:
            return block_cache[cache_key]

        block, _, _ = translator.from_universal(block)
        if "universal" in block.namespaced_name:
            print(f"Conversion failed for {block.namespaced_name}")
            result = False
        else:
            name = block.namespaced_name + "|"
            result = any(target_block in name for target_block in target_blocks)

        block_cache[cache_key] = result
        return result

    def _save_progress(
        self, directory: str, dimension: str, name: str, config: dict, data: dict
    ) -> None:
        data_dir = self._get_data_directory(directory)
        os.makedirs(data_dir, exist_ok=True)

        dimension = dimension.replace(":", "_")
        temp_file_path = os.path.join(data_dir, f"{dimension}_{name}_progress_temp.pkl")
        final_file_path = os.path.join(data_dir, f"{dimension}_{name}_progress.pkl")

        # Try to save multiple times with a delay in case the file is locked
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Make sure temp file is closed and deleted if it exists
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except FileNotFoundError:
                        pass

                # Write to temp file
                with open(temp_file_path, "wb") as file:
                    pickle.dump(
                        {
                            "config": config,
                            "data": data,
                        },
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                # Try to replace the file
                os.replace(temp_file_path, final_file_path)
                break
            except PermissionError:
                print(f"Failed to save {name} progress to {final_file_path}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                print("All retries failed")
                raise

    def _load_progress(
        self, directory: str, dimension: str, name: str, current_config: dict
    ) -> dict | None:
        data_dir = self._get_data_directory(directory)
        dimension = dimension.replace(":", "_")
        path = os.path.join(data_dir, f"{dimension}_{name}_progress.pkl")

        if os.path.exists(path):
            try:
                with open(path, "rb") as file:
                    progress = pickle.load(file)

                # Handle unrecognized progress file format
                if (
                    not isinstance(progress, dict)
                    or "config" not in progress
                    or "data" not in progress
                ):
                    print(
                        f"Unrecognized progress file format detected, resetting {name} progress"
                    )
                    return None

                if progress["config"] != current_config:
                    print(f"Configuration has changed, resetting {name} progress")
                    return None

                return progress["data"]
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                print(f"Error reading {name} progress file, resetting progress: {e}")
                return None
        return None

    def _chunk_contains_target_blocks(
        self,
        chunk: Chunk,
        target_blocks: set,
        translator: BlockTranslator,
        block_cache: dict,
    ) -> bool:
        """Returns True if the chunk contains any of the target blocks"""
        for block in chunk.block_palette:
            if self._check_block(block, target_blocks, translator, block_cache):
                return True
        return False

    def _get_chunk_config(self, target_blocks: list) -> dict:
        """Returns the configuration for marking chunks"""
        return {
            "target_blocks": target_blocks,
            "mark_radius": self.chunk_mark_radius,
        }

    def _save_chunk_progress(
        self,
        directory: str,
        dimension: str,
        target_blocks: list,
        visited_chunks: set,
        relevant_chunks: set,
    ) -> None:
        """Save the current chunk progress to a file"""
        config = self._get_chunk_config(target_blocks)
        data = {
            "visited_chunks": visited_chunks,
            "relevant_chunks": relevant_chunks,
        }
        self._save_progress(directory, dimension, "chunk", config, data)

    def _load_chunk_progress(
        self, directory: str, dimension: str, target_blocks: list
    ) -> tuple:
        """Load the current chunk progress from a file"""
        current_config = self._get_chunk_config(target_blocks)
        data = self._load_progress(directory, dimension, "chunk", current_config)
        return (
            (data["visited_chunks"], data["relevant_chunks"])
            if data
            else (set(), set())
        )

    def _get_sample_config(self, target_blocks: list) -> dict:
        """Returns the configuration for identifying samples"""
        return {
            "target_blocks": target_blocks,
            "offset": self.sample_offset,
            "size": self.sample_size,
            "target_block_threshold": self.sample_target_block_threshold,
            "minimum_air_threshold": self.sample_minimum_air_threshold,
        }

    def _save_sample_progress(
        self,
        directory: str,
        dimension: str,
        target_blocks: list,
        sampled_chunks: set,
        sample_positions: set,
    ) -> None:
        """Save the current sample progress to a file"""
        config = self._get_sample_config(target_blocks)
        data = {
            "sampled_chunks": sampled_chunks,
            "sample_positions": sample_positions,
        }
        self._save_progress(directory, dimension, "sample", config, data)

    def _load_sample_progress(
        self, directory: str, dimension: str, target_blocks: list
    ) -> tuple:
        """Load the current sample progress from a file"""
        current_config = self._get_sample_config(target_blocks)
        data = self._load_progress(directory, dimension, "sample", current_config)
        return (
            (data["sampled_chunks"], data["sample_positions"])
            if data
            else (set(), set())
        )

    def _mark_chunks_worker(
        self,
        directory: str,
        dimension: str,
        target_blocks: set,
        all_chunks_queue: Queue,
        visited_chunks_queue: Queue,
        relevant_chunks_queue: Queue,
    ) -> None:
        """Worker function for marking chunks"""
        try:
            # Create a new block cache for this worker
            block_cache = {}

            # Load the world data from the directory
            world = amulet.load_level(directory)

            # Use the translator for the project level Minecraft version which target blocks are defined for
            translator = world.translation_manager.get_version(
                MINECRAFT_PLATFORM, MINECRAFT_VERSION
            ).block

            while True:
                chunk_coords = all_chunks_queue.get()
                if chunk_coords is None:
                    all_chunks_queue.put(None)
                    break

                try:
                    chunk = world.level_wrapper.load_chunk(*chunk_coords, dimension)
                    if self._chunk_contains_target_blocks(
                        chunk, target_blocks, translator, block_cache
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
        x_coords: list,
        z_coords: list,
        intensities: list,
        title: str,
        colorbar: str = None,
    ) -> None:
        data_dir = self._get_data_directory(directory)
        os.makedirs(data_dir, exist_ok=True)

        # Use Agg backend for headless operation
        matplotlib.use("Agg")

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
        plt.savefig(os.path.join(data_dir, f"{dimension}_{title}.png"))
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

    def _mark_chunks(
        self,
        root_directory: str,
        directory: str,
        dimension: str,
        target_blocks: list,
    ) -> set:
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
            directory, dimension, target_blocks
        )

        # Remove visited chunks from the list
        remaining_chunk_coords = all_chunk_coords - visited_chunks
        if len(remaining_chunk_coords) == 0:
            print(f"All {len(all_chunk_coords)} chunks have already been visited")
            return relevant_chunks

        # Set up worker directories
        self.setup_worker_directories(
            root_directory, directory, self.num_mark_chunks_workers
        )

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
        target_blocks_set = set(target_blocks)
        try:
            # Create and start worker processes
            for i in tqdm(
                range(self.num_mark_chunks_workers),
                desc="Starting worker processes",
                leave=False,
            ):
                process = Process(
                    target=self._mark_chunks_worker,
                    args=(
                        self.get_worker_directory(root_directory, directory, i),
                        dimension,
                        target_blocks_set,
                        all_chunks_queue,
                        visited_chunks_queue,
                        relevant_chunks_queue,
                    ),
                )
                process.start()
                processes.append(process)

            # Create a tqdm progress bar
            pbar = tqdm(
                total=len(all_chunk_coords),
                initial=len(visited_chunks),
                desc="Marking chunks",
            )
            pbar.set_postfix({"relevant_chunks": len(relevant_chunks)})

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
                            directory,
                            dimension,
                            target_blocks,
                            visited_chunks,
                            relevant_chunks,
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
            directory, dimension, target_blocks, visited_chunks, relevant_chunks
        )

        return relevant_chunks

    def _get_target_palette_indices(
        self,
        translator: BlockTranslator,
        target_blocks: set,
        chunk: Chunk,
        block_cache: dict,
    ) -> set:
        """Returns a set of indices of blocks from the chunk palette that we are targeting"""
        target_indices = set()
        for i, block in enumerate(chunk.block_palette):
            if self._check_block(block, target_blocks, translator, block_cache):
                target_indices.add(i)
        return target_indices

    def _get_deterministic_random_offsets(self, chunk_coords: tuple) -> tuple:
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
        self,
        world: World,
        dimension: str,
        translator: BlockTranslator,
        target_blocks: set,
        chunk_coords: tuple,
        block_cache: dict,
    ) -> set:
        """Identifies samples from a chunk"""
        sample_positions = set()
        min_height = world.bounds(dimension).min_y
        max_height = world.bounds(dimension).max_y

        # Load all chunks that a selection starting in this chunk could possibly intersect
        num_chunks = self.sample_size // 16 + 2
        # Create single 3D arrays instead of nested lists - using more efficient dtype specification
        chunk_blocks = np.zeros(
            (num_chunks * 16, max_height - min_height, num_chunks * 16),
            dtype=np.int32,
            order="C",
        )
        target_block = np.zeros_like(chunk_blocks, dtype=bool)
        air_block = np.zeros_like(chunk_blocks, dtype=bool)

        # Load chunks directly into the arrays
        for dx in range(num_chunks):
            for dz in range(num_chunks):
                inner_chunk_coords = (chunk_coords[0] + dx, chunk_coords[1] + dz)
                x_start, z_start = dx * 16, dz * 16
                # Pre-calculate slice objects
                x_slice = slice(x_start, x_start + 16)
                z_slice = slice(z_start, z_start + 16)

                if world.has_chunk(*inner_chunk_coords, dimension):
                    chunk = world.get_chunk(*inner_chunk_coords, dimension)
                    blocks = np.asarray(chunk.blocks[:, min_height:max_height, :])
                    target_indices = self._get_target_palette_indices(
                        translator, target_blocks, chunk, block_cache
                    )
                else:
                    blocks = np.zeros((16, max_height - min_height, 16), dtype=np.int32)
                    target_indices = set()

                # Use pre-calculated slices
                chunk_blocks[x_slice, :, z_slice] = blocks

                # Create masks for target blocks and air blocks
                if target_indices:
                    target_block[x_slice, :, z_slice] = np.isin(
                        blocks, list(target_indices)
                    )
                air_block[x_slice, :, z_slice] = blocks == 0

        world.purge()

        # Calculate cumulative sums with optimal axis order for memory access
        marked_count = np.cumsum(
            np.cumsum(np.cumsum(target_block, axis=2), axis=1), axis=0
        )
        air_count = np.cumsum(np.cumsum(np.cumsum(air_block, axis=2), axis=1), axis=0)

        # Iterate through grid of possible selection start positions
        x_offset, y_offset, z_offset = self._get_deterministic_random_offsets(
            chunk_coords
        )
        m = self.sample_size
        y_size = max_height - min_height
        y_limit = y_size - m
        middle_offset = m // 2
        x_start, z_start = chunk_coords_to_block_coords(*chunk_coords)

        i = x_offset
        while i < 16:
            found_valid_position_x = False
            j = y_offset
            while j < y_limit:
                found_valid_position_y = False
                k = z_offset
                while k < 16:
                    if (
                        chunk_blocks[
                            i + middle_offset, j + middle_offset, k + middle_offset
                        ]
                        == 0
                    ):
                        k += 1
                        continue
                    found_valid_position_y = True

                    # Calculate total marked (target) blocks
                    total_marked = marked_count[i + m - 1, j + m - 1, k + m - 1]
                    # Calculate total air blocks
                    total_air = air_count[i + m - 1, j + m - 1, k + m - 1]

                    # Subtract previous slices for both counts
                    if i > 0:
                        total_marked -= marked_count[i - 1, j + m - 1, k + m - 1]
                        total_air -= air_count[i - 1, j + m - 1, k + m - 1]
                    if j > 0:
                        total_marked -= marked_count[i + m - 1, j - 1, k + m - 1]
                        total_air -= air_count[i + m - 1, j - 1, k + m - 1]
                    if k > 0:
                        total_marked -= marked_count[i + m - 1, j + m - 1, k - 1]
                        total_air -= air_count[i + m - 1, j + m - 1, k - 1]

                    # Add back double-subtracted regions
                    if i > 0 and j > 0:
                        total_marked += marked_count[i - 1, j - 1, k + m - 1]
                        total_air += air_count[i - 1, j - 1, k + m - 1]
                    if i > 0 and k > 0:
                        total_marked += marked_count[i - 1, j + m - 1, k - 1]
                        total_air += air_count[i - 1, j + m - 1, k - 1]
                    if j > 0 and k > 0:
                        total_marked += marked_count[i + m - 1, j - 1, k - 1]
                        total_air += air_count[i + m - 1, j - 1, k - 1]

                    # Subtract back triple-subtracted regions
                    if i > 0 and j > 0 and k > 0:
                        total_marked -= marked_count[i - 1, j - 1, k - 1]
                        total_air -= air_count[i - 1, j - 1, k - 1]

                    # Check both conditions
                    if (
                        total_marked > self.sample_target_block_threshold
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
        target_blocks: set,
        relevant_chunks_queue: Queue,
        sampled_chunks_queue: Queue,
        sample_positions_queue: Queue,
    ) -> None:
        """Worker function for identifying samples"""

        try:
            # Create a new block cache for this worker
            block_cache = {}

            # Load the world data from the directory
            world = amulet.load_level(directory)

            # Use the translator for the project level Minecraft version which target blocks are defined for
            translator = world.translation_manager.get_version(
                MINECRAFT_PLATFORM, MINECRAFT_VERSION
            ).block

            while True:
                chunk_coords = relevant_chunks_queue.get()
                if chunk_coords is None:
                    relevant_chunks_queue.put(None)
                    break

                try:
                    if world.has_chunk(*chunk_coords, dimension):
                        samples = self._identify_samples_in_chunk(
                            world,
                            dimension,
                            translator,
                            target_blocks,
                            chunk_coords,
                            block_cache,
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
        self,
        root_directory: str,
        directory: str,
        dimension: str,
        target_blocks: list,
        relevant_chunks: set,
    ) -> set:
        """Identifies samples from the marked chunks"""

        # Load progress
        sampled_chunks, sample_positions = self._load_sample_progress(
            directory, dimension, target_blocks
        )

        # Get all relevant chunks that have not been sampled
        remaining_relevant_chunks = relevant_chunks - sampled_chunks
        if len(remaining_relevant_chunks) == 0:
            print(
                f"All {len(relevant_chunks)} relevant chunks have already been sampled"
            )
            return sample_positions

        # Set up worker directories
        self.setup_worker_directories(
            root_directory, directory, self.num_identify_samples_workers
        )

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
        target_blocks_set = set(target_blocks)
        try:
            # Create and start worker processes
            for i in tqdm(
                range(self.num_identify_samples_workers),
                desc="Starting worker processes",
                leave=False,
            ):
                process = Process(
                    target=self._identify_samples_worker,
                    args=(
                        self.get_worker_directory(root_directory, directory, i),
                        dimension,
                        target_blocks_set,
                        relevant_chunks_queue,
                        sampled_chunks_queue,
                        sample_positions_queue,
                    ),
                )
                process.start()
                processes.append(process)

            # Create a tqdm progress bar
            pbar = tqdm(
                total=len(relevant_chunks),
                initial=len(sampled_chunks),
                desc="Identifying samples from chunks",
            )
            pbar.set_postfix({"sample_positions": len(sample_positions)})

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
                            directory,
                            dimension,
                            target_blocks,
                            sampled_chunks,
                            sample_positions,
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
            directory, dimension, target_blocks, sampled_chunks, sample_positions
        )

        return sample_positions

    def _get_schematic_hash(
        self, world_name: str, dimension: str, position: tuple
    ) -> str:
        """Returns the hash of the schematic file for the given position"""
        filename = world_name + dimension + str(position)
        return hashlib.sha256(filename.encode()).hexdigest()

    def _get_dimension_directory(self, world_name: str, dimension: str) -> str:
        """Returns the directory path for a given dimension"""
        dimension_name = dimension.replace(
            "minecraft:", ""
        )  # Convert minecraft:overworld to overworld
        return os.path.join(self.schematic_directory, world_name, dimension_name)

    def _get_schematic_path(
        self, world_name: str, dimension: str, position: tuple
    ) -> str:
        """Returns the path to the schematic file for the given position"""
        file_hash = self._get_schematic_hash(world_name, dimension, position)
        dimension_dir = self._get_dimension_directory(world_name, dimension)
        return os.path.join(dimension_dir, file_hash + ".schem")

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

                    # The schematic must be saved in a specific MC version so we choose the one we are using thoughout the project
                    wrapper.create_and_open(
                        MINECRAFT_PLATFORM,
                        MINECRAFT_VERSION,
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
        schematic_directory = self._get_dimension_directory(world_name, dimension)

        # Check if the schematic directory exists
        if not os.path.exists(schematic_directory):
            os.makedirs(schematic_directory)
            sample_positions = all_sample_positions
        else:
            # Get set of existing schematic hashes
            existing_hashes = set(
                {
                    f.split(".")[0]
                    for f in os.listdir(schematic_directory)
                    if f.endswith(".schem")
                }
            )

            # Map positions to their schematic hashes
            desired_hash_map = {
                self._get_schematic_hash(world_name, dimension, position): position
                for position in all_sample_positions
            }
            desired_hashes = set(desired_hash_map.keys())

            # Get the difference between the desired and existing schematics
            sample_hashes = desired_hashes - existing_hashes

            # Get the positions for the remaining hashes
            sample_positions = {desired_hash_map[hash] for hash in sample_hashes}

            # Remove any schematics that are not desired
            unwanted_hashes = existing_hashes - desired_hashes
            if unwanted_hashes:
                print(f"Removing {len(unwanted_hashes)} unwanted schematics")
            for hash in unwanted_hashes:
                try:
                    os.remove(os.path.join(schematic_directory, hash + ".schem"))
                except OSError as e:
                    print(f"Error deleting schematic {hash}.schem: {e}")

        if len(sample_positions) == 0:
            print(
                f"All {len(all_sample_positions)} samples have already been collected"
            )
            return

        # Create the schematic directory if it doesn't exist
        os.makedirs(os.path.join(self.schematic_directory, world_name), exist_ok=True)

        # Set up worker directories
        self.setup_worker_directories(
            root_directory, directory, self.num_collect_samples_workers
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
            for i in tqdm(
                range(self.num_collect_samples_workers),
                desc="Starting worker processes",
                leave=False,
            ):
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

            # Create a tqdm progress bar
            pbar = tqdm(
                total=len(all_sample_positions),
                initial=(len(all_sample_positions) - len(sample_positions)),
                desc="Collecting samples",
            )

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
                    print("--------------------")
                    self.sample_world(directory, root)
                    # Prevent further traversal into subdirectories of a world
                    dirs[:] = []
            schematic_count = sum(
                len(files) for _, _, files in os.walk(self.schematic_directory)
            )
            print("--------------------")
            print(f"Total schematics: {schematic_count}")
            print("Done sampling directory")
        except KeyboardInterrupt:
            pass

    def sample_world(self, root_directory: str, directory: str) -> None:
        """Samples a world"""
        print(f"Sampling {directory}")
        chunk_target_blocks, sample_target_blocks = self.load_target_blocks(directory)
        for dimension in self.DIMENSIONS:
            print(f"Sampling dimension: {dimension}")
            relevant_chunks = self._mark_chunks(
                root_directory, directory, dimension, chunk_target_blocks[dimension]
            )
            if len(relevant_chunks) == 0:
                print("No relevant chunks found")
                continue
            try:
                self._visualize_marked_chunks(directory, dimension, relevant_chunks)
            except Exception as e:
                print(f"Error visualizing marked chunks: {e}")
            sample_positions = self._identify_samples(
                root_directory,
                directory,
                dimension,
                sample_target_blocks[dimension],
                relevant_chunks,
            )
            if len(sample_positions) == 0:
                print("No sample positions found")
                continue
            try:
                self._visualize_sample_positions(directory, dimension, sample_positions)
            except Exception as e:
                print(f"Error visualizing sample positions: {e}")
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
        data_dir = self._get_data_directory(directory)
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        print(f"Done clearing {directory}")
