import fnmatch
import hashlib
import json
import logging
import math
import os
import pickle
import random
import shutil
import time
from collections import Counter
from multiprocessing import Process, Queue
from queue import Empty
from typing import TypedDict

import psutil
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


class TargetBlock(TypedDict):
    name: str
    properties: dict[str, list[str]]


class WorldSampler:
    DIMENSIONS = {
        "minecraft:overworld": "overworld",
        "minecraft:the_nether": "nether",
        "minecraft:the_end": "end",
    }

    def __init__(
        self,
        schematic_directory,
        temp_directory,
        progress_save_period,
        chunk_mark_radius,
        sample_offset,
        sample_size,
        sample_target_block_threshold,
        sample_minimum_air_threshold,
        sampling_purge_interval,
        chunk_search_limit=None,
        sample_search_limit=None,
        sample_limit=None,
        worker_check_period=5,
        resource_usage_limit=0.70,
        progress_per_second_target=0.1,
        worker_scaling_factor=0.5,
    ):
        self.schematic_directory = schematic_directory
        self.temp_directory = temp_directory
        self.progress_save_period = progress_save_period
        self.chunk_mark_radius = chunk_mark_radius
        self.sample_offset = sample_offset
        self.sample_size = sample_size
        self.sample_target_block_threshold = sample_target_block_threshold
        self.sample_minimum_air_threshold = sample_minimum_air_threshold
        self.sampling_purge_interval = sampling_purge_interval
        self.worker_check_period = worker_check_period
        self.resource_usage_limit = resource_usage_limit
        self.progress_per_second_target = progress_per_second_target
        self.worker_scaling_factor = worker_scaling_factor
        self.chunk_search_limit = chunk_search_limit
        self.sample_search_limit = sample_search_limit
        self.sample_limit = sample_limit

    def _get_data_directory(self, directory: str) -> str:
        """Returns the path to the data directory for a world"""
        return os.path.join(directory, ".minecraft_schematic_generator")

    def _get_world_timestamp(self, directory: str) -> int:
        """Gets the timestamp of the Minecraft world"""
        try:
            world = amulet.load_level(directory)
            timestamp = world.level_wrapper.last_played
        finally:
            world.close()
        return timestamp

    def _get_world_info(
        self, directory: str
    ) -> tuple[str, int, tuple[int, int, int], int]:
        """Gets the name, version, data version, and timestamp of the Minecraft world"""
        try:
            world = amulet.load_level(directory)
            name = world.level_wrapper.level_name
            data_version = world.level_wrapper.version
            version = world.translation_manager._get_version_number(
                world.level_wrapper.platform, data_version
            )
            timestamp = world.level_wrapper.last_played
        finally:
            world.close()
        return name, version, data_version, timestamp

    def _get_available_target_block_versions(
        self,
    ) -> dict[int, tuple[tuple[int, int, int], str]]:
        """Returns a list of available target block versions"""
        dir = os.path.join(os.path.dirname(__file__), "target_blocks")

        # Get all available version directories
        version_dirs = {}
        for item in os.listdir(dir):
            version_path = os.path.join(dir, item)
            if os.path.isdir(version_path):
                try:
                    # Split
                    version_string, data_version = item.split("_")
                    data_version = int(data_version)
                    # Split version string and pad with zeros
                    version_parts = version_string.split(".")
                    version = tuple(int(x) for x in version_parts) + (0,) * (
                        3 - len(version_parts)
                    )
                    version_dirs[data_version] = version, version_path
                except ValueError:
                    continue

        return version_dirs

    def _get_version_specific_target_block_path(
        self, world_version: tuple[int, int, int], world_data_version: int
    ) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
        """Gets the path to the version-specific file that's closest to but not older than the world version"""
        available_versions = self._get_available_target_block_versions()

        # Find the oldest version that's not older than the world version
        valid_versions = [
            data_version
            for data_version in available_versions
            if data_version >= world_data_version
        ]
        if not valid_versions:
            raise ValueError(
                f"No compatible target block files found for {world_version} {world_data_version}"
            )
        closest_data_version = min(valid_versions)
        closest_version, closest_path = available_versions[closest_data_version]

        # Return path and version
        return closest_path, closest_version, closest_data_version

    def load_target_blocks(
        self, directory: str, version: tuple[int, int, int], data_version: int
    ) -> tuple[dict[str, list[TargetBlock]], dict[str, list[TargetBlock]]]:
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
                        version_path, selected_version, selected_data_version = (
                            self._get_version_specific_target_block_path(
                                version, data_version
                            )
                        )
                        selected_version_str = ".".join(
                            str(x) for x in selected_version
                        )
                        print(
                            f"Using target blocks from version {selected_data_version} ({selected_version_str})"
                        )
                    config_path = os.path.join(version_path, filename)

                # Load the block file
                try:
                    with open(config_path, "r") as file:
                        target_blocks = json.load(file).get("blocks", [])
                except AttributeError:
                    raise ValueError(f"Error loading target blocks from {config_path}")

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

    def setup_worker_directory(
        self, root_directory: str, src_directory: str, timestamp: str, worker_num: int
    ) -> str:
        """Sets up worker directories for a given source directory"""
        COPY_LIST = {
            "level.dat",  # Essential world data
            "region",  # Block data
            "DIM-1",  # Nether
            "DIM1",  # End
        }

        # Get the target worker directory
        worker_directory = self.get_worker_directory(
            root_directory, src_directory, worker_num
        )

        # Check if the worker directory has already been set up
        if os.path.exists(worker_directory):
            return worker_directory

        # Create the worker directory
        os.makedirs(worker_directory)
        for item in os.listdir(src_directory):
            if item in COPY_LIST:
                src_path = os.path.abspath(os.path.join(src_directory, item))
                dst_path = os.path.abspath(os.path.join(worker_directory, item))
                os.symlink(src_path, dst_path)

        return worker_directory

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

    def _check_block_property(
        self, prop_key: str, accepted_values: list, block_props: dict
    ) -> bool:
        """Returns True if the block property matches the accepted values"""
        # If the block doesn't have this property at all, fail the rule
        if prop_key not in block_props:
            return False

        value = str(block_props[prop_key])
        if value not in accepted_values:
            return False

        return True

    def _check_block_rule(
        self, rule: TargetBlock, block_name: str, block_props: dict
    ) -> bool:
        """Returns True if the block matches the rule"""
        # Check name
        rule_name = rule.get("name", "*")
        if not fnmatch.fnmatch(block_name, rule_name):
            return False

        # Check properties
        required_props = rule.get("properties", {})
        for prop_name, accepted_values in required_props.items():
            if not self._check_block_property(prop_name, accepted_values, block_props):
                return False

        return True

    def _check_block_rules(
        self,
        filters: list[TargetBlock],
        block: Block,
    ) -> bool:
        """
        Checks whether a 'block' (with 'namespaced_name' and 'properties' dict) matches
        ANY of the provided block_filters (a list of rule dicts).

        :param block: Amulet universal block or a custom block object with:
                        - block.namespaced_name (str)
                        - block.properties (dict)
        :param block_filters: List of filter rules, e.g.:
            [
                {
                    "name": "universal_minecraft:trapdoor",  # wildcard supported
                    "properties": {
                        "material": ["oak", "birch"],
                        "open": ["true"]
                    }
                },
                {
                    "name": "*"
                    "properties": {
                        "material": ["oak"]
                    }
                }
            ]
        :return: True if 'block' matches at least one rule, otherwise False.
        """
        block_props = getattr(block, "properties", {})

        # Try each rule
        for rule in filters:
            if self._check_block_rule(rule, block.namespaced_name, block_props):
                # print("Match")
                # print(block.full_blockstate)
                # print(rule)
                return True

        return False

    def _check_block(
        self, filters: list[TargetBlock], block: Block, cache: dict[str, bool]
    ) -> bool:
        """Returns True if the block matches the filters"""
        cache_key = block.full_blockstate
        if cache_key not in cache:
            cache[cache_key] = self._check_block_rules(filters, block)
        return cache[cache_key]

    def _save_progress(
        self,
        directory: str,
        timestamp: int,
        dimension: str,
        name: str,
        config: dict,
        data: dict,
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
                            "timestamp": timestamp,
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
        self,
        directory: str,
        timestamp: int,
        dimension: str,
        name: str,
        current_config: dict,
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
                if progress["timestamp"] != timestamp:
                    print(f"World has been modified, resetting {name} progress")
                    return None

                return progress["data"]
            except (pickle.UnpicklingError, EOFError, AttributeError) as e:
                print(f"Error reading {name} progress file, resetting progress: {e}")
                return None
        return None

    def _chunk_contains_target_blocks(
        self,
        target_blocks: list[TargetBlock],
        chunk: Chunk,
        block_cache: dict[str, bool],
    ) -> bool:
        """Returns True if the chunk contains any of the target blocks"""
        # return random.random() < 0.01
        for block in chunk.block_palette:
            if self._check_block(target_blocks, block, block_cache):
                return True
        return False

    def _get_chunk_config(self, target_blocks: list[TargetBlock]) -> dict:
        """Returns the configuration for marking chunks"""
        return {
            "target_blocks": target_blocks,
            "mark_radius": self.chunk_mark_radius,
        }

    def _save_chunk_progress(
        self,
        directory: str,
        timestamp: int,
        dimension: str,
        target_blocks: list[TargetBlock],
        visited_chunks: set[tuple[int, int]],
        relevant_chunks: set[tuple[int, int]],
    ) -> None:
        """Save the current chunk progress to a file"""
        config = self._get_chunk_config(target_blocks)
        data = {
            "visited_chunks": visited_chunks,
            "relevant_chunks": relevant_chunks,
        }
        self._save_progress(directory, timestamp, dimension, "chunk", config, data)

    def _load_chunk_progress(
        self,
        directory: str,
        timestamp: int,
        dimension: str,
        target_blocks: list[TargetBlock],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Load the current chunk progress from a file"""
        current_config = self._get_chunk_config(target_blocks)
        data = self._load_progress(
            directory, timestamp, dimension, "chunk", current_config
        )
        return (
            (data["visited_chunks"], data["relevant_chunks"])
            if data
            else (set(), set())
        )

    def _get_sample_config(self, target_blocks: list[TargetBlock]) -> dict:
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
        timestamp: int,
        dimension: str,
        target_blocks: list[TargetBlock],
        sampled_chunks: set[tuple[int, int]],
        sample_positions: set[tuple[int, int, int]],
    ) -> None:
        """Save the current sample progress to a file"""
        config = self._get_sample_config(target_blocks)
        data = {
            "sampled_chunks": sampled_chunks,
            "sample_positions": sample_positions,
        }
        self._save_progress(directory, timestamp, dimension, "sample", config, data)

    def _load_sample_progress(
        self,
        directory: str,
        timestamp: int,
        dimension: str,
        target_blocks: list[TargetBlock],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]]]:
        """Load the current sample progress from a file"""
        current_config = self._get_sample_config(target_blocks)
        data = self._load_progress(
            directory, timestamp, dimension, "sample", current_config
        )
        return (
            (data["sampled_chunks"], data["sample_positions"])
            if data
            else (set(), set())
        )

    def _mark_chunks_worker(
        self,
        directory: str,
        dimension: str,
        target_blocks: list[TargetBlock],
        all_chunks_queue: Queue,
        visited_chunks_queue: Queue,
    ) -> None:
        """Worker function for marking chunks"""
        # Create a new block cache for this worker
        block_cache = {}

        world = None
        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)

            while True:
                # Get the next chunk to process
                try:
                    chunk_coords = all_chunks_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Check if the worker should stop
                if chunk_coords is None:
                    # Put None back into the queue for the next worker
                    all_chunks_queue.put(None)
                    break

                successful = False
                relevant_chunks = set()
                try:
                    # Load the chunk and check if it contains any target blocks
                    chunk = world.level_wrapper.load_chunk(*chunk_coords, dimension)
                    if self._chunk_contains_target_blocks(
                        target_blocks, chunk, block_cache
                    ):
                        # Add this chunk and its neighbors
                        for dx in range(
                            -self.chunk_mark_radius, self.chunk_mark_radius + 1
                        ):
                            for dz in range(
                                -self.chunk_mark_radius, self.chunk_mark_radius + 1
                            ):
                                if world.has_chunk(
                                    chunk_coords[0] + dx,
                                    chunk_coords[1] + dz,
                                    dimension,
                                ):
                                    relevant_chunks.add(
                                        (chunk_coords[0] + dx, chunk_coords[1] + dz)
                                    )
                    successful = True
                except (ChunkDoesNotExist, ChunkLoadError):
                    pass
                except Exception as e:
                    print(f"Unknown error loading chunk {chunk_coords}: {e}")
                visited_chunks_queue.put((chunk_coords, successful, relevant_chunks))
        except KeyboardInterrupt:
            pass
        finally:
            if world:
                try:
                    world.close()
                except Exception:
                    pass

    def _create_visualization(
        self,
        directory: str,
        dimension: str,
        x_coords: list[int],
        z_coords: list[int],
        intensities: list[int],
        title: str,
        colorbar: str = None,
    ) -> None:
        data_dir = self._get_data_directory(directory)
        dimension = dimension.replace(":", "_")
        path = os.path.join(data_dir, f"{dimension}_{title}.png")

        # Check if there are no coordinates to plot
        if not x_coords or not z_coords:
            # Clear any existing visualization
            if os.path.exists(path):
                os.remove(path)
            return

        os.makedirs(data_dir, exist_ok=True)

        # Use Agg backend for headless operation
        matplotlib.use("Agg")

        # Set the background for dark mode
        plt.style.use("dark_background")

        # Plotting
        plt.figure(figsize=(10, 10))
        # Add origin point in red
        plt.scatter([0], [0], c="red", s=100, zorder=1)
        # Plot the data points
        plt.scatter(
            x_coords,
            z_coords,
            c=intensities,
            cmap="cool",
            edgecolor="none",
            s=10,
            zorder=2,
        )

        plt.colorbar(label=colorbar)
        plt.xlabel("X Coordinate")
        plt.ylabel("Z Coordinate")
        plt.title(title)

        # Adjust axes limits to have the same range for a square aspect ratio
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(z_coords), max(z_coords)
        # Include 0,0 in the plot range
        xy_min = min(x_min, y_min, 0)
        xy_max = max(x_max, y_max, 0)
        plt.xlim(xy_min, xy_max)
        plt.ylim(xy_min, xy_max)

        plt.gca().set_aspect("equal", adjustable="box")

        # Save the plot as an image
        plt.savefig(path)
        plt.close()

    def _visualize_marked_chunks(
        self, directory: str, dimension: str, relevant_chunks: set[tuple[int, int]]
    ) -> None:
        if len(relevant_chunks) == 0:
            self._create_visualization(
                directory,
                dimension,
                [],
                [],
                [],
                "selected_chunks",
                "Sample Count",
            )
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
        self,
        directory: str,
        dimension: str,
        sample_positions: set[tuple[int, int, int]],
    ) -> None:
        if len(sample_positions) == 0:
            self._create_visualization(
                directory,
                dimension,
                [],
                [],
                [],
                "sample_positions",
                "Sample Count",
            )
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

    def _calculate_workers_to_start(
        self,
        current_workers: int,
        current_progress: int,
        last_progress: int,
        total: int,
        last_time: float,
    ) -> int:
        if current_progress == total:
            return 0

        # Get CPU and memory usage
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        resource_usage = max(cpu_usage, memory_usage) * 0.01

        # Calculate progress
        progress = current_progress - last_progress
        percentage_remaining_progress = progress / (total - current_progress)
        progress_per_second = percentage_remaining_progress / (time.time() - last_time)

        # Calculate the number of workers to start
        resource_scaling = (
            max(0, self.resource_usage_limit - resource_usage)
            / self.resource_usage_limit
        )
        progress_scaling = (
            max(0, self.progress_per_second_target - progress_per_second)
            / self.progress_per_second_target
        )
        workers_to_start = math.ceil(
            self.worker_scaling_factor
            * psutil.cpu_count()
            * resource_scaling
            * progress_scaling
        )

        # Limit to the cpu count
        if workers_to_start + current_workers > psutil.cpu_count():
            workers_to_start = psutil.cpu_count() - current_workers

        return workers_to_start

    def _mark_chunks(
        self,
        root_directory: str,
        directory: str,
        timestamp: int,
        dimension: str,
        target_blocks: list[TargetBlock],
    ) -> set[tuple[int, int]]:
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
            directory, timestamp, dimension, target_blocks
        )

        # Remove visited chunks from the list
        remaining_chunk_coords = all_chunk_coords - visited_chunks

        # Limit the number of chunks to check
        if self.chunk_search_limit and len(all_chunk_coords) > self.chunk_search_limit:
            remaining_chunks = max(self.chunk_search_limit - len(visited_chunks), 0)
            remaining_chunk_coords = set(
                random.sample(list(remaining_chunk_coords), remaining_chunks)
            )

        # Check if all chunks have already been visited
        if len(remaining_chunk_coords) == 0:
            print(f"All {len(all_chunk_coords)} world chunks have already been visited")
            return relevant_chunks

        # Create queues
        all_chunks_queue = Queue()
        visited_chunks_queue = Queue()

        processes: list[Process] = []

        def start_worker(i: int) -> None:
            worker_directory = self.setup_worker_directory(
                root_directory, directory, timestamp, i
            )
            process = Process(
                target=self._mark_chunks_worker,
                args=(
                    worker_directory,
                    dimension,
                    target_blocks,
                    all_chunks_queue,
                    visited_chunks_queue,
                ),
            )
            process.start()
            processes.append(process)

        pbar = None
        try:
            # Create the progress bar
            pbar = tqdm(
                total=len(visited_chunks) + len(remaining_chunk_coords),
                initial=len(visited_chunks),
                desc="Marking chunks",
            )

            # Start the first worker
            start_worker(0)

            # Add data to the queue
            for chunk_coords in remaining_chunk_coords:
                all_chunks_queue.put(chunk_coords)
            all_chunks_queue.put(None)

            # Monitor the queue and update the progress
            unsuccessful_chunks_count = 0
            last_save_time = time.time()
            last_worker_check_time = time.time()
            last_progress = pbar.n
            while (
                any(p.is_alive() for p in processes) or not visited_chunks_queue.empty()
            ):
                # Update the progress bar
                pbar.set_postfix(
                    {
                        "relevant_chunks": len(relevant_chunks),
                        "unsuccessful_chunks": unsuccessful_chunks_count,
                        "workers": len(processes),
                    }
                )

                # Process all results currently in the queue
                while not visited_chunks_queue.empty():
                    chunk_coords, successful, visited_relevant_chunks = (
                        visited_chunks_queue.get()
                    )
                    relevant_chunks.update(visited_relevant_chunks)
                    visited_chunks.add(chunk_coords)
                    if not successful:
                        unsuccessful_chunks_count += 1
                    pbar.update(1)

                # Start new workers if needed
                if time.time() - last_worker_check_time > self.worker_check_period:
                    workers_to_start = self._calculate_workers_to_start(
                        len(processes),
                        pbar.n,
                        last_progress,
                        pbar.total,
                        last_worker_check_time,
                    )

                    # Start workers
                    if workers_to_start > 0:
                        for _ in tqdm(
                            range(workers_to_start),
                            desc="Starting worker processes",
                            leave=False,
                        ):
                            start_worker(len(processes))

                    last_worker_check_time = time.time()
                    last_progress = pbar.n

                    # Restart the CPU usage counter
                    psutil.cpu_percent()

                # Save progress
                if time.time() - last_save_time > self.progress_save_period:
                    self._save_chunk_progress(
                        directory,
                        timestamp,
                        dimension,
                        target_blocks,
                        visited_chunks,
                        relevant_chunks,
                    )
                    last_save_time = time.time()

                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)
        except KeyboardInterrupt:
            if pbar:
                pbar.clear()
                pbar.close()
            print("Marking chunks interrupted, shutting down")
            raise
        finally:
            # Clear the queue
            while True:
                try:
                    all_chunks_queue.get_nowait()
                except Empty:
                    break
            all_chunks_queue.put(None)
            all_chunks_queue.close()

            time.sleep(0.1)

            # Terminate all processes
            for process in processes:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()

            # Final save
            self._save_chunk_progress(
                directory,
                timestamp,
                dimension,
                target_blocks,
                visited_chunks,
                relevant_chunks,
            )

        return relevant_chunks

    def _get_target_palette_indices(
        self,
        target_blocks: list[TargetBlock],
        chunk: Chunk,
        block_cache: dict[str, bool],
    ) -> set[int]:
        """Returns a set of indices of blocks from the chunk palette that we are targeting"""
        target_indices = set()
        for i, block in enumerate(chunk.block_palette):
            if self._check_block(target_blocks, block, block_cache):
                target_indices.add(i)
        return target_indices

    def _get_deterministic_random_offsets(
        self, chunk_coords: tuple[int, int]
    ) -> tuple[int, int, int]:
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
        target_blocks: list[TargetBlock],
        chunk_coords: tuple[int, int],
        block_cache: dict[str, bool],
    ) -> set[tuple[int, int, int]]:
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
                        target_blocks, chunk, block_cache
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
        target_blocks: list[TargetBlock],
        relevant_chunks_queue: Queue,
        sampled_chunks_queue: Queue,
    ) -> None:
        """Worker function for identifying samples"""

        # Create a new block cache for this worker
        block_cache = {}

        world = None
        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)

            while True:
                # Get the next chunk to process
                try:
                    chunk_coords = relevant_chunks_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Check if the worker should stop
                if chunk_coords is None:
                    # Put None back into the queue for the next worker
                    relevant_chunks_queue.put(None)
                    break

                successful = False
                sample_positions = set()
                try:
                    # Check if the world has the chunk
                    if world.has_chunk(*chunk_coords, dimension):
                        # Identify samples in the chunk
                        samples = self._identify_samples_in_chunk(
                            world,
                            dimension,
                            target_blocks,
                            chunk_coords,
                            block_cache,
                        )
                        sample_positions.update(samples)
                        successful = True
                except ChunkLoadError:
                    pass
                except Exception as e:
                    print(
                        f"Unknown error identifying samples in chunk {chunk_coords}: {e}"
                    )
                sampled_chunks_queue.put((chunk_coords, successful, sample_positions))
        except KeyboardInterrupt:
            pass
        finally:
            if world:
                try:
                    world.close()
                except Exception:
                    pass

    def _identify_samples(
        self,
        root_directory: str,
        directory: str,
        timestamp: int,
        dimension: str,
        target_blocks: list[TargetBlock],
        relevant_chunks: set[tuple[int, int]],
    ) -> set:
        """Identifies samples from the marked chunks"""

        # Load progress
        sampled_chunks, sample_positions = self._load_sample_progress(
            directory, timestamp, dimension, target_blocks
        )

        # Store original size of sampled chunks
        original_sampled_chunks_size = len(sampled_chunks)

        # Remove any sampled chunks that are no longer relevant
        sampled_chunks = sampled_chunks & relevant_chunks

        # If any chunks were removed (size decreased after intersection)
        if len(sampled_chunks) < original_sampled_chunks_size:
            original_sample_positions_size = len(sample_positions)

            # Convert sample positions to chunk coordinates
            position_chunk_map = {
                (x, y, z): (x // 16, z // 16) for x, y, z in sample_positions
            }
            # Keep only samples from chunks that are still relevant
            sample_positions = {
                pos
                for pos, chunk in position_chunk_map.items()
                if chunk in sampled_chunks
            }

            print(
                f"Removing {original_sample_positions_size - len(sample_positions)} unwanted sample positions"
            )

            # Save the updated sample positions
            self._save_sample_progress(
                directory,
                timestamp,
                dimension,
                target_blocks,
                sampled_chunks,
                sample_positions,
            )

        # Get all relevant chunks that have not been sampled
        remaining_relevant_chunk_positions = relevant_chunks - sampled_chunks

        # Limit the number of chunks to search
        if self.sample_search_limit and len(relevant_chunks) > self.sample_search_limit:
            remaining_relevant_chunks = max(
                self.sample_search_limit - len(sampled_chunks), 0
            )
            remaining_relevant_chunk_positions = set(
                random.sample(
                    list(remaining_relevant_chunk_positions), remaining_relevant_chunks
                )
            )

        # Check if all relevant chunks have already been sampled
        if len(remaining_relevant_chunk_positions) == 0:
            if len(relevant_chunks) > 0:
                print(
                    f"All {len(relevant_chunks)} relevant chunks have already been sampled"
                )
            else:
                print("No relevant chunks to sample")
            return sample_positions

        # Create queues
        relevant_chunks_queue = Queue()
        sampled_chunks_queue = Queue()

        processes: list[Process] = []

        def start_worker(i: int) -> None:
            worker_directory = self.setup_worker_directory(
                root_directory, directory, timestamp, i
            )
            process = Process(
                target=self._identify_samples_worker,
                args=(
                    worker_directory,
                    dimension,
                    target_blocks,
                    relevant_chunks_queue,
                    sampled_chunks_queue,
                ),
            )
            process.start()
            processes.append(process)

        pbar = None
        try:
            # Create the progress bar
            pbar = tqdm(
                total=len(sampled_chunks) + len(remaining_relevant_chunk_positions),
                initial=len(sampled_chunks),
                desc="Identifying samples from chunks",
            )

            # Start the first worker
            start_worker(0)

            # Add data to the
            for chunk_coords in remaining_relevant_chunk_positions:
                relevant_chunks_queue.put(chunk_coords)
            relevant_chunks_queue.put(None)

            # Monitor the queue and update the progress
            unsuccessful_chunks_count = 0
            last_save_time = time.time()
            last_worker_check_time = time.time()
            last_progress = pbar.n
            while (
                any(p.is_alive() for p in processes) or not sampled_chunks_queue.empty()
            ):
                # Update the progress bar
                pbar.set_postfix(
                    {
                        "sample_positions": len(sample_positions),
                        "unsuccessful_chunks": unsuccessful_chunks_count,
                        "workers": len(processes),
                    }
                )

                # Process all results currently in the queue
                while not sampled_chunks_queue.empty():
                    chunk_coords, successful, found_sample_positions = (
                        sampled_chunks_queue.get()
                    )
                    sample_positions.update(found_sample_positions)
                    sampled_chunks.add(chunk_coords)
                    if not successful:
                        unsuccessful_chunks_count += 1
                    pbar.update(1)

                # Start new workers if needed
                if time.time() - last_worker_check_time > self.worker_check_period:
                    workers_to_start = self._calculate_workers_to_start(
                        len(processes),
                        pbar.n,
                        last_progress,
                        pbar.total,
                        last_worker_check_time,
                    )

                    # Start workers
                    if workers_to_start > 0:
                        for _ in tqdm(
                            range(workers_to_start),
                            desc="Starting worker processes",
                            leave=False,
                        ):
                            start_worker(len(processes))

                    last_worker_check_time = time.time()
                    last_progress = pbar.n

                    # Restart the CPU usage counter
                    psutil.cpu_percent()

                # Save progress
                if time.time() - last_save_time > self.progress_save_period:
                    self._save_sample_progress(
                        directory,
                        timestamp,
                        dimension,
                        target_blocks,
                        sampled_chunks,
                        sample_positions,
                    )
                    last_save_time = time.time()

                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)
        except KeyboardInterrupt:
            if pbar:
                pbar.clear()
                pbar.close()
            print("Identifying samples interrupted, shutting down")
            raise
        finally:
            # Clear the queue
            while True:
                try:
                    relevant_chunks_queue.get_nowait()
                except Empty:
                    break
            relevant_chunks_queue.put(None)
            relevant_chunks_queue.close()

            time.sleep(0.1)

            # Terminate all processes
            for process in processes:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()

            # Final save
            self._save_sample_progress(
                directory,
                timestamp,
                dimension,
                target_blocks,
                sampled_chunks,
                sample_positions,
            )

        return sample_positions

    def _get_schematic_hash(
        self,
        level_name: str,
        dimension: str,
        position: tuple[int, int, int],
        timestamp: int,
    ) -> str:
        """Returns the hash of the schematic file for the given position"""
        filename = (
            level_name
            + dimension
            + str(position)
            + str(timestamp)
            + str(self.sample_size)
            + str(self.sample_size)
            + str(self.sample_size)
        )
        return hashlib.sha256(filename.encode()).hexdigest()

    def _get_dimension_directory(self, world_name: str, dimension: str) -> str:
        """Returns the directory path for a given dimension"""
        dimension_name = dimension.replace(
            "minecraft:", ""
        )  # Convert minecraft:overworld to overworld
        return os.path.join(self.schematic_directory, world_name, dimension_name)

    def _get_schematic_path(
        self,
        world_name: str,
        level_name: str,
        dimension: str,
        position: tuple[int, int, int],
        timestamp: int,
    ) -> str:
        """Returns the path to the schematic file for the given position"""
        file_hash = self._get_schematic_hash(level_name, dimension, position, timestamp)
        dimension_dir = self._get_dimension_directory(world_name, dimension)
        return os.path.join(dimension_dir, file_hash + ".schem")

    def _collect_samples_worker(
        self,
        directory: str,
        dimension: str,
        world_name: str,
        level_name: str,
        sample_positions_queue: Queue,
        sampled_positions_queue: Queue,
        timestamp: int,
    ) -> None:
        """Worker function for collecting samples"""

        world = None
        try:
            # Load the world data from the directory
            world = amulet.load_level(directory)

            purge_counter = 0
            while True:
                # Get the next position to process
                try:
                    position = sample_positions_queue.get(timeout=0.1)
                except Empty:
                    continue

                # Check if the worker should stop
                if position is None:
                    # Put None back into the queue for the next worker
                    sample_positions_queue.put(None)
                    break

                # Get inputs ready
                path = self._get_schematic_path(
                    world_name,
                    level_name,
                    dimension,
                    position,
                    timestamp,
                )
                x, y, z = position
                selection = SelectionBox(
                    (x, y, z),
                    (x + self.sample_size, y + self.sample_size, z + self.sample_size),
                )

                # Extract the structure from the world
                structure = world.extract_structure(selection, dimension)

                tmp_path = f"{directory}/tmp.schem"

                wrapper = None
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
                    if wrapper:
                        try:
                            wrapper.close()
                        except Exception:
                            pass

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
            if world:
                try:
                    world.close()
                except Exception:
                    pass

    def _collect_samples(
        self,
        root_directory: str,
        directory: str,
        level_name: str,
        timestamp: int,
        dimension: str,
        all_sample_positions: set[tuple[int, int, int]],
    ) -> None:
        """Collects samples from the world at the identified positions"""
        world_name = os.path.relpath(directory, root_directory)
        schematic_directory = self._get_dimension_directory(world_name, dimension)

        # Check if the schematic directory exists
        if not os.path.exists(schematic_directory):
            remaining_sample_positions = all_sample_positions
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
                self._get_schematic_hash(
                    level_name, dimension, position, timestamp
                ): position
                for position in all_sample_positions
            }
            desired_hashes = set(desired_hash_map.keys())

            # Get the difference between the existing and desired schematics
            unwanted_hashes = existing_hashes - desired_hashes

            # Remove any schematics that are not desired
            if unwanted_hashes:
                print(f"Removing {len(unwanted_hashes)} unwanted schematics")
                for hash in unwanted_hashes:
                    try:
                        os.remove(os.path.join(schematic_directory, hash + ".schem"))
                    except OSError as e:
                        print(f"Error deleting schematic {hash}.schem: {e}")

            # Get the difference between the desired and existing schematics
            remaining_hashes = desired_hashes - existing_hashes

            # Get the positions for the remaining hashes
            remaining_sample_positions = {
                desired_hash_map[hash] for hash in remaining_hashes
            }

        existing_samples = len(all_sample_positions) - len(remaining_sample_positions)

        # Limit the number of samples to collect
        if self.sample_limit and len(all_sample_positions) > self.sample_limit:
            remaining_samples = max(
                self.sample_limit
                - (len(all_sample_positions) - len(remaining_sample_positions)),
                0,
            )
            remaining_sample_positions = set(
                random.sample(list(remaining_sample_positions), remaining_samples)
            )

        # Check if there are any samples to collect
        if len(remaining_sample_positions) == 0:
            if len(all_sample_positions) > 0:
                print(
                    f"All {len(all_sample_positions)} samples have already been collected"
                )
            else:
                print("No samples to collect")
            if existing_samples == 0 and os.path.exists(schematic_directory):
                os.rmdir(schematic_directory)
            return

        # Create the schematic directory if it doesn't exist
        os.makedirs(schematic_directory, exist_ok=True)

        # Create queues
        sample_positions_queue = Queue()
        sampled_positions_queue = Queue()

        processes: list[Process] = []

        def start_worker(i: int) -> None:
            worker_directory = self.setup_worker_directory(
                root_directory, directory, timestamp, i
            )
            process = Process(
                target=self._collect_samples_worker,
                args=(
                    worker_directory,
                    dimension,
                    world_name,
                    level_name,
                    sample_positions_queue,
                    sampled_positions_queue,
                    timestamp,
                ),
            )
            process.start()
            processes.append(process)

        pbar = None
        try:
            # Create the progress bar
            pbar = tqdm(
                total=existing_samples + len(remaining_sample_positions),
                initial=existing_samples,
                desc="Collecting samples",
            )

            # Start the first worker
            start_worker(0)

            # Add data to the queue
            for position in remaining_sample_positions:
                sample_positions_queue.put(position)
            sample_positions_queue.put(None)

            # Monitor the queue and update the progress
            last_worker_check_time = time.time()
            last_progress = pbar.n
            while (
                any(p.is_alive() for p in processes)
                or not sampled_positions_queue.empty()
            ):
                # Update the progress bar
                pbar.set_postfix({"workers": len(processes)})

                # Process all results currently in the queue
                while not sampled_positions_queue.empty():
                    sampled_positions_queue.get()
                    pbar.update(1)

                # Start new workers if needed
                if time.time() - last_worker_check_time > self.worker_check_period:
                    workers_to_start = self._calculate_workers_to_start(
                        len(processes),
                        pbar.n,
                        last_progress,
                        pbar.total,
                        last_worker_check_time,
                    )

                    # Start workers
                    if workers_to_start > 0:
                        for _ in tqdm(
                            range(workers_to_start),
                            desc="Starting worker processes",
                            leave=False,
                        ):
                            start_worker(len(processes))

                    last_worker_check_time = time.time()
                    last_progress = pbar.n

                    # Restart the CPU usage counter
                    psutil.cpu_percent()

                # Sleep for a short time to avoid busy waiting
                time.sleep(0.1)
        except KeyboardInterrupt:
            if pbar:
                pbar.clear()
                pbar.close()
            print("Collecting samples interrupted, shutting down")
            raise
        finally:
            # Clear the queue
            while True:
                try:
                    sample_positions_queue.get_nowait()
                except Empty:
                    break
            sample_positions_queue.put(None)
            sample_positions_queue.close()

            time.sleep(0.1)

            # Terminate all processes
            for process in processes:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.terminate()

    def sample_directory(self, directory: str) -> None:
        """Samples a directory of worlds recursively."""
        try:
            print(f"Sampling directory: {directory}")
            for root, dirs, files in os.walk(directory):
                # Skip if we're at the root directory
                if root == directory:
                    continue

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
        name, version, data_version, timestamp = self._get_world_info(directory)
        print(f"Level name: {name}")
        version_str = ".".join(str(x) for x in version)
        print(f"Version: {version_str}")
        print(
            f"Last played: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}"
        )
        chunk_target_blocks, sample_target_blocks = self.load_target_blocks(
            directory, version, data_version
        )
        for dimension in self.DIMENSIONS:
            print(f"Sampling dimension: {dimension}")
            relevant_chunks = self._mark_chunks(
                root_directory,
                directory,
                timestamp,
                dimension,
                chunk_target_blocks[dimension],
            )
            try:
                self._visualize_marked_chunks(directory, dimension, relevant_chunks)
            except Exception as e:
                print(f"Error visualizing marked chunks: {e}")
            sample_positions = self._identify_samples(
                root_directory,
                directory,
                timestamp,
                dimension,
                sample_target_blocks[dimension],
                relevant_chunks,
            )
            try:
                self._visualize_sample_positions(directory, dimension, sample_positions)
            except Exception as e:
                print(f"Error visualizing sample positions: {e}")
            self._collect_samples(
                root_directory, directory, name, timestamp, dimension, sample_positions
            )
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
