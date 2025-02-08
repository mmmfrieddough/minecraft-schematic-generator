import json
import multiprocessing
import os
from multiprocessing import Manager, Pool, RLock
from pathlib import Path
from typing import Dict, Set, Tuple

import h5py
import numpy as np
from h5py import File
from schempy import Block, Schematic
from tqdm import tqdm

from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    SchematicArrayConverter,
    SharedDictBlockTokenMapper,
)


class SchematicLoader:
    block_properties_to_remove = {
        "minecraft:sugar_cane": ["age"],
        "minecraft:lectern": ["powered"],
        "minecraft:daylight_detector": ["power"],
        "minecraft:note_block": ["instrument", "note", "powered"],
        "minecraft:observer": ["powered"],
        "minecraft:dispenser": ["triggered"],
        "minecraft:hopper": ["enabled"],
        "minecraft:tripwire": ["powered"],
        "minecraft:fire": ["age"],
        "minecraft:barrel": ["open"],
        "minecraft:cactus": ["age"],
        "minecraft:tnt": ["unstable"],
        "minecraft:chorus_flower": ["age"],
        "minecraft:dropper": ["triggered"],
    }
    block_id_contains_properties_to_remove = {
        "leaves": ["distance", "persistent"],
        "door": ["powered"],
        "button": ["powered"],
        "pressure_plate": ["powered", "power"],
    }

    _schematic_array_converter: SchematicArrayConverter = None

    @staticmethod
    def clean_block_properties(block: Block) -> None:
        if not block or not block.properties:
            return
        for block_id, properties in SchematicLoader.block_properties_to_remove.items():
            if block.id == block_id:
                for property in properties:
                    block.properties.pop(property, None)
        for (
            block_id_contains,
            properties,
        ) in SchematicLoader.block_id_contains_properties_to_remove.items():
            if block_id_contains in block.id:
                for property in properties:
                    block.properties.pop(property, None)

    @staticmethod
    def clean_schematic(schematic: Schematic) -> None:
        for block in schematic.get_block_palette().keys():
            SchematicLoader.clean_block_properties(block)

    @staticmethod
    def get_schematic_data(sample_name: str, schematic_path: str) -> np.ndarray:
        # Load the schematic
        try:
            schematic = Schematic.from_file(Path(schematic_path))
        except Exception as e:
            print(f"Failed to load schematic: {sample_name}")
            raise e

        # Clean the schematic
        SchematicLoader.clean_schematic(schematic)

        # Convert the schematic to an array
        # The update_mapping flag is set to True because loading the schematics here is the source of the mapping
        schematic_data = SchematicLoader._schematic_array_converter.schematic_to_array(
            schematic, update_mapping=True
        )

        return schematic_data

    @staticmethod
    def split_data(
        all_files: set, split_ratios: Tuple[float, float, float]
    ) -> Dict[str, Set[str]]:
        """
        Split the data deterministically based on the hash of the file names.

        :param all_files: Set of all file names to split.
        :param split_ratios: Ratios to split the data into (train, validation, test).
        :return: A dictionary with keys 'train', 'validation', and 'test' mapping to the respective file sets.
        """
        # Calculate cumulative ratios for determining splits
        cumulative_ratios = [
            sum(split_ratios[: i + 1]) for i in range(len(split_ratios))
        ]

        # Initialize the split sets
        splits = {"train": set(), "validation": set(), "test": set()}

        # Assign files to splits based on the hash value of their names
        for file_name in all_files:
            # Remove the file extension to get the hash
            hash_hex = Path(file_name).stem

            # Use the hash of the file name to get a number between 0 and 1
            hash_fraction = int(hash_hex, 16) / 16 ** len(hash_hex)

            # Determine the split based on the hash fraction and cumulative ratios
            if hash_fraction < cumulative_ratios[0]:
                splits["train"].add(file_name)
            elif hash_fraction < cumulative_ratios[1]:
                splits["validation"].add(file_name)
            else:
                splits["test"].add(file_name)

        return splits

    @staticmethod
    def _init_worker(shared_dict, lock):
        """Initialize worker with shared mapping dictionary"""
        block_token_mapper = SharedDictBlockTokenMapper(shared_dict, lock)
        block_token_converter = BlockTokenConverter(block_token_mapper)
        SchematicLoader._schematic_array_converter = SchematicArrayConverter(
            block_token_converter
        )

    @staticmethod
    def _process_schematic(args):
        """Helper function to process individual schematic files"""
        dataset_path, name = args
        schematic_path = os.path.join(dataset_path, f"{name}.schem")
        try:
            structure = SchematicLoader.get_schematic_data(name, schematic_path)
            return name, structure
        except Exception as e:
            print(f"Failed to process schematic: {name}")
            print(e)
            return None

    @staticmethod
    def _sync_dataset(
        hdf5_file: File, set_type: str, dataset: str, required_names: set[str]
    ) -> tuple[set[str], set[str]]:
        """Sync the dataset by removing outdated samples and returning the names that need processing"""
        existing_names = [
            n.decode("utf-8") for n in hdf5_file[set_type][dataset]["names"][:]
        ]
        names_to_remove = set(existing_names) - required_names

        if names_to_remove:
            print(
                f"Removing {len(names_to_remove)}/{len(existing_names)} outdated samples from {dataset} {set_type}"
            )

            # Create mask for elements to keep
            keep_mask = [name not in names_to_remove for name in existing_names]

            # Get the existing data
            existing_structures = hdf5_file[set_type][dataset]["structures"][:]
            filtered_names = [n for n, k in zip(existing_names, keep_mask) if k]
            filtered_structures = existing_structures[keep_mask]

            # Delete old datasets
            del hdf5_file[set_type][dataset]["names"]
            del hdf5_file[set_type][dataset]["structures"]

            # Create new filtered datasets
            dt = h5py.string_dtype(encoding="utf-8")
            hdf5_file[set_type][dataset].create_dataset(
                "names", data=filtered_names, maxshape=(None,), dtype=dt
            )
            hdf5_file[set_type][dataset].create_dataset(
                "structures",
                data=filtered_structures,
                maxshape=(None, *filtered_structures.shape[1:]),
            )

        # Return names that need to be processed (names in required_names but not in existing_names)
        existing_names = set(existing_names) - names_to_remove
        names_to_process = required_names - existing_names

        return existing_names, names_to_process

    @staticmethod
    def load_schematics(
        schematics_dir: str,
        hdf5_path: str,
        split_ratios: Tuple[float, float, float],
        dataset_names: list[str] = None,
        validation_only_datasets: list[str] = [],
        num_workers: int | None = None,
    ) -> None:
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"Loading schematics from {schematics_dir} into {hdf5_path}")

        # Go through all schematic files
        dataset_files = {}
        print("Searching for schematic files...")
        for root, _, names in os.walk(schematics_dir):
            print(f"Processing {root}")
            schematic_names = {
                os.path.splitext(n)[0] for n in names if n.endswith(".schem")
            }
            if not schematic_names:
                print("No schematics found in the directory.")
                continue
            print(f"Found {len(schematic_names)} schematic files.")

            dataset = os.path.relpath(root, schematics_dir)
            print(f"Dataset: {dataset}")
            if dataset_names and dataset not in dataset_names:
                print("Skipping dataset.")
                continue

            # Pre-calculate splits
            if dataset in validation_only_datasets:
                dataset_files[dataset]["validation"] = schematic_names
            else:
                dataset_files[dataset] = SchematicLoader.split_data(
                    schematic_names, split_ratios
                )

        if not dataset_files:
            print("No schematic files found.")
            return

        # Read existing datasets and mapping
        with h5py.File(hdf5_path, "a") as hdf5_file:
            # Go through all datasets in the file
            for set_type in ["train", "validation", "test"]:
                if set_type in hdf5_file:
                    existing_datasets = list(hdf5_file[set_type].keys())
                    for dataset in existing_datasets:
                        # Check if the dataset is not in the input files
                        if (
                            dataset not in dataset_files
                            or not dataset_files[dataset][set_type]
                        ):
                            print(
                                f"Removing unrecognized dataset: {dataset} from {set_type}"
                            )
                            del hdf5_file[set_type][dataset]
                            continue

                        # Sync the dataset by removing outdated samples and return the names that need processing
                        required_names = set(dataset_files[dataset][set_type])
                        need_processing_names = SchematicLoader._sync_dataset(
                            hdf5_file, set_type, dataset, required_names
                        )
                        dataset_files[dataset][set_type] = need_processing_names

            # Load the mapping
            if "mapping" in hdf5_file:
                mapping_group = hdf5_file["mapping"]
                mapping_str = mapping_group["block_to_token"][()]
                mapping_json = json.loads(mapping_str)
                block_to_token = dict(mapping_json)
            else:
                block_to_token = {}

        # Create a manager to share the mapping between processes
        manager = Manager()
        shared_block_to_token = manager.dict(block_to_token)
        lock = RLock()

        # Calculate total number of files to process
        total_files = sum(
            len(files) for info in dataset_files.values() for files in info.values()
        )

        # Create process pool
        with Pool(
            num_workers,
            initializer=SchematicLoader._init_worker,
            initargs=(shared_block_to_token, lock),
        ) as pool:
            pbar_overall = tqdm(total=total_files, desc="Processing schematic files")

            for dataset, info in dataset_files.items():
                for set_type, names in info.items():
                    # Prepare arguments for parallel processing
                    path = os.path.join(schematics_dir, dataset)
                    process_args = [(path, name) for name in names]

                    # Process files in parallel
                    results = tqdm(
                        pool.imap(SchematicLoader._process_schematic, process_args),
                        total=len(process_args),
                        desc=f"Processing {dataset} {set_type}",
                        leave=False,
                    )

                    # Filter out None results and separate names and structures
                    valid_results = [r for r in results if r is not None]
                    if valid_results:
                        names, structures = zip(*valid_results)
                        block_to_token_str = json.dumps(dict(shared_block_to_token))

                        # Write to HDF5 file
                        with h5py.File(hdf5_path, "a") as hdf5_file:
                            # Save updated mapping
                            mapping_group = hdf5_file.require_group("mapping")
                            if "block_to_token" in mapping_group:
                                del mapping_group["block_to_token"]
                            mapping_group.create_dataset(
                                "block_to_token", data=block_to_token_str
                            )

                            # Create or update dataset group
                            set_group = hdf5_file.require_group(set_type).require_group(
                                dataset
                            )

                            if "names" in set_group and "structures" in set_group:
                                # Append to existing datasets
                                current_size = len(set_group["names"])
                                new_size = current_size + len(names)

                                set_group["names"].resize((new_size,))
                                set_group["names"][current_size:] = names

                                set_group["structures"].resize(
                                    (new_size, *set_group["structures"].shape[1:])
                                )
                                set_group["structures"][current_size:] = structures
                            else:
                                # Create new datasets
                                dt = h5py.string_dtype(encoding="utf-8")
                                set_group.create_dataset(
                                    "names", data=names, maxshape=(None,), dtype=dt
                                )
                                set_group.create_dataset(
                                    "structures",
                                    data=structures,
                                    maxshape=(None, *structures[0].shape),
                                )

                    pbar_overall.update(len(names))

            pbar_overall.close()
            print("Finished updating HDF5 file.")
