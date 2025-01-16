import multiprocessing
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Set, Tuple

import h5py
from schempy import Block, Schematic
from tqdm import tqdm

from minecraft_schematic_generator.converter import SchematicArrayConverter

converter = SchematicArrayConverter()

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


def clean_block_properties(block: Block) -> None:
    if not block or not block.properties:
        return
    for block_id, properties in block_properties_to_remove.items():
        if block.id == block_id:
            for property in properties:
                block.properties.pop(property, None)
    for block_id_contains, properties in block_id_contains_properties_to_remove.items():
        if block_id_contains in block.id:
            for property in properties:
                block.properties.pop(property, None)


def clean_schematic(schematic: Schematic) -> None:
    for block in schematic.get_block_palette().keys():
        clean_block_properties(block)


def get_schematic_data(sample_name: str, schematic_path: str) -> None:
    # Load the schematic
    try:
        schematic = Schematic.from_file(Path(schematic_path))
    except Exception as e:
        print(f"Failed to load schematic: {sample_name}")
        raise e
        # print(e)
        # return

    # Clean the schematic
    clean_schematic(schematic)

    # Convert the schematic to an array
    schematic_data = converter.schematic_to_array(schematic, update_mapping=True)

    return schematic_data


def split_data(
    dataset_path: str, split_ratios: Tuple[float, float, float]
) -> Dict[str, Set[str]]:
    """
    Split the data deterministically based on the hash of the file names.

    :param dataset_path: Path to the directory containing schematic files.
    :param split_ratios: Ratios to split the data into (train, validation, test).
    :return: A dictionary with keys 'train', 'validation', and 'test' mapping to the respective file sets.
    """
    # Calculate cumulative ratios for determining splits
    cumulative_ratios = [sum(split_ratios[: i + 1]) for i in range(len(split_ratios))]

    # Initialize the split sets
    splits = {"train": set(), "validation": set(), "test": set()}

    # Get all file names
    all_files = [
        f
        for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
    ]

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


def process_schematic_file(args):
    """Helper function to process individual schematic files"""
    file, dataset_path, sample_name = args
    schematic_path = os.path.join(dataset_path, file)
    try:
        structure = get_schematic_data(sample_name, schematic_path)
        return sample_name, structure
    except Exception as e:
        print(f"Failed to process schematic: {file}")
        print(e)
        return None


def load_schematics(
    schematics_dir: str,
    hdf5_path: str,
    split_ratios: Tuple[float, float, float],
    dataset_names: list[str] = None,
    validation_only_datasets: list[str] = [],
    num_workers: int = None,
) -> None:
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Loading schematics from {schematics_dir} into {hdf5_path}")

    total_files = sum([len(files) for _, _, files in os.walk(schematics_dir)])
    pbar_overral = tqdm(total=total_files, desc="Processing schematic files")

    # Create process pool outside the loop
    with Pool(num_workers) as pool:
        for root, dirs, files in os.walk(schematics_dir):
            for dataset in dirs:
                if dataset_names and dataset not in dataset_names:
                    continue

                dataset_path = os.path.join(root, dataset)

                # Check if the directory contains any schematic files, break as soon as one is found
                has_schematic = False
                for file in os.listdir(dataset_path):
                    if file.endswith(".schem"):
                        has_schematic = True
                        break

                if not has_schematic:
                    continue  # Skip this directory if no schematic files are found

                # Split the data
                if dataset in validation_only_datasets:
                    splits = {"validation": set()}
                    for file_name in os.listdir(dataset_path):
                        splits["validation"].add(file_name)
                else:
                    splits = split_data(dataset_path, split_ratios)

                for set_type, files in splits.items():
                    # Prepare arguments for parallel processing
                    process_args = [
                        (file, dataset_path, os.path.splitext(file)[0])
                        for file in files
                    ]

                    # Process files in parallel
                    results = list(
                        tqdm(
                            pool.imap(process_schematic_file, process_args),
                            total=len(process_args),
                            desc=f"Processing {dataset} {set_type}",
                            leave=False,
                        )
                    )

                    # Filter out None results and separate names and structures
                    valid_results = [r for r in results if r is not None]
                    names, structures = (
                        zip(*valid_results) if valid_results else ([], [])
                    )

                    with h5py.File(hdf5_path, "a") as hdf5_file:
                        set_group = hdf5_file.require_group(set_type).require_group(
                            dataset
                        )
                        set_group.create_dataset("names", data=names)
                        set_group.create_dataset("structures", data=structures)

                    pbar_overral.update(len(files))

        print("Finished updating HDF5 file.")
