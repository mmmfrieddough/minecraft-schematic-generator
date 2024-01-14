import os
from pathlib import Path
from typing import Dict, Set, Tuple

import h5py
from schempy import Schematic
from tqdm import tqdm

from converter import SchematicArrayConverter

converter = SchematicArrayConverter()


def process_schematic(sample_name: str, schematic_path: str, group: h5py.Group) -> None:
    # Load the schematic
    try:
        schematic = Schematic.from_file(Path(schematic_path))
    except Exception as e:
        print(f"Failed to load schematic: {sample_name}")
        print(e)
        return

    # Convert the schematic to an array
    schematic_data = converter.schematic_to_array(schematic)

    # Create the group
    group = group.require_group(sample_name)

    # Create the dataset
    group.create_dataset('structure', data=schematic_data)


def split_data(dataset_path: str, split_ratios: Tuple[float, float, float]) -> Dict[str, Set[str]]:
    """
    Split the data deterministically based on the hash of the file names.

    :param dataset_path: Path to the directory containing schematic files.
    :param split_ratios: Ratios to split the data into (train, validation, test).
    :return: A dictionary with keys 'train', 'validation', and 'test' mapping to the respective file sets.
    """
    # Calculate cumulative ratios for determining splits
    cumulative_ratios = [sum(split_ratios[:i+1])
                         for i in range(len(split_ratios))]

    # Initialize the split sets
    splits = {'train': set(), 'validation': set(), 'test': set()}

    # Get all file names
    all_files = [f for f in os.listdir(dataset_path) if os.path.isfile(
        os.path.join(dataset_path, f))]

    # Assign files to splits based on the hash value of their names
    for file_name in all_files:
        # Remove the file extension to get the hash
        hash_hex = Path(file_name).stem

        # Use the hash of the file name to get a number between 0 and 1
        hash_fraction = int(hash_hex, 16) / 16**len(hash_hex)

        # Determine the split based on the hash fraction and cumulative ratios
        if hash_fraction < cumulative_ratios[0]:
            splits['train'].add(file_name)
        elif hash_fraction < cumulative_ratios[1]:
            splits['validation'].add(file_name)
        else:
            splits['test'].add(file_name)

    print(
        f"Split data into {len(splits['train'])} training samples, {len(splits['validation'])} validation samples, and {len(splits['test'])} test samples.")

    return splits


def load_schematics(schematics_dir: str, hdf5_path: str, split_ratios: Tuple[float, float, float], dataset_names: list[str] = None) -> None:
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        print(f"Loading schematics from {schematics_dir} into {hdf5_path}")

        for dataset in os.listdir(schematics_dir):
            if dataset_names and dataset not in dataset_names:
                continue

            dataset_path = os.path.join(schematics_dir, dataset)

            print(f"Processing dataset: {dataset}")

            # Split the data
            splits = split_data(dataset_path, split_ratios)

            for set_type, files in splits.items():
                set_group = hdf5_file.require_group(
                    set_type).require_group(dataset)

                files_bar = tqdm(
                    files, desc=f"Generating set: {set_type}")
                for i, schematic_file in enumerate(files_bar):
                    sample_name = os.path.splitext(schematic_file)[0]
                    schematic_path = os.path.join(
                        dataset_path, schematic_file)
                    process_schematic(sample_name, schematic_path, set_group)

        print("Finished updating HDF5 file.")
