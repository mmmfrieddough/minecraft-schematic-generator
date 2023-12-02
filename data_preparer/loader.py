import os
from pathlib import Path
from typing import Dict, Set, Tuple

import h5py
from schempy import Schematic
from tqdm import tqdm

from converter import SchematicArrayConverter

converter = SchematicArrayConverter()


def process_schematic(sample_name: str, schematic_path: str, group: h5py.Group) -> None:
    # print(f"Processing schematic: {sample_name}")

    # Load the schematic
    schematic = Schematic.from_file(Path(schematic_path))

    # Convert the schematic to an array
    schematic_data = converter.schematic_to_array(schematic)

    # Make sure the datasets exist
    if 'names' not in group:
        group.create_dataset('names', shape=(0,), maxshape=(
            None,), dtype=h5py.string_dtype())
    if 'prompts' not in group:
        group.create_dataset('prompts', shape=(
            0,), maxshape=(None,), dtype=h5py.string_dtype())
    if 'structures' not in group:
        group.create_dataset('structures', shape=(0,) + schematic_data.shape,
                             maxshape=(None,) + schematic_data.shape, dtype=schematic_data.dtype)

    # Append the data to the datasets
    names_dataset = group['names']
    names_dataset.resize(names_dataset.shape[0] + 1, axis=0)
    names_dataset[-1] = sample_name
    prompts_dataset = group['prompts']
    prompts_dataset.resize(prompts_dataset.shape[0] + 1, axis=0)
    prompts_dataset[-1] = schematic.name
    structures_dataset = group['structures']
    structures_dataset.resize(structures_dataset.shape[0] + 1, axis=0)
    structures_dataset[-1] = schematic_data


def split_data(generator_path: str, split_ratios: Tuple[float, float, float]) -> Dict[str, Set[str]]:
    """
    Split the data deterministically based on the hash of the file names.

    :param generator_path: Path to the directory containing schematic files.
    :param split_ratios: Ratios to split the data into (train, validation, test).
    :return: A dictionary with keys 'train', 'validation', and 'test' mapping to the respective file sets.
    """
    # Calculate cumulative ratios for determining splits
    cumulative_ratios = [sum(split_ratios[:i+1])
                         for i in range(len(split_ratios))]

    # Initialize the split sets
    splits = {'train': set(), 'validation': set(), 'test': set()}

    # Get all file names
    all_files = [f for f in os.listdir(generator_path) if os.path.isfile(
        os.path.join(generator_path, f))]

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


def load_schematics(schematics_dir: str, hdf5_path: str, split_ratios: Tuple[float, float, float]) -> None:
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        print(f"Loading schematics from {schematics_dir} into {hdf5_path}")

        for generator_type in os.listdir(schematics_dir):
            generator_path = os.path.join(schematics_dir, generator_type)

            print(f"Processing generator type: {generator_type}")

            # Split the data
            splits = split_data(generator_path, split_ratios)

            for set_type, files in splits.items():
                set_group = hdf5_file.require_group(
                    set_type).require_group(generator_type)

                files_bar = tqdm(
                    files, desc=f"Generating set: {set_type} for generator: {generator_type}")
                for i, schematic_file in enumerate(files_bar):
                    sample_name = os.path.splitext(schematic_file)[0]
                    schematic_path = os.path.join(
                        generator_path, schematic_file)
                    process_schematic(sample_name, schematic_path, set_group)

        print("Finished updating HDF5 file.")
