import json
import os
from pathlib import Path
from typing import Dict, Set, Tuple

import h5py
import numpy as np
from openai import OpenAI
from schempy import Schematic
from tqdm import tqdm

from converter import SchematicArrayConverter

client = OpenAI()
converter = SchematicArrayConverter()


def process_schematic(sample_name: str, schematic_path: str, hdf5_group: h5py.Group) -> None:
    # print(f"Processing schematic: {sample_name}")

    # Load the schematic
    schematic = Schematic.from_file(Path(schematic_path))

    # Get the values we need
    embedding = client.embeddings.create(
        input=schematic.name, model="text-embedding-ada-002").data[0].embedding
    schematic_data = converter.schematic_to_array(schematic)
    properties = schematic.metadata['SchematicGenerator']

    # Update the HDF5 group
    update_hdf5_group(hdf5_group, sample_name, embedding,
                      schematic_data, properties)


def update_hdf5_group(hdf5_group: h5py.Group, sample_name: str, embedding: np.ndarray, schematic_data: np.ndarray, properties: dict) -> None:
    # Create a group for the sample
    sample_group = hdf5_group.require_group(sample_name)

    # Create a dataset for the embedding within the sample group
    sample_group.create_dataset('features', data=embedding)

    # Create a dataset for the schematic data within the sample group
    sample_group.create_dataset('target', data=schematic_data)

    # Serialize the properties to a string and save it as an attribute
    properties_string = json.dumps(properties)
    sample_group.attrs['properties'] = properties_string


def remove_deleted_samples(hdf5_group: h5py.Group, existing_samples: Set[str]) -> None:
    for sample_name in existing_samples:
        del hdf5_group[sample_name]
        print(
            f"Removed {sample_name} from HDF5 as it no longer belongs to this group.")


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


def load_schematics(schematics_dir: str, hdf5_path: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
    with h5py.File(hdf5_path, 'a') as hdf5_file:
        print(f"Loading schematics from {schematics_dir} into {hdf5_path}")

        existing_splits = set(hdf5_file.keys())

        for generator_type in os.listdir(schematics_dir):
            generator_path = os.path.join(schematics_dir, generator_type)

            if not os.path.isdir(generator_path):
                continue

            print(f"Processing generator type: {generator_type}")

            # Split the data
            splits = split_data(generator_path, split_ratios)

            for set_type, files in splits.items():
                set_group = hdf5_file.require_group(
                    set_type).require_group(generator_type)
                existing_samples = set(set_group.keys())

                files_bar = tqdm(
                    files, desc=f"Updating set: {set_type} for generator: {generator_type}")
                for i, schematic_file in enumerate(files_bar):
                    sample_name = os.path.splitext(schematic_file)[0]

                    if sample_name in existing_samples:
                        # print(
                        #     f"Skipping {sample_name} as it already exists in the HDF5 file.")
                        existing_samples.remove(sample_name)
                        continue

                    schematic_path = os.path.join(
                        generator_path, schematic_file)
                    process_schematic(sample_name, schematic_path, set_group)

                    if i % 50 == 0:
                        hdf5_file.flush()

                hdf5_file.flush()
                remove_deleted_samples(set_group, existing_samples)

        # Remove any generator type groups that no longer exist
        for split in existing_splits:
            for group_name in hdf5_file[split].keys():
                if group_name not in os.listdir(schematics_dir):
                    del hdf5_file[split][group_name]
                    print(
                        f"Removed group {group_name} from HDF5 as it no longer has a corresponding directory.")

        print("Finished updating HDF5 file.")
