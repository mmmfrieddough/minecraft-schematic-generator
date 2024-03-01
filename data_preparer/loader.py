import os
from pathlib import Path
from typing import Dict, Set, Tuple

import h5py
from schempy import Schematic
from tqdm import tqdm

from converter import SchematicArrayConverter

converter = SchematicArrayConverter()

block_properties_to_remove = {
    'minecraft:sugar_cane': ['age'],
    'minecraft:lectern': ['powered'],
    'minecraft:daylight_detector': ['power'],
    'minecraft:note_block': ['instrument', 'note', 'powered'],
    'minecraft:observer': ['powered'],
    'minecraft:dispenser': ['triggered'],
    'minecraft:hopper': ['enabled'],
    'minecraft:tripwire': ['powered'],
    'minecraft:fire': ['age'],
    'minecraft:barrel': ['open'],
    'minecraft:cactus': ['age'],
    'minecraft:tnt': ['unstable'],
    'minecraft:chorus_flower': ['age'],
    'minecraft:dropper': ['triggered'],
}
block_id_contains_properties_to_remove = {
    'leaves': ['distance', 'persistent'],
    'door': ['powered'],
    'button': ['powered'],
    'pressure_plate': ['powered', 'power'],
}


def clean_schematic(schematic: Schematic) -> None:
    for block in schematic.get_block_palette().keys():
        if not block or not block.properties:
            continue
        for block_id, properties in block_properties_to_remove.items():
            if block.id == block_id:
                for property in properties:
                    block.properties.pop(property, None)
        for block_id_contains, properties in block_id_contains_properties_to_remove.items():
            if block_id_contains in block.id:
                for property in properties:
                    block.properties.pop(property, None)


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
    schematic_data = converter.schematic_to_array(schematic)

    return schematic_data


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

    return splits


def load_schematics(schematics_dir: str, hdf5_path: str, split_ratios: Tuple[float, float, float], dataset_names: list[str] = None, validation_only_datasets: list[str] = []) -> None:
    with h5py.File(hdf5_path, 'w') as hdf5_file:
        print(f"Loading schematics from {schematics_dir} into {hdf5_path}")

        total_files = sum([len(files)
                          for r, d, files in os.walk(schematics_dir)])
        pbar = tqdm(total=total_files, desc="Processing schematic files")

        for dataset in os.listdir(schematics_dir):
            if dataset_names and dataset not in dataset_names:
                continue

            dataset_path = os.path.join(schematics_dir, dataset)

            # Split the data
            if dataset in validation_only_datasets:
                splits = {'validation': set()}
                for file_name in os.listdir(dataset_path):
                    splits['validation'].add(file_name)
            else:
                splits = split_data(dataset_path, split_ratios)

            for set_type, files in splits.items():
                set_group = hdf5_file.require_group(
                    set_type).require_group(dataset)

                names_dataset = set_group.create_dataset(
                    'names', (len(files),), dtype=h5py.string_dtype())
                structures_dataset = set_group.create_dataset(
                    'structures', (len(files), 11, 11, 11), dtype='uint16')

                pbar.set_postfix(dataset=dataset, set_type=set_type)
                for i, schematic_file in enumerate(files):
                    sample_name = os.path.splitext(schematic_file)[0]
                    schematic_path = os.path.join(
                        dataset_path, schematic_file)
                    try:
                        structure = get_schematic_data(
                            sample_name, schematic_path)
                        names_dataset[i] = sample_name
                        structures_dataset[i] = structure
                    except Exception as e:
                        print(
                            f"Failed to process schematic: {schematic_file}")
                        print(e)
                        continue
                    finally:
                        pbar.update()

        pbar.close()

        print("Finished updating HDF5 file.")
