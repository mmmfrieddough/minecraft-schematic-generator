import hashlib
import json
import os
from pathlib import Path

from schempy import Schematic
from tqdm import tqdm

from common.file_paths import SCHEMATICS_DIR

from . import shape
from .configs import expand_configs


def generate_file_hash(properties: dict) -> str:
    """
    Generate a unique hash for a given set of properties.
    """
    metadata_string = json.dumps(properties, sort_keys=True)
    return hashlib.sha256(metadata_string.encode('utf-8')).hexdigest()


def get_generator_class(properties: dict) -> type:
    """
    Dispatch to the appropriate generator function based on properties.
    """
    generator_type = properties.get("generator_type")
    if generator_type == "shape":
        return shape
    elif generator_type == "pixel_art":
        raise NotImplementedError("Pixel art generator not implemented")
    elif generator_type == "modify_structure":
        raise NotImplementedError(
            "Structure modification generator not implemented")
    else:
        raise ValueError("Invalid generator type specified in properties")


def process_schematic(properties: dict, schematics_directory: str, dry_run: bool = False) -> str:
    """
    Ensure schematic is generated.
    """
    file_hash = generate_file_hash(properties)
    schematic_path = os.path.join(schematics_directory, f'{file_hash}.schem')

    # Check if the schematic needs to be generated
    if not os.path.exists(schematic_path):
        # print(f"Generating schematic for {file_hash}")
        # Determine which generator function to use based on properties
        generator_class = get_generator_class(properties)
        if generator_class:
            schematic: Schematic = generator_class.generate_schematic(
                properties)
            descriptions: list[str] = generator_class.generate_descriptions(
                properties)
            # Save the schematic data
            if schematic:
                schematic.name = descriptions[0]
                schematic.author = "mmmfrieddough"
                schematic.metadata = {'SchematicGenerator': {
                    'Hash': file_hash, 'Prompts': descriptions, 'Properties': properties}}
                if dry_run:
                    print(
                        f'Dry run: Would have saved schematic to {schematic_path}')
                    return file_hash
                schematic.save_to_file(Path(schematic_path), 2)
    return file_hash


def remove_old_samples(hashes: list[str], schematics_directory: str, dry_run: bool = False) -> None:
    """
    Remove all samples that are not in the given list of hashes.
    """
    for file in os.listdir(schematics_directory):
        file_hash = file.split('.')[0]
        if file_hash not in hashes:
            if dry_run:
                print(f'Dry run: Would have removed {file}')
                continue
            print(f"Removing old sample {file}")
            os.remove(os.path.join(schematics_directory, file))


def generate_samples(parameters_list: list[dict], dataset_name: str, dry_run: bool = False) -> None:
    """
    Generate samples for a given list of parameters.
    """
    schematics_directory = os.path.join(SCHEMATICS_DIR, dataset_name)
    os.makedirs(schematics_directory, exist_ok=True)

    hashes = []
    parameters_list_bar = tqdm(
        parameters_list, desc=f"Generating samples for {dataset_name}")
    for parameters in parameters_list_bar:
        hash = process_schematic(
            parameters, schematics_directory, dry_run=dry_run)
        hashes.append(hash)
    remove_old_samples(hashes, schematics_directory, dry_run=dry_run)


def generate_samples_from_configurations(parameters_configuration: list[dict], dataset_name: str, dry_run: bool = False) -> None:
    """
    Expand the given paramerters configuration into the full list of parameters and generate samples.
    """
    parameters_list = expand_configs(parameters_configuration)
    generate_samples(parameters_list, dataset_name, dry_run=dry_run)
