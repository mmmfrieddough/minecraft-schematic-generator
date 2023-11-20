import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI
from schempy import Schematic
from tqdm import tqdm

from common.file_paths import SCHEMATICS_DIR

from . import shape
from .configs import expand_configs

client = OpenAI()


def generate_file_hash(properties):
    """
    Generate a unique hash for a given set of properties.
    """
    metadata_string = json.dumps(properties, sort_keys=True)
    return hashlib.sha256(metadata_string.encode('utf-8')).hexdigest()


def get_generator_class(properties):
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


def process_schematic(file_hash, properties, dry_run: bool = False):
    """
    Ensure schematic is generated.
    """
    schematic_path = os.path.join(
        SCHEMATICS_DIR, properties["generator_type"], f'{file_hash}.schem')

    # Check if the schematic needs to be generated
    if not os.path.exists(schematic_path):
        # print(f"Generating schematic for {file_hash}")
        # Determine which generator function to use based on properties
        generator_class = get_generator_class(properties)
        if generator_class:
            schematic: Schematic = generator_class.generate_schematic(
                properties)
            description: str = generator_class.generate_description(properties)
            # Save the schematic data
            if schematic:
                schematic.name = description
                schematic.author = "mmmfrieddough"
                schematic.metadata = {'SchematicGenerator': {
                    'Hash': file_hash, 'Properties': properties}}
                if dry_run:
                    print(
                        f'Dry run: Would have saved schematic to {schematic_path}')
                    return
                schematic.save_to_file(Path(schematic_path), 2)


def generate_sample(parameters, dry_run: bool = False):
    """
    Process all data types for a given set of parameters.
    """
    file_hash = generate_file_hash(parameters)
    # print(f"Processing {file_hash}")
    process_schematic(file_hash, parameters, dry_run=dry_run)
    return file_hash


def remove_old_samples(hashes, dry_run: bool = False):
    """
    Remove all samples that are not in the given list of hashes.
    """
    for root_dir in os.listdir(SCHEMATICS_DIR):
        root_path = os.path.join(SCHEMATICS_DIR, root_dir)
        if os.path.isdir(root_path):
            for file in os.listdir(root_path):
                file_hash = file.split('.')[0]
                if file_hash not in hashes:
                    if dry_run:
                        print(f'Dry run: Would have removed {file}')
                        continue
                    print(f"Removing old sample {file}")
                    os.remove(os.path.join(root_path, file))


def generate_samples(parameters_list, dry_run=False):
    """
    Generate samples for a given list of parameters.
    """
    # Ensure directories exist
    os.makedirs(SCHEMATICS_DIR, exist_ok=True)

    hashes = []
    parameters_list_bar = tqdm(parameters_list, desc="Generating samples")
    for parameters in parameters_list_bar:
        hash = generate_sample(parameters, dry_run=dry_run)
        hashes.append(hash)
    remove_old_samples(hashes, dry_run=dry_run)


def generate_samples_from_configurations(parameters_configuration, dry_run=False):
    """
    Expand the given paramerters configuration into the full list of parameters and generate samples.
    """
    parameters_list = expand_configs(parameters_configuration)
    generate_samples(parameters_list, dry_run=dry_run)
