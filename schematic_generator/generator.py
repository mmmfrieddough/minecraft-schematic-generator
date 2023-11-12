from common.file_paths import METADATA_DIR, SCHEMATICS_DIR, DESCRIPTIONS_DIR, EMBEDDINGS_DIR, TARGET_TENSORS_DIR
import hashlib
import json
import os

from . import shape
import nbtlib
import numpy as np
from litemapy import BlockState, Region, Schematic
from nbtlib import File
from openai import OpenAI
from .configs import expand_configs
from converter.converter import RegionTensorConverter

client = OpenAI()
converter = RegionTensorConverter()

DIRS = [METADATA_DIR, SCHEMATICS_DIR, DESCRIPTIONS_DIR,
        EMBEDDINGS_DIR, TARGET_TENSORS_DIR]


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


def process_metadata(file_hash, properties, dry_run: bool = False):
    """
    Ensure metadata is generated.
    """
    metadata_path = os.path.join(METADATA_DIR, f'{file_hash}.json')

    # Check if the metadata needs to be generated
    if not os.path.exists(metadata_path):
        print(f"Generating metadata for {file_hash}")
        metadata = {
            "file_hash": file_hash,
            "properties": properties
        }
        # Save the metadata
        if dry_run:
            print(f'Dry run: Would have saved metadata to {metadata_path}')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)


def process_schematic(file_hash, properties, dry_run: bool = False):
    """
    Ensure schematic is generated.
    """
    schematic_path = os.path.join(SCHEMATICS_DIR, f'{file_hash}.schem')

    # Check if the schematic needs to be generated
    if not os.path.exists(schematic_path):
        print(f"Generating schematic for {file_hash}")
        # Determine which generator function to use based on properties
        generator_class = get_generator_class(properties)
        if generator_class:
            schematic_data: Schematic = generator_class.generate_schematic(
                properties)
            # Save the schematic data
            if schematic_data:
                if dry_run:
                    print(
                        f'Dry run: Would have saved schematic to {schematic_path}')
                    return
                region: Region = next(iter(schematic_data.regions.values()))
                file: File = region.to_sponge_nbt()
                file.save(schematic_path)


def process_description(file_hash, properties, dry_run: bool = False):
    """
    Ensure description is generated.
    """
    description_path = os.path.join(DESCRIPTIONS_DIR, f'{file_hash}.txt')

    # Check if the description needs to be generated
    if not os.path.exists(description_path):
        print(f"Generating description for {file_hash}")
        # Determine which generator function to use based on properties
        generator_class = get_generator_class(properties)
        if generator_class:
            description: str = generator_class.generate_description(properties)
            # Save the description
            if description:
                if dry_run:
                    print(
                        f'Dry run: Would have saved description to {description_path}')
                    return
                with open(description_path, 'w') as f:
                    f.write(description)


def process_embeddings(file_hash, dry_run: bool = False):
    """
    Ensure embeddings are generated.
    """
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f'{file_hash}.npy')
    # Check if the embeddings need to be generated
    if not os.path.exists(embeddings_path):
        print(f"Generating embeddings for {file_hash}")
        # Get the description
        description_path = os.path.join(DESCRIPTIONS_DIR, f'{file_hash}.txt')
        with open(description_path, 'r') as f:
            description = f.read()

        # Get the embedding
        embedding = client.embeddings.create(
            input=description, model="text-embedding-ada-002").data[0].embedding

        # Save the embedding
        if embedding:
            if dry_run:
                print(
                    f'Dry run: Would have saved embedding to {embeddings_path}')
                return
            np.save(embeddings_path, embedding)


def process_target_tensor(file_hash, dry_run: bool = False):
    """
    Ensure target tensor is generated.
    """
    target_tensor_path = os.path.join(TARGET_TENSORS_DIR, f'{file_hash}.npy')

    # Check if the target tensor needs to be generated
    if not os.path.exists(target_tensor_path):
        print(f"Generating target tensor for {file_hash}")
        # Load the region
        schematic_path = os.path.join(SCHEMATICS_DIR, f'{file_hash}.schem')
        nbt = nbtlib.load(schematic_path)
        region, _ = Region.from_sponge_nbt(nbt)

        # Convert
        target_tensor = converter.region_to_tensor(region)
        target_tensor = target_tensor.numpy()

        # Save the target tensor
        if dry_run:
            print(
                f'Dry run: Would have saved target tensor to {target_tensor_path}')
            return
        np.save(target_tensor_path, target_tensor)


def generate_sample(parameters, dry_run: bool = False):
    """
    Process all data types for a given set of parameters.
    """
    file_hash = generate_file_hash(parameters)
    print(f"Processing {file_hash}")
    process_metadata(file_hash, parameters, dry_run=dry_run)
    process_schematic(file_hash, parameters, dry_run=dry_run)
    process_description(file_hash, parameters, dry_run=dry_run)
    process_embeddings(file_hash, dry_run=dry_run)
    process_target_tensor(file_hash, dry_run=dry_run)
    return file_hash


def remove_old_samples(hashes, dry_run: bool = False):
    """
    Remove all samples that are not in the given list of hashes.
    """
    for dir in DIRS:
        for file in os.listdir(dir):
            file_hash = file.split('.')[0]
            if file_hash not in hashes:
                if dry_run:
                    print(f'Dry run: Would have removed {file}')
                    continue
                print(f"Removing old sample {file}")
                os.remove(os.path.join(dir, file))


def generate_samples(parameters_list, dry_run=False):
    """
    Generate samples for a given list of parameters.
    """
    # Ensure directories exist
    for dir in DIRS:
        os.makedirs(dir, exist_ok=True)

    hashes = []
    for parameters in parameters_list:
        hash = generate_sample(parameters, dry_run=dry_run)
        hashes.append(hash)
        # Print completion percentage
        print(f"{len(hashes) / len(parameters_list) * 100:.2f}% complete")
    remove_old_samples(hashes, dry_run=dry_run)


def generate_samples_from_configurations(parameters_configuration, dry_run=False):
    """
    Expand the given paramerters configuration into the full list of parameters and generate samples.
    """
    parameters_list = expand_configs(parameters_configuration)
    generate_samples(parameters_list, dry_run=dry_run)
