import os
from pathlib import Path

from tqdm import tqdm


def migrate_schematics(base_dir: str):
    """
    If a world directory contains any .schem files, renames the directory to 'overworld'
    and moves it under a new parent directory with the original name.

    Args:
        base_dir: The base directory containing world directories with schematics
    """
    base_path = Path(base_dir)
    schem_dirs = set()

    # Find all directories containing .schem files
    for root, _, files in os.walk(str(base_path)):
        if any(f.endswith(".schem") for f in files):
            dir_path = Path(root)
            if dir_path.name != "overworld":
                schem_dirs.add(dir_path)

    # Process each directory containing schematics
    for world_dir in tqdm(schem_dirs, desc="Processing world directories"):
        print(f"Processing {world_dir.name}...")

        temp_name = world_dir.with_name(world_dir.name + "_temp")
        world_dir.rename(temp_name)  # Rename original to temp

        new_parent = world_dir  # Create new parent with original name
        new_parent.mkdir()

        temp_name.rename(
            new_parent / "overworld"
        )  # Move temp to overworld under new parent


if __name__ == "__main__":
    schematic_dir = "data/schematics"
    migrate_schematics(schematic_dir)
