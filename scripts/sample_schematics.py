import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def sample_directory(source_dir, target_dir, max_files_per_dir=5):
    """
    Recursively walk through source_dir and copy up to max_files_per_dir files
    from each subdirectory to target_dir, maintaining directory structure.

    Args:
        source_dir (str): Path to source directory
        target_dir (str): Path to target directory
        max_files_per_dir (int): Maximum number of files to copy from each directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Clear target directory if it exists
    if target_path.exists():
        shutil.rmtree(target_path)

    print(f"Sampling files from {source_path} to {target_path}")

    # Walk through all directories and files
    for root, _, files in tqdm(os.walk(source_path), desc="Walking directory"):
        # Convert current directory path to Path object
        current_path = Path(root)

        # Calculate relative path from source directory
        rel_path = current_path.relative_to(source_path)

        # Create corresponding directory in target
        target_dir_path = target_path / rel_path
        target_dir_path.mkdir(parents=True, exist_ok=True)

        # If there are files in current directory
        if files:
            # Randomly sample up to max_files_per_dir files
            selected_files = random.sample(files, min(max_files_per_dir, len(files)))

            # Copy selected files
            for file in selected_files:
                source_file = current_path / file
                target_file = target_dir_path / file
                shutil.copy2(source_file, target_file)


def main():
    source_directory = "data/schematics"
    target_directory = "data/schematics_sampled"
    max_files = 100

    sample_directory(source_directory, target_directory, max_files)


if __name__ == "__main__":
    main()
