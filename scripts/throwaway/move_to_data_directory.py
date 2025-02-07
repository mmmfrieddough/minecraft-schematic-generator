import os
import shutil
from pathlib import Path

# Define the file patterns we're looking for
FILE_PATTERNS = [
    # Block definition files
    "*_chunk_blocks.json",
    "*_sample_blocks.json",
    # Progress files
    "*_chunk_progress.pkl",
    "*_sample_progress.pkl",
    # Visualization files
    "*_selected_chunks.png",
    "*_sample_positions.png",
    # Backup files
    "interested_blocks.json.bak",
]


def move_files_to_data_directory(directory: str) -> None:
    """
    Recursively searches for relevant files and moves them to .minecraft_schematic_generator subdirectory
    """
    print(f"Scanning directory: {directory}")

    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        # Check if any of our patterns match files in this directory
        matching_files = []
        for pattern in FILE_PATTERNS:
            matching_files.extend(Path(root).glob(pattern))

        if matching_files:
            print(f"\nProcessing: {root}")

            # Create data directory
            data_dir = os.path.join(root, ".minecraft_schematic_generator")
            os.makedirs(data_dir, exist_ok=True)

            # Move each file
            for file_path in matching_files:
                dest_path = os.path.join(data_dir, file_path.name)
                try:
                    # Check if destination already exists
                    if os.path.exists(dest_path):
                        print(
                            f"Skipping {file_path.name} - already exists in data directory"
                        )
                        continue

                    shutil.move(str(file_path), dest_path)
                    print(f"Moved {file_path.name}")
                except Exception as e:
                    print(f"Error moving {file_path.name}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python move_to_data_directory.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    move_files_to_data_directory(directory)
    print("\nDone!")
