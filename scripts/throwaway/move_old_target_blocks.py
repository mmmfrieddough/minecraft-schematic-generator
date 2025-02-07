import json
import os
import shutil
from pathlib import Path


def check_and_move_json_files(directory, execute=False):
    # Convert to Path object for easier handling
    base_dir = Path(directory)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Skip if not in .minecraft_schematic_generator directory
        if Path(root).name != ".minecraft_schematic_generator":
            continue

        # Remove 'old' from dirs to skip those directories
        if "old" in dirs:
            dirs.remove("old")

        # Filter for json files
        json_files = [f for f in files if f.lower().endswith(".json")]
        if not json_files:
            continue

        # Flag to track if we found a list-containing JSON
        found_list = False

        # Check each JSON file
        for json_file in json_files:
            file_path = Path(root) / json_file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        found_list = True
                        break
            except json.JSONDecodeError:
                print(f"Error reading {file_path}: Invalid JSON")
                continue
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        # If we found a list in any JSON file, prepare to move all JSON files
        if found_list:
            # Create 'old' directory if it doesn't exist
            old_dir = Path(root) / "old"

            if execute:
                old_dir.mkdir(exist_ok=True)

            # Process all JSON files in the directory
            for json_file in json_files:
                src = Path(root) / json_file
                dst = old_dir / json_file

                if execute:
                    try:
                        shutil.move(str(src), str(dst))
                        print(f"Moved: {src} -> {dst}")
                    except Exception as e:
                        print(f"Error moving {src}: {str(e)}")
                else:
                    print(f"Would move: {src} -> {dst}")


def main():
    check_and_move_json_files("data/worlds", execute=True)


if __name__ == "__main__":
    main()
