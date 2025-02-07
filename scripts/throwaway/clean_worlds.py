import argparse
import os
import shutil
from pathlib import Path

# Directories/files to keep
KEEP_LIST = {
    "level.dat",  # Essential world data
    "session.lock",  # Required for game running
    "region",  # Block data
    "DIM-1",  # Nether
    "DIM1",  # End
    "entities",  # Entity data
    "poi",  # Points of Interest
    "data",  # Contains maps and other data
    "icon.png",  # World icon
    ".minecraft_schematic_generator",  # Special directory to preserve
}

# Directories that might contain world folders
POTENTIAL_WORLD_CONTAINERS = {
    "saves",  # Default saves directory
    "world",  # Common server world name
    "worlds",  # Common multiple worlds directory
}


def is_minecraft_world(path):
    """Check if a directory is a Minecraft world by looking for level.dat"""
    return os.path.isfile(os.path.join(path, "level.dat"))


def find_minecraft_worlds(start_path):
    """Recursively find Minecraft worlds"""
    worlds = []

    for root, dirs, _ in os.walk(start_path):
        # Skip if we're already too deep in a world
        if any(parent.name == "region" for parent in Path(root).parents):
            continue

        # Check if current directory is a world
        if is_minecraft_world(root):
            worlds.append(root)
            continue

        # Prioritize checking common MC directories
        dirs[:] = sorted(dirs, key=lambda x: x not in POTENTIAL_WORLD_CONTAINERS)

    return worlds


def analyze_world(world_path, delete=False):
    """Analyze a Minecraft world and optionally delete unnecessary files"""
    print(f"\nAnalyzing world: {world_path}")
    items_to_remove = []

    for item in os.listdir(world_path):
        item_path = os.path.join(world_path, item)

        if item not in KEEP_LIST:
            items_to_remove.append(item_path)

    if items_to_remove:
        print("Would remove:")
        for item in items_to_remove:
            print(f"  - {item}")

        if delete:
            print("Deleting items...")
            for item in items_to_remove:
                try:
                    if os.path.isfile(item):
                        os.remove(item)
                    elif os.path.isdir(item):
                        shutil.rmtree(item)
                    print(f"  Deleted: {item}")
                except Exception as e:
                    print(f"  Error deleting {item}: {e}")
    else:
        print("No unnecessary files found.")

    return len(items_to_remove)


def main():
    parser = argparse.ArgumentParser(
        description="Clean unnecessary files from Minecraft worlds"
    )
    parser.add_argument("path", help="Path to search for Minecraft worlds")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    args = parser.parse_args()

    print(f"Searching for Minecraft worlds in: {args.path}")
    print("Mode:", "Delete" if args.delete else "Dry-run")

    worlds = find_minecraft_worlds(args.path)

    if not worlds:
        print("\nNo Minecraft worlds found!")
        return

    print(f"\nFound {len(worlds)} world(s)")
    total_removable = 0

    for world in worlds:
        total_removable += analyze_world(world, args.delete)

    print("\nSummary:")
    print(f"Worlds analyzed: {len(worlds)}")
    print(
        f"Total items that {'were' if args.delete else 'would be'} removed: {total_removable}"
    )


if __name__ == "__main__":
    main()
