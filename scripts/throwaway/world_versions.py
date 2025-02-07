import os
from collections import defaultdict
from pathlib import Path

import amulet
import matplotlib.pyplot as plt


def get_world_version(directory: str) -> tuple:
    """Gets the version of the Minecraft world"""
    world = amulet.load_level(directory)
    try:
        version = world.translation_manager._get_version_number(
            world.level_wrapper.platform, world.level_wrapper.version
        )
        return version
    finally:
        world.close()


def find_minecraft_worlds(root_dir: str) -> dict:
    """
    Search through directories for Minecraft worlds and track their versions.
    Returns a dictionary mapping versions to lists of world directories.
    """
    version_map = defaultdict(list)

    print("Searching for Minecraft worlds...")
    for root, dirs, files in os.walk(root_dir):
        if "level.dat" in files:
            try:
                version = get_world_version(root)
                version_str = ".".join(str(x) for x in version)
                version_map[version].append(root)
                print(f"Found world at {root} (version {version_str})")
            except Exception as e:
                print(f"Error reading world at {root}: {e}")

    return version_map


def main():
    root_dir = "data/worlds"

    if not os.path.exists(root_dir):
        print("Directory does not exist!")
        return

    # Find all worlds and their versions
    version_map = find_minecraft_worlds(root_dir)

    if not version_map:
        print("\nNo Minecraft worlds found!")
        return

    # Sort versions and get the 5 oldest
    oldest_versions = sorted(version_map.keys())[:5]

    print("\n5 oldest versions found:")
    print("-" * 50)
    for version in oldest_versions:
        version_str = ".".join(str(x) for x in version)
        print(f"\nVersion {version_str}:")
        for world_dir in version_map[version]:
            print(f"  - {world_dir}")

    # Create bar graph
    versions = sorted(version_map.keys())
    version_labels = [".".join(str(x) for x in v) for v in versions]
    world_counts = [len(version_map[v]) for v in versions]

    plt.figure(figsize=(10, 6))
    plt.bar(version_labels, world_counts)
    plt.title("Minecraft World Versions Distribution")
    plt.xlabel("Version")
    plt.ylabel("Number of Worlds")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    output_dir = Path("data/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "world_versions.png")
    print(f"\nBar graph saved to {output_dir / 'world_versions.png'}")


if __name__ == "__main__":
    main()
