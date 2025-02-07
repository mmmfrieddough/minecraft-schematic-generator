import json
import os

# Define dimensions and their file prefixes (same as WorldSampler)
DIMENSIONS = {
    "minecraft:overworld": "overworld",
    "minecraft:the_nether": "nether",
    "minecraft:the_end": "end",
}


def split_interested_blocks(directory: str) -> None:
    """
    Recursively searches for interested_blocks.json files and splits them into
    dimension-specific files.
    """
    print(f"Scanning directory: {directory}")

    # Walk through all subdirectories
    for root, _, files in os.walk(directory):
        if "interested_blocks.json" in files or "interested_blocks.json.bak" in files:
            print(f"\nProcessing: {root}")
            # Try both the original and .bak file
            input_path = os.path.join(root, "interested_blocks.json")
            if not os.path.exists(input_path):
                input_path = os.path.join(root, "interested_blocks.json.bak")

            # Read the original file
            try:
                with open(input_path, "r") as f:
                    data = json.load(f)

                # Extract chunk and sample blocks for each dimension
                for mc_dimension, prefix in DIMENSIONS.items():
                    # Get blocks for this dimension (using short name)
                    short_name = prefix.split(":")[
                        -1
                    ]  # e.g., "overworld" from "minecraft:overworld"

                    # Get chunk and sample blocks
                    chunk_blocks = data["chunk"][short_name]
                    sample_blocks = data["sample"][short_name]

                    # Save chunk blocks
                    chunk_filename = f"{prefix}_chunk_blocks.json"
                    chunk_path = os.path.join(root, chunk_filename)
                    with open(chunk_path, "w") as f:
                        json.dump(chunk_blocks, f, indent=2)
                    print(f"Created {chunk_filename}")

                    # Save sample blocks
                    sample_filename = f"{prefix}_sample_blocks.json"
                    sample_path = os.path.join(root, sample_filename)
                    with open(sample_path, "w") as f:
                        json.dump(sample_blocks, f, indent=2)
                    print(f"Created {sample_filename}")

                # If we processed the original file (not .bak), back it up
                if not input_path.endswith(".bak"):
                    backup_path = input_path + ".bak"
                    os.rename(input_path, backup_path)
                    print(
                        f"Original file backed up to: {os.path.basename(backup_path)}"
                    )

            except json.JSONDecodeError:
                print(f"Error: Could not parse {input_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python split_interested_blocks.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    split_interested_blocks(directory)
    print("\nDone!")
