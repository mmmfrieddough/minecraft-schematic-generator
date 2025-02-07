import json


def remap_block_states(old_file_path, new_file_path, output_file_path):
    # Load both JSON files
    with open(old_file_path, "r") as f:
        old_mapping = json.load(f)

    with open(new_file_path, "r") as f:
        new_mapping = json.load(f)

    # Create new mapping using new keys but old values
    final_mapping = {}
    missing_blocks = []

    # For each block in the new (clean) file
    for block_state in new_mapping.keys():
        if block_state in old_mapping:
            final_mapping[block_state] = old_mapping[block_state]
        else:
            missing_blocks.append(block_state)

    # Validation checks
    print(f"Old mapping size: {len(old_mapping)}")
    print(f"New mapping size: {len(new_mapping)}")
    print(f"Final mapping size: {len(final_mapping)}")

    if missing_blocks:
        print(
            "WARNING: The following blocks from the new file were not found in the old file:"
        )
        for block in missing_blocks:
            print(f"  - {block}")
        raise ValueError("Some blocks could not be mapped!")

    if len(final_mapping) != len(new_mapping):
        raise ValueError("Final mapping size doesn't match new mapping size!")

    # Save the result
    with open(output_file_path, "w") as f:
        json.dump(final_mapping, f)


# Usage
remap_block_states(
    "block_state_mapping_old.json",
    "block_state_mapping_new.json",
    "block_state_mapping_final.json",
)
