import json

import h5py

from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    DictBlockTokenMapper,
)
from minecraft_schematic_generator.model.structure_transformer import (
    StructureTransformer,
)


def main():
    hdf5_path = "data/data_v6.h5"

    with h5py.File(hdf5_path, "a") as hdf5_file:
        mapping_group = hdf5_file.require_group("mapping")
        mapping_str = mapping_group["block_to_token"][()]
        mapping_json = json.loads(mapping_str)
        mapping = dict(mapping_json)

    block_token_mapper = DictBlockTokenMapper(mapping)
    block_token_converter = BlockTokenConverter(block_token_mapper)

    blocks = [
        block_token_converter.token_to_universal_block(i)
        for i in range(1, block_token_converter.get_unused_token())
    ]

    before_total = len(mapping)
    for block in blocks:
        # Rotations
        for rotation in range(1, 4):
            rotated_block = StructureTransformer.rotate_block_properties(
                block, rotation
            )
            block_token_converter.universal_block_to_token(
                rotated_block, update_mapping=True
            )

        # Flips
        flipped_block = StructureTransformer.flip_block_properties(block)
        block_token_converter.universal_block_to_token(
            flipped_block, update_mapping=True
        )

    print(f"Before: {before_total}")
    print(f"After: {len(mapping)}")

    with h5py.File(hdf5_path, "a") as hdf5_file:
        mapping_str = json.dumps(mapping)
        mapping_group = hdf5_file.require_group("mapping")
        del mapping_group["block_to_token"]
        mapping_group.create_dataset("block_to_token", data=mapping_str)


if __name__ == "__main__":
    main()
