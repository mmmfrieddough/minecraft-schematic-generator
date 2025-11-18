from amulet_nbt import StringTag
from amulet import Block


class BlockNormalizer:
    """Static utilities for normalizing and cleaning blocks."""

    BLOCK_PROPERTIES_TO_REMOVE = {
        # Plants
        "sugar_cane": ["age"],
        "cactus": ["age"],
        "chorus_flower": ["age"],
        "kelp": ["age"],
        "kelp_plant": ["age"],
        "cave_vines": ["age"],
        "twisting_vines": ["age"],
        "twisting_vines_plant": ["age"],
        "weeping_vines": ["age"],
        "weeping_vines_plant": ["age"],
        "bamboo": ["stage"],
        "bamboo_sapling": ["stage"],
        "sapling": ["stage"],
        "mangrove_propagule": ["stage"],
        "leaves": ["check_decay", "distance", "persistent"],
        "flower_pot": ["update"],
        # Redstone
        "door": ["powered"],
        "trapdoor": ["powered"],
        "fence_gate": ["powered"],
        "button": ["powered"],
        "observer": ["powered"],
        "lectern": ["powered"],
        "tripwire": ["powered"],
        "tripwire_hook": ["powered"],
        "lightning_rod": ["powered"],
        "waxed_copper_bulb": ["powered"],
        "waxed_exposed_copper_bulb": ["powered"],
        "waxed_weathered_copper_bulb": ["powered"],
        "waxed_oxidized_copper_bulb": ["powered"],
        "note_block": ["instrument", "note", "powered"],
        "bell": ["powered", "toggle"],
        "head": ["powered", "no_drop"],
        "wall_head": ["powered", "no_drop"],
        "pressure_plate": ["powered"],
        "light_weighted_pressure_plate": ["power"],
        "heavy_weighted_pressure_plate": ["power"],
        "daylight_detector": ["power"],
        "target": ["power"],
        "sculk_sensor": ["power", "sculk_sensor_phase"],
        "calibrated_sculk_sensor": ["power", "sculk_sensor_phase"],
        "dispenser": ["triggered"],
        "dropper": ["triggered"],
        "hopper": ["enabled"],
        # Misc
        "fire": ["age"],
        "soul_fire": ["age"],
        "barrel": ["open"],
        "tnt": ["unstable", "underwater"],
        "scaffolding": ["bottom", "distance", "stability_checked"],
        "jukebox": ["has_record"],
        "bed": ["occupied"],
    }

    BLOCK_PROPERTIES_TO_ADD = {
        # Add waterlogged property for blocks that are always waterlogged
        "kelp": {"waterlogged": StringTag("true")},
        "kelp_plant": {"waterlogged": StringTag("true")},
        "seagrass": {"waterlogged": StringTag("true")},
        "tall_seagrass": {"waterlogged": StringTag("true")},
    }

    BLOCK_NAMES_TO_CHANGE = {
        "cave_air": "air",
        "void_air": "stone",
    }

    @staticmethod
    def _clean_block_properties(block: Block) -> Block:
        if (
            not block.properties
            or block.base_name not in BlockNormalizer.BLOCK_PROPERTIES_TO_REMOVE
        ):
            return block
        block_properties: dict = block.properties
        for property in BlockNormalizer.BLOCK_PROPERTIES_TO_REMOVE[block.base_name]:
            block_properties.pop(property, None)
        if block_properties != block.properties:
            return Block(block.namespace, block.base_name, block_properties)
        return block

    @staticmethod
    def _add_block_properties(block: Block) -> Block:
        if block.base_name not in BlockNormalizer.BLOCK_PROPERTIES_TO_ADD:
            return block
        block_properties: dict = block.properties
        for property, value in BlockNormalizer.BLOCK_PROPERTIES_TO_ADD[
            block.base_name
        ].items():
            block_properties[property] = value
        if block_properties != block.properties:
            return Block(block.namespace, block.base_name, block_properties)
        return block

    @staticmethod
    def normalize(block: Block) -> Block:
        """Normalize a block by cleaning properties, adding required properties, and renaming."""
        if block.base_name in BlockNormalizer.BLOCK_NAMES_TO_CHANGE:
            block = Block(
                block.namespace,
                BlockNormalizer.BLOCK_NAMES_TO_CHANGE[block.base_name],
                block.properties,
            )
        block = BlockNormalizer._clean_block_properties(block)
        block = BlockNormalizer._add_block_properties(block)
        return block

    @staticmethod
    def preserve_waterlogging_universal(block: Block) -> Block:
        """Preserve waterlogging when converting to universal format."""
        if any(block.base_name == "water" for block in block.extra_blocks):
            return Block(
                namespace=block.namespace,
                base_name=block.base_name,
                properties={"waterlogged": StringTag("true"), **block.properties},
            )
        return block

    @staticmethod
    def preserve_waterlogging_conversion(
        source_block: Block, target_block: Block
    ) -> Block:
        """Preserve waterlogging when converting between formats."""
        waterlogged = source_block.properties.get("waterlogged")
        if waterlogged and waterlogged.value == "true":
            return Block(
                namespace=target_block.namespace,
                base_name=target_block.base_name,
                properties={"waterlogged": waterlogged, **target_block.properties},
            )
        return target_block

