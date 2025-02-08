import logging

from amulet_nbt import StringTag

from minecraft_schematic_generator.constants import (
    MINECRAFT_PLATFORM,
    MINECRAFT_VERSION,
)

from .block_token_mapper import BlockTokenMapperInterface

# Set PyMCTranslate logging level before importing it
logging.getLogger("PyMCTranslate").setLevel(logging.WARNING)
import PyMCTranslate  # noqa: E402
from amulet import Block  # noqa: E402


class BlockTokenConverter:
    block_properties_to_remove = {
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
    }

    block_names_to_change = {
        "cave_air": "air",
        "void_air": "air",
    }

    def __init__(
        self,
        block_token_mapper: BlockTokenMapperInterface,
        version_translator: PyMCTranslate.Version | None = None,
    ):
        self._block_token_mapper = block_token_mapper
        self._version_translator = (
            version_translator
            # Default to the version we are using for the project
            or PyMCTranslate.new_translation_manager().get_version(
                MINECRAFT_PLATFORM, MINECRAFT_VERSION
            )
        )

    @staticmethod
    def _clean_block_properties(block: Block) -> Block:
        if (
            not block.properties
            or block.base_name not in BlockTokenConverter.block_properties_to_remove
        ):
            return block
        block_properties: dict = block.properties
        for property in BlockTokenConverter.block_properties_to_remove[block.base_name]:
            block_properties.pop(property, None)
        if block_properties != block.properties:
            return Block(block.namespace, block.base_name, block_properties)
        return block

    @staticmethod
    def _clean_block(block: Block) -> Block:
        if block.base_name in BlockTokenConverter.block_names_to_change:
            block = Block(
                block.namespace,
                BlockTokenConverter.block_names_to_change[block.base_name],
                block.properties,
            )
        block = BlockTokenConverter._clean_block_properties(block)
        return block

    @staticmethod
    def _preserve_waterlogging_conversion(
        source_block: Block, target_block: Block
    ) -> Block:
        waterlogged = source_block.properties.get("waterlogged")
        if waterlogged and waterlogged.value == "true":
            return Block(
                namespace=target_block.namespace,
                base_name=target_block.base_name,
                properties={"waterlogged": waterlogged, **target_block.properties},
            )
        return target_block

    @staticmethod
    def _preserve_waterlogging_universal(block: Block) -> Block:
        if any(block.base_name == "water" for block in block.extra_blocks):
            return Block(
                namespace=block.namespace,
                base_name=block.base_name,
                properties={"waterlogged": StringTag("true"), **block.properties},
            )
        return block

    def token_to_universal_str(self, token: int) -> str:
        return self._block_token_mapper.get_block_str(token)

    def universal_str_to_token(
        self, universal_block_str: str, update_mapping: bool = False
    ) -> int:
        return self._block_token_mapper.get_token(universal_block_str, update_mapping)

    def token_to_versioned_str(self, token: int) -> str:
        universal_block_str = self.token_to_universal_str(token)
        universal_block = Block.from_string_blockstate(universal_block_str)
        versioned_block, _, _ = self._version_translator.block.from_universal(
            universal_block
        )
        BlockTokenConverter._preserve_waterlogging_conversion(
            universal_block, versioned_block
        )
        return versioned_block.blockstate

    def versioned_str_to_universal_block(self, versioned_block_str: str) -> Block:
        versioned_block = Block.from_string_blockstate(versioned_block_str)
        universal_block, _, _ = self._version_translator.block.to_universal(
            versioned_block
        )
        return universal_block

    def universal_block_to_token(
        self, universal_block: Block, update_mapping: bool = False
    ) -> int:
        universal_block = BlockTokenConverter._preserve_waterlogging_universal(
            universal_block
        )
        universal_block = BlockTokenConverter._clean_block(universal_block)
        return self.universal_str_to_token(universal_block.blockstate, update_mapping)

    def versioned_str_to_token(
        self, versioned_block_str: str, update_mapping: bool = False
    ) -> int:
        universal_block = self.versioned_str_to_universal_block(versioned_block_str)
        return self.universal_block_to_token(universal_block, update_mapping)

    def get_unused_token(self):
        return self._block_token_mapper.find_next_available_token()
