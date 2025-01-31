import amulet
import PyMCTranslate

from minecraft_schematic_generator.constants import (
    MINECRAFT_PLATFORM,
    MINECRAFT_VERSION,
)

from .file_handler import BlockTokenFileHandler


class BlockTokenMapper:
    def __init__(
        self,
        file_handler: BlockTokenFileHandler | None = None,
        version_translator: PyMCTranslate.Version | None = None,
    ):
        self.file_handler = file_handler or BlockTokenFileHandler()
        self.version_translator = (
            version_translator
            # Default to the version we are using for the project
            or PyMCTranslate.new_translation_manager().get_version(
                MINECRAFT_PLATFORM, MINECRAFT_VERSION
            )
        )

    def _versioned_to_universal(self, versioned_block_str: str) -> str:
        versioned_block = amulet.Block.from_string_blockstate(versioned_block_str)
        universal_block, _, _ = self.version_translator.block.to_universal(
            versioned_block
        )
        return universal_block.blockstate

    def _universal_to_versioned(self, universal_block_str: str) -> str:
        universal_block = amulet.Block.from_string_blockstate(universal_block_str)
        versioned_block, _, _ = self.version_translator.block.from_universal(
            universal_block
        )
        return versioned_block.blockstate

    def token_to_universal(self, token: int) -> str:
        try:
            return self.file_handler.get_block_str(token)
        except KeyError:
            return "universal_minecraft:air"

    def universal_to_token(
        self, universal_block_str: str, update_mapping: bool = False
    ) -> int:
        return self.file_handler.get_token(universal_block_str, update_mapping)

    def token_to_versioned(self, token: int) -> str:
        return self._universal_to_versioned(self.token_to_universal(token))

    def versioned_to_token(
        self, versioned_block_str: str, update_mapping: bool = False
    ) -> int:
        return self.universal_to_token(
            self._versioned_to_universal(versioned_block_str), update_mapping
        )

    def get_unused_token(self):
        return self.file_handler.find_next_available_token()
