import logging

from .block_normalizer import BlockNormalizer
from .block_token_converter import BlockTokenConverter

# Set PyMCTranslate logging level before importing it
logging.getLogger("PyMCTranslate").setLevel(logging.WARNING)
import PyMCTranslate  # noqa: E402
from amulet import Block  # noqa: E402


class VersionedBlockTokenConverter:
    """Adds versioned block operations on top of BlockTokenConverter."""

    def __init__(
        self,
        block_token_converter: BlockTokenConverter,
        version_translator: PyMCTranslate.Version | None = None,
    ):
        self._converter = block_token_converter
        if version_translator is None:
            # Default to latest version if none provided
            translation_manager = PyMCTranslate.new_translation_manager()
            latest_version = translation_manager.version_numbers("java")[-1]
            version_translator = translation_manager.get_version("java", latest_version)
        self._version_translator = version_translator

    # Delegate universal methods to the base converter
    def token_to_universal_str(self, token: int) -> str:
        """Convert a token to a universal block string."""
        return self._converter.token_to_universal_str(token)

    def universal_str_to_token(
        self, universal_block_str: str, update_mapping: bool = False
    ) -> int:
        """Convert a universal block string to a token."""
        return self._converter.universal_str_to_token(
            universal_block_str, update_mapping
        )

    def token_to_universal_block(self, token: int) -> Block:
        """Convert a token to a universal Block object."""
        return self._converter.token_to_universal_block(token)

    def universal_block_to_token(
        self, universal_block: Block, update_mapping: bool = False
    ) -> int:
        """Convert a universal Block object to a token."""
        return self._converter.universal_block_to_token(universal_block, update_mapping)

    def get_unused_token(self) -> int:
        """Get the next available unused token."""
        return self._converter.get_unused_token()

    # Versioned methods that require version_translator
    def token_to_versioned_str(self, token: int) -> str:
        """Convert a token to a versioned block string."""
        universal_block = self.token_to_universal_block(token)
        versioned_block, _, _ = self._version_translator.block.from_universal(
            universal_block
        )
        versioned_block = BlockNormalizer.preserve_waterlogging_conversion(
            universal_block, versioned_block
        )
        return versioned_block.blockstate

    def versioned_str_to_universal_block(self, versioned_block_str: str) -> Block:
        """Convert a versioned block string to a universal Block object."""
        versioned_block = Block.from_string_blockstate(versioned_block_str)
        universal_block, _, _ = self._version_translator.block.to_universal(
            versioned_block
        )
        return universal_block

    def versioned_str_to_token(
        self, versioned_block_str: str, update_mapping: bool = False
    ) -> int:
        """Convert a versioned block string to a token."""
        universal_block = self.versioned_str_to_universal_block(versioned_block_str)
        return self.universal_block_to_token(universal_block, update_mapping)
