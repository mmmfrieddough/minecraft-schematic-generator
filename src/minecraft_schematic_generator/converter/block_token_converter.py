from amulet import Block

from .block_normalizer import BlockNormalizer
from .block_token_mapper import BlockTokenMapperInterface


class BlockTokenConverter:
    """Handles universal block ↔ token conversions. No version dependency."""

    def __init__(self, block_token_mapper: BlockTokenMapperInterface):
        self._block_token_mapper = block_token_mapper

    def token_to_universal_str(self, token: int) -> str:
        """Convert a token to a universal block string."""
        return self._block_token_mapper.get_block_str(token)

    def universal_str_to_token(
        self, universal_block_str: str, update_mapping: bool = False
    ) -> int:
        """Convert a universal block string to a token."""
        return self._block_token_mapper.get_token(universal_block_str, update_mapping)

    def token_to_universal_block(self, token: int) -> Block:
        """Convert a token to a universal Block object."""
        universal_block_str = self.token_to_universal_str(token)
        return Block.from_string_blockstate(universal_block_str)

    def universal_block_to_token(
        self, universal_block: Block, update_mapping: bool = False
    ) -> int:
        """Convert a universal Block object to a token."""
        universal_block = BlockNormalizer.preserve_waterlogging_universal(
            universal_block
        )
        universal_block = BlockNormalizer.normalize(universal_block)
        return self.universal_str_to_token(universal_block.blockstate, update_mapping)

    def get_unused_token(self) -> int:
        """Get the next available unused token."""
        return self._block_token_mapper.find_next_available_token()
