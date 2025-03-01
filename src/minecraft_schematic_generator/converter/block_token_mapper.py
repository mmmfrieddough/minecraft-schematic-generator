from abc import ABC, abstractmethod
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Lock as LockBase

from minecraft_schematic_generator.constants import AIR_BLOCK_ID, AIR_BLOCK_STR


class BlockTokenMapperInterface(ABC):
    """Interface for block token mappers"""

    @abstractmethod
    def find_next_available_token(self) -> int:
        """Finds the next available token"""
        pass

    @abstractmethod
    def get_block_str(self, token: int) -> str:
        """Gets the block string for a token"""
        pass

    @abstractmethod
    def get_token(self, block_str: str, update_mapping: bool = False) -> int:
        """Gets the token for a block string"""
        pass

    def _ensure_air_block(self):
        """Ensures the air block is mapped to token 1"""
        self.get_token(AIR_BLOCK_STR, update_mapping=True)


class DictBlockTokenMapper(BlockTokenMapperInterface):
    """Block token mapper using a dictionary"""

    def __init__(self, block_id_to_token_map: dict):
        self._next_available_token = AIR_BLOCK_ID
        self._block_id_to_token_map = block_id_to_token_map
        self._update_reverse_mapping()

    def _update_reverse_mapping(self):
        self._token_to_block_id_map = {
            v: k for k, v in self._block_id_to_token_map.items()
        }

    def find_next_available_token(self) -> int:
        """Finds the next available token, filling in any gaps in the sequence."""
        existing_tokens = set(self._block_id_to_token_map.values())
        while self._next_available_token in existing_tokens:
            self._next_available_token += 1
        return self._next_available_token

    def get_block_str(self, token: int) -> str:
        if token not in self._token_to_block_id_map:
            raise KeyError(f"Token {token} not found in mapping")
        return self._token_to_block_id_map[token]

    def get_token(self, block_str: str, update_mapping: bool = False) -> int:
        if block_str not in self._block_id_to_token_map:
            if not update_mapping:
                raise KeyError(f"Block {block_str} not found in mapping")
            token = self.find_next_available_token()
            self._block_id_to_token_map[block_str] = token
            self._update_reverse_mapping()
        return self._block_id_to_token_map[block_str]


class SharedDictBlockTokenMapper(BlockTokenMapperInterface):
    """Block token mapper using a shared dictionary"""

    def __init__(self, shared_dict: DictProxy, lock: LockBase):
        self._shared_dict = shared_dict
        self._lock = lock
        self._next_available_token = AIR_BLOCK_ID
        self._block_id_to_token_map = {}
        self._token_to_block_id_map = {}
        self._ensure_air_block()

    def find_next_available_token(self) -> int:
        """Finds the next available token, filling in any gaps in the sequence."""
        with self._lock:
            # Get all existing tokens as integers
            existing_tokens = set(self._shared_dict.values())

            # Find the first gap in the sequence
            while self._next_available_token in existing_tokens:
                self._next_available_token += 1
            return self._next_available_token

    def get_block_str(self, token: int) -> str:
        # Check if token is in the reverse mapping
        if token in self._token_to_block_id_map:
            return self._token_to_block_id_map[token]

        # Update the mapping
        with self._lock:
            self._block_id_to_token_map = self._shared_dict.copy()

        # Update the reverse mapping
        self._token_to_block_id_map = {
            v: k for k, v in self._block_id_to_token_map.items()
        }

        # Check again after updating
        if token not in self._token_to_block_id_map:
            raise KeyError(f"Token {token} not found in mapping")

        return self._token_to_block_id_map[token]

    def get_token(self, block_str: str, update_mapping: bool = False) -> int:
        # Check if block is in the mapping
        if block_str in self._block_id_to_token_map:
            return self._block_id_to_token_map[block_str]

        with self._lock:
            if block_str not in self._shared_dict:
                if not update_mapping:
                    raise KeyError(f"Block {block_str} not found in mapping")
                token = self.find_next_available_token()
                self._shared_dict[block_str] = token
            self._block_id_to_token_map = self._shared_dict.copy()

        # Update the reverse mapping
        self._token_to_block_id_map = {
            v: k for k, v in self._block_id_to_token_map.items()
        }

        return self._block_id_to_token_map[block_str]
