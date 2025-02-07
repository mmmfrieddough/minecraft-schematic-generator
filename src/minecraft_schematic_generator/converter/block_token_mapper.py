import json
from abc import ABC, abstractmethod
from importlib import resources
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Lock as LockBase

import portalocker

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


class FileBlockTokenMapper(BlockTokenMapperInterface):
    """File-based block token mapper"""

    def __init__(self):
        # Get the data directory path
        data_path = resources.files("minecraft_schematic_generator.converter")
        self.mapping_path = data_path.joinpath("block_state_mapping.json")

        # Initialize the mapping
        self.next_available_token = AIR_BLOCK_ID
        self.block_id_to_token_map = {}
        self.token_to_block_id_map = {}
        self._ensure_air_block()

    def find_next_available_token(self) -> int:
        while self.next_available_token in self.token_to_block_id_map:
            self.next_available_token += 1
        return self.next_available_token

    def get_block_str(self, token: int) -> str:
        if token not in self.token_to_block_id_map:
            raise KeyError(f"Token {token} not found in mapping")

        # Get the block ID from the reverse mapping
        return self.token_to_block_id_map[token]

    def get_token(self, block_str: str, update_mapping: bool = False) -> int:
        # If the block ID is already in the mapping, return the token
        if block_str in self.block_id_to_token_map:
            return self.block_id_to_token_map[block_str]

        # Throw an error if updates are not allowed
        if not update_mapping:
            raise KeyError(f"Block {block_str} not found in mapping")

        try:
            # Acquire an exclusive lock on the mapping file before reading/updating
            with portalocker.Lock(self.mapping_path, "a+", timeout=60) as fh:
                # Read the file again in case it was updated by another process
                fh.seek(0)
                file_contents = fh.read().strip()

                # Check if the file is empty
                if file_contents:
                    # Load the mapping from the file
                    self.block_id_to_token_map = json.loads(file_contents)

                # Generate the reverse mapping
                self.token_to_block_id_map = {
                    v: k for k, v in self.block_id_to_token_map.items()
                }

                # Check if the block ID is already in the mapping
                if block_str not in self.block_id_to_token_map:
                    # Generate a new token for the block ID
                    token = self.find_next_available_token()

                    # Update the mapping
                    self.block_id_to_token_map[block_str] = token
                    self.token_to_block_id_map[token] = block_str

                    # Save the updated mapping to the file
                    fh.seek(0)
                    fh.truncate()
                    json.dump(self.block_id_to_token_map, fh)

        except portalocker.exceptions.LockException:
            raise TimeoutError("Unable to acquire lock for mapping file after")

        # Return the token
        return self.block_id_to_token_map[block_str]
