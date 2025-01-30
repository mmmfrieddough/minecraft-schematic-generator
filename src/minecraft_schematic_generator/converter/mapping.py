import json
from importlib import resources

import portalocker


class BlockTokenMapper:
    def find_next_available_token(self) -> int:
        while self.next_available_token in self.token_to_block_id_map:
            self.next_available_token += 1
        return self.next_available_token

    def __init__(self):
        # Get the data directory path
        data_path = resources.files("minecraft_schematic_generator.converter")
        self.mapping_path = data_path.joinpath("block_state_mapping.json")

        # Initialize the mapping
        self.next_available_token = 1
        self.block_id_to_token_map = {}
        self.token_to_block_id_map = {}
        self.block_str_to_token("universal_minecraft:air", update_mapping=True)

    def token_to_block_str(self, token: int) -> str:
        if token not in self.token_to_block_id_map:
            raise KeyError(f"Token {token} not found in mapping")

        # Get the block ID from the reverse mapping
        return self.token_to_block_id_map[token]

    def block_str_to_token(self, block_str: str, update_mapping: bool = False) -> int:
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
