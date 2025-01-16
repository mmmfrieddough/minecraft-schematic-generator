import json
from importlib import resources

from schempy import Block


class BlockTokenMapper:
    def find_next_available_token(self) -> int:
        while self.next_available_token in self.token_to_block_id_map:
            self.next_available_token += 1
        return self.next_available_token

    def __init__(self):
        # Get the data directory path
        data_path = resources.files("minecraft_schematic_generator.converter")
        self.mapping_path = data_path.joinpath("block_state_mapping.json")

        self.next_available_token = 1

        # Load mappings or initialize if not present
        try:
            with self.mapping_path.open("r") as f:
                self.block_id_to_token_map = json.load(f)
            # Generate the reverse mapping once at load time
            self.token_to_block_id_map = {
                v: k for k, v in self.block_id_to_token_map.items()
            }
        except FileNotFoundError:
            self.block_id_to_token_map = {}
            self.token_to_block_id_map = {}
            self.block_to_token(Block("minecraft:air"), update_mapping=True)

    def id_to_block(self, id: str) -> Block:
        # Convert the properties to a dict
        property_dict = {}
        if id.find("[") == -1:
            block_id = id
        else:
            entries = id.split("[")
            block_id = entries[0]
            properties = entries[1].replace("]", "").split(",")
            for property in properties:
                key, value = property.split("=")
                property_dict[key] = value

        return Block(block_id, properties=property_dict)

    def save_mapping(self) -> None:
        # Save the forward mapping to a file
        with self.mapping_path.open("w") as f:
            json.dump(self.block_id_to_token_map, f)

    def token_to_block(self, token: int) -> Block:
        # Get the block ID from the reverse mapping
        id = self.token_to_block_id_map[token]

        # Decode the block ID and return it
        block_id = self.id_to_block(id)
        return block_id

    def block_str_to_token(self, block_str: str, update_mapping: bool = False) -> int:
        # If the block ID has not been tokenized, assign a new token
        if block_str not in self.block_id_to_token_map:
            if not update_mapping:
                raise KeyError(f"Block {block_str} not found in mapping")

            token = self.find_next_available_token()
            self.block_id_to_token_map[block_str] = token
            self.token_to_block_id_map[token] = block_str
            self.save_mapping()

        # Return the token
        return self.block_id_to_token_map[block_str]

    def block_to_token(self, block: Block, update_mapping: bool = False) -> int:
        # Encode the block ID
        block_str = str(block)

        return self.block_str_to_token(block_str, update_mapping)
