import json

from schempy import Block


class BlockTokenMapper:
    def __init__(self):
        self.mapping_path = 'block_state_mapping.json'

        # Load mappings or initialize if not present
        try:
            with open(self.mapping_path, 'r') as f:
                self.block_id_to_token_map = json.load(f)
            # Generate the reverse mapping once at load time
            self.token_to_block_id_map = {
                v: k for k, v in self.block_id_to_token_map.items()}
            self.next_available_token = max(
                self.token_to_block_id_map.keys()) + 1
        except FileNotFoundError:
            self.block_id_to_token_map = {}
            self.token_to_block_id_map = {}
            self.next_available_token = 1
            self.block_to_token(Block('minecraft:air'))

    def id_to_block(self, id: str) -> Block:
        # Convert the properties to a dict
        property_dict = {}
        if id.find('[') == -1:
            block_id = id
        else:
            entries = id.split('[')
            block_id = entries[0]
            properties = entries[1].replace(']', '').split(',')
            for property in properties:
                key, value = property.split('=')
                property_dict[key] = value

        return Block(block_id, properties=property_dict)

    def save_mapping(self) -> None:
        # Save the forward mapping to a file
        with open(self.mapping_path, 'w') as f:
            json.dump(self.block_id_to_token_map, f)

    def token_to_block(self, token: int) -> Block:
        # Get the block ID from the reverse mapping
        id = self.token_to_block_id_map[token]

        # Decode the block ID and return it
        block_id = self.id_to_block(id)
        return block_id

    def block_to_token(self, block: Block) -> int:
        # Encode the block ID
        id = str(block)

        # If the block ID has not been tokenized, assign a new token
        if id not in self.block_id_to_token_map:
            self.block_id_to_token_map[id] = self.next_available_token
            # Update the reverse mapping as well
            self.token_to_block_id_map[self.next_available_token] = id
            self.next_available_token += 1
            self.save_mapping()

        # Return the token
        return self.block_id_to_token_map[id]
