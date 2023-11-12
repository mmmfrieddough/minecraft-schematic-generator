import json

class BlockTokenMapper:
    def __init__(self):
        self.mapping_path = 'block_state_mapping.json'

        # Load mappings or initialize if not present
        try:
            with open(self.mapping_path, 'r') as f:
                self.block_id_to_token_map = json.load(f)
            # Generate the reverse mapping once at load time
            self.token_to_block_id_map = {v: k for k, v in self.block_id_to_token_map.items()}
            self.next_available_token = max(self.token_to_block_id_map.keys()) + 1
        except FileNotFoundError:
            self.block_id_to_token_map = {}
            self.token_to_block_id_map = {}
            self.next_available_token = 1

    def encode_block_id(self, block_id: str) -> int:
        # Remove the minecraft: prefix
        block_id = block_id.removeprefix('minecraft:')

        # Check if there are any properties to normalize
        if '[' not in block_id:
            return block_id

        # Parse and sort the properties to ensure consistent ordering
        properties = block_id[block_id.index('[')+1:-1].split(',')
        properties.sort()
        normalized_properties = ','.join(properties)

        # Reassemble into a canonical form
        block_type = block_id.split('[')[0]
        block_id = f'{block_type}[{normalized_properties}]'

        return block_id
    
    def decode_block_id(self, block_id: str) -> str:
        # Add the minecraft: prefix
        block_id = f'minecraft:{block_id}'
        
        return block_id

    def save_mapping(self) -> None:
        # Save the forward mapping to a file
        with open(self.mapping_path, 'w') as f:
            json.dump(self.block_id_to_token_map, f)

    def token_to_block_id(self, token: int) -> str:
        # Get the block ID from the reverse mapping
        encoded_block_id = self.token_to_block_id_map.get(token, None)

        # Decode the block ID and return it
        block_id = self.decode_block_id(encoded_block_id)
        return block_id

    def block_id_to_token(self, block_id: str) -> int:
        # Encode the block ID
        encoded_block_id = self.encode_block_id(block_id)

        # If the block ID has not been tokenized, assign a new token
        if encoded_block_id not in self.block_id_to_token_map:
            self.block_id_to_token_map[encoded_block_id] = self.next_available_token
            self.token_to_block_id_map[self.next_available_token] = encoded_block_id  # Update the reverse mapping as well
            self.next_available_token += 1
            self.save_mapping()

        # Return the token
        return self.block_id_to_token_map[encoded_block_id]
