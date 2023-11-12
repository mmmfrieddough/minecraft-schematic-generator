import json

# Paths for the mapping file
mapping_path = 'block_state_mapping.json'

# Load mappings or initialize if not present
try:
    with open(mapping_path, 'r') as f:
        block_state_to_token = json.load(f)
    # Generate the reverse mapping once at load time
    token_to_block_state = {v: k for k, v in block_state_to_token.items()}
    next_available_token = max(token_to_block_state.keys()) + 1
except FileNotFoundError:
    block_state_to_token = {}
    token_to_block_state = {}
    next_available_token = 1


def normalize_block_state(state_string):
    # Remove the 'minecraft:' prefix before encoding
    state_string = state_string.replace('minecraft:', '')

    if '[' not in state_string:
        # No properties to normalize
        return state_string

    # Parse and sort the properties to ensure consistent ordering
    properties = state_string[state_string.index('[')+1:-1].split(',')
    properties.sort()
    normalized_properties = ','.join(properties)

    # Reassemble into a canonical form
    block_type = state_string.split('[')[0]
    normalized_state = f'{block_type}[{normalized_properties}]'

    return normalized_state


def save_mapping():
    # Save the forward mapping to a file
    with open(mapping_path, 'w') as f:
        json.dump(block_state_to_token, f)


def encode_block_state(block_state):
    global next_available_token

    # Normalize the block state
    block_state = normalize_block_state(block_state)

    # If the block state has not been tokenized, assign a new token
    if block_state not in block_state_to_token:
        block_state_to_token[block_state] = next_available_token
        # Update the reverse mapping as well
        token_to_block_state[next_available_token] = block_state
        next_available_token += 1
        save_mapping()

    # Return the token
    return block_state_to_token[block_state]


def decode_token_to_block_state(token):
    # Return the block state from the token
    compacted_block_state = token_to_block_state.get(token, None)
    if compacted_block_state is not None:
        return 'minecraft:' + compacted_block_state
    return None


def token_to_rgb(token):
    r = (token >> 16) & 0xFF
    g = (token >> 8) & 0xFF
    b = token & 0xFF
    return (r, g, b)


def rgb_to_token(rgb):
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def block_state_to_rgb(block_state):
    token = encode_block_state(block_state)
    return token_to_rgb(token)


def rgb_to_block_state(rgb):
    token = rgb_to_token(rgb)
    return decode_token_to_block_state(token)


# Example usage
block_state = 'minecraft:smooth_quartz_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]'
color = block_state_to_rgb(block_state)
decoded_block_state = rgb_to_block_state(color)

print(f"Original Block State: {block_state}")
print(f"Color Representation: {color}")
print(f"Decoded Block State: {decoded_block_state}")
