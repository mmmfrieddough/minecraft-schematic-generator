import nbtlib
from litemapy import Region

from converter.converter import RegionTensorConverter

# file_name = '0a0b45a9b3494ec02d9d73017510edb9d59e62aee88d2047c2d824aac8a28d66.schem'
file_name = 'KbvChvZGXS-Hardcore House.schem'
converted_file_name = f'converted_{file_name}'

# Load a region from a file
nbt = nbtlib.load(file_name)
region, _ = Region.from_sponge_nbt(nbt)

# Debug: Print the region details
print(f"Region dimensions: {region.width}x{region.height}x{region.length}")

# Find each unique block ID in the region
unique_block_ids = set()
for x, y, z in region.allblockpos():
    blockstate = region.getblock(x, y, z)
    unique_block_ids.add(blockstate.to_block_state_identifier())
print(f"Unique block IDs: {unique_block_ids}")

# Create a converter
converter = RegionTensorConverter()

# Convert the region to a tensor
tensor = converter.region_to_tensor(region)

# Debug: Print the tensor details
print(f"Tensor dimensions: {tensor.shape}")
print(f"Tensor dtype: {tensor.dtype}")

# Print the unique tokens in the tensor
unique_tokens = set()
for z in range(tensor.shape[0]):
    for y in range(tensor.shape[1]):
        for x in range(tensor.shape[2]):
            unique_tokens.add(tensor[z, y, x].item())
print(f"Unique tokens: {unique_tokens}")

# Convert the tensor back to a region
converted_region = converter.tensor_to_region(tensor)

# Debug: Print the region details
print(
    f"Region dimensions: {converted_region.width}x{converted_region.height}x{converted_region.length}")

# Find each unique block ID in the region
unique_block_ids = set()
for x, y, z in converted_region.allblockpos():
    blockstate = converted_region.getblock(x, y, z)
    unique_block_ids.add(blockstate.to_block_state_identifier())
print(f"Unique block IDs: {unique_block_ids}")

# Save the region
file: nbtlib.File = converted_region.to_sponge_nbt()
file.save(converted_file_name)
print("Region saved successfully.")
