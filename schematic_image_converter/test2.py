fc_params = 768 * 512  # Fully connected layer params
conv1_params = 32 * 8 * 3**3  # First conv layer params
conv2_params = 64 * 32 * 3**3  # Second conv layer params
conv3_params = 128 * 64 * 3**3  # Third conv layer params
output_params = 256 * 128 * 3**3  # Output layer params

total_params = fc_params + conv1_params + \
    conv2_params + conv3_params + output_params
print(f'Total params: {total_params}')
