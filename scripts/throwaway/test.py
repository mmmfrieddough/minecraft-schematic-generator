import torch


def precompute_linfinity_order(
    N: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns three 1D PyTorch LongTensors of length N^3 containing the x, y, z coordinates
    ordered from the center outward by L∞ distance.
    Within each shell, (x, y, z) are sorted lexicographically.

    L∞ distance from center c = N//2 is d = max(|x-c|, |y-c|, |z-c|).
    """
    c = N // 2
    items = []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                d = max(abs(x - c), abs(y - c), abs(z - c))
                items.append((d, x, y, z))

    # Sort by (distance, x, y, z) to get a stable ordering within each shell
    items.sort()

    # Separate x, y, z coordinates into their own lists
    x_indices = [x for (_, x, _, _) in items]
    y_indices = [y for (_, _, y, _) in items]
    z_indices = [z for (_, _, _, z) in items]

    return (
        torch.tensor(x_indices, dtype=torch.long),
        torch.tensor(y_indices, dtype=torch.long),
        torch.tensor(z_indices, dtype=torch.long),
    )


# Precompute once for all N up to 15
BFS_ORDERS = {}
for n in range(1, 16):
    BFS_ORDERS[n] = precompute_linfinity_order(n)


def extract_minimal_cube(
    array: torch.Tensor,
    center_x: int,
    center_y: int,
    center_z: int,
    default_value: float = 0.0,
) -> torch.Tensor:
    """
    Extract the smallest possible cube from a 3D tensor that includes all values from the original array,
    centered on the given position. The cube size is determined by the maximum distance needed in any
    dimension to reach the array bounds.

    Args:
        array: Input 3D tensor of shape (X, Y, Z)
        center_x, center_y, center_z: Center position of the cube
        default_value: Value to use for positions outside the original array

    Returns:
        torch.Tensor: Cube with odd-sized dimensions, centered on the given position
    """
    # Calculate the required radius in each dimension
    radius_x = max(center_x, array.shape[0] - 1 - center_x)
    radius_y = max(center_y, array.shape[1] - 1 - center_y)
    radius_z = max(center_z, array.shape[2] - 1 - center_z)

    # Use the maximum radius to ensure a perfect cube
    radius = max(radius_x, radius_y, radius_z)
    cube_size = 2 * radius + 1

    # Create the output cube filled with default value
    result = torch.full(
        (cube_size, cube_size, cube_size),
        default_value,
        dtype=array.dtype,
        device=array.device,
    )

    # Calculate the bounds for the original array
    x_min = max(0, center_x - radius)
    x_max = min(array.shape[0], center_x + radius + 1)
    y_min = max(0, center_y - radius)
    y_max = min(array.shape[1], center_y + radius + 1)
    z_min = max(0, center_z - radius)
    z_max = min(array.shape[2], center_z + radius + 1)

    # Calculate the corresponding positions in the result cube
    cube_x_start = radius - (center_x - x_min)
    cube_y_start = radius - (center_y - y_min)
    cube_z_start = radius - (center_z - z_min)

    # Copy the valid portion of the input array into the result cube
    result[
        cube_x_start : cube_x_start + (x_max - x_min),
        cube_y_start : cube_y_start + (y_max - y_min),
        cube_z_start : cube_z_start + (z_max - z_min),
    ] = array[x_min:x_max, y_min:y_max, z_min:z_max]

    return result


def create_3d_grid(Nx: int, Ny: int, Nz: int) -> torch.Tensor:
    """
    Creates a 3D grid of shape (Nx, Ny, Nz) where each element is a unique value
    while maintaining the same L∞ distance relationship to the center.
    Values are constructed relative to the center position so that
    equivalent positions in different size grids get the same value.
    """
    grid = torch.zeros((Nx, Ny, Nz), dtype=torch.long)
    cx = Nx // 2  # center point x
    cy = Ny // 2  # center point y
    cz = Nz // 2  # center point z

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # L∞ distance from center
                d = max(abs(x - cx), abs(y - cy), abs(z - cz))
                # Calculate relative positions from center (-d to +d)
                rel_x = x - cx
                rel_y = y - cy
                rel_z = z - cz
                # Create a unique value using relative positions
                # Add d+7 to relative coordinates to make them positive (since they range from -7 to +7)
                grid[x, y, z] = (
                    d * 1000000 + (rel_x + 7) * 10000 + (rel_y + 7) * 100 + (rel_z + 7)
                )
    return grid


# Example usage for N=3
if __name__ == "__main__":
    # Test prefix property
    def test_prefix_property():
        for n in range(1, 15, 2):
            small_grid = create_3d_grid(n, n, n)

            # Generate random odd dimensions between n and n+5
            large_nx = (
                n + 2 * torch.randint(0, 3, (1,)).item()
            )  # Random odd number between n and n+4
            large_ny = n + 2 * torch.randint(0, 3, (1,)).item()
            large_nz = n + 2 * torch.randint(0, 3, (1,)).item()
            large_grid = create_3d_grid(large_nx, large_ny, large_nz)

            # print(small_grid)
            # print(large_grid)

            # Get flattened versions in L∞ order
            small_flat = extract_minimal_cube(small_grid, n // 2, n // 2, n // 2)[
                BFS_ORDERS[n]
            ]
            large_flat = extract_minimal_cube(
                large_grid, large_nx // 2, large_ny // 2, large_nz // 2
            )[BFS_ORDERS[max(large_nx, large_ny, large_nz)]]

            # print(small_flat)
            # print(large_flat)

            # Check if smaller grid is a prefix of larger grid
            small_len = len(small_flat)
            is_prefix = torch.allclose(small_flat, large_flat[:small_len])
            print(
                f"N={n}: smaller grid ({n}x{n}x{n}) is{' ' if is_prefix else ' not '}a prefix of ({large_nx}x{large_ny}x{large_nz})"
            )
            if not is_prefix:
                print(
                    f"First mismatch at index {torch.where(small_flat != large_flat[:small_len])[0][0]}"
                )
                return False
        return True

    print("\nTesting prefix property:")
    all_prefixes_valid = test_prefix_property()
    print(f"All prefixes valid: {all_prefixes_valid}")
