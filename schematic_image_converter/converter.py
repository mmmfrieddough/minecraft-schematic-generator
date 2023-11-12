from litemapy import Schematic, Region, BlockState
from PIL import Image
import numpy as np
import mapping

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
SCHEMATIC_WIDTH = 64
SCHEMATIC_HEIGHT = 64
SCHEMATIC_DEPTH = 64

# Size of each tile
TILE_WIDTH = SCHEMATIC_WIDTH
TILE_HEIGHT = SCHEMATIC_DEPTH

# Number of tiles in each dimension
TILES_X = 8
TILES_Y = 8


def convert_schematic_to_image(schematic_path, image_path):
    # Load the schematic and get its first region
    schem = Schematic.load(schematic_path)
    reg = list(schem.regions.values())[0]

    # Create the image
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Draw the image
    for layer in reg.yrange():
        if layer >= SCHEMATIC_HEIGHT:
            break

        # Calculate the position of the layer in the grid
        grid_x = (layer % TILES_X) * TILE_WIDTH
        grid_y = (layer // TILES_X) * TILE_HEIGHT

        for x in reg.xrange():
            if x >= SCHEMATIC_WIDTH:
                break
            for z in reg.zrange():
                if z >= SCHEMATIC_DEPTH:
                    break
                b = reg.getblock(x, layer, z)
                color = mapping.block_state_to_rgb(b.blockid)

                # Set the pixel in the image
                img.putpixel((grid_x + x, grid_y + z), color)

    # Save the image
    img.save(image_path)


def convert_image_to_schematic(image_path, schematic_path):
    # Load the image
    img = Image.open(image_path)

    # Shortcut to create a schematic with a single region
    reg = Region(0, 0, 0, SCHEMATIC_WIDTH, SCHEMATIC_HEIGHT, SCHEMATIC_DEPTH)
    schem = reg.as_schematic(
        name="Planet", author="mmmfrieddough", description="Made with litemapy")

    # Create a reverse color mapping if you have multiple blocks
    color_to_block = {
        (0, 0, 0): "minecraft:air",
        (255, 255, 255): "minecraft:stone",
        # Add other mappings as needed
    }

    # Iterate over each layer in the grid
    for layer in range(SCHEMATIC_HEIGHT):
        # Calculate the position of the layer in the grid
        grid_x = (layer % TILES_X) * TILE_WIDTH
        grid_y = (layer // TILES_X) * TILE_HEIGHT

        for x in range(SCHEMATIC_WIDTH):
            for z in range(SCHEMATIC_DEPTH):
                # Get the pixel color from the image
                color = img.getpixel((grid_x + x, grid_y + z))
                # Convert the color to a block ID
                # Default to air if color not found
                block_id = color_to_block.get(color, "minecraft:air")

                # Set the block in the schematic
                reg.setblock(x, layer, z, BlockState(block_id))

    # Save the schematic data structure to a file
    schem.save(schematic_path)


convert_schematic_to_image("lighthouse.litematic", "lighthouse.png")
# convert_schematic_to_image("planet.litematic", "planet.png")
# convert_schematic_to_image("sphere.litematic", "sphere.png")
# convert_schematic_to_image("cube.litematic", "cube.png")
