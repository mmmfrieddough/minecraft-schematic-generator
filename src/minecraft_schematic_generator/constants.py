from typing import Tuple

# Minecraft Version
MINECRAFT_VERSION: Tuple[int, int, int] = (1, 21, 4)
MINECRAFT_PLATFORM: str = "java"

# Model Constants
MAX_STRUCTURE_SIZE: int = 15

# Block Constants
MASK_BLOCK_ID: int = 0
AIR_BLOCK_ID: int = 1

# Block Strings
AIR_BLOCK_STR: str = "universal_minecraft:air"
REPLACEABLE_BLOCK_STRINGS: list[str] = [
    "universal_minecraft:plant[plant_type=grass]",
    "universal_minecraft:double_plant[half=lower,plant_type=tall_grass]",
    "universal_minecraft:double_plant[half=upper,plant_type=tall_grass]",
    "universal_minecraft:plant[plant_type=fern]",
    "universal_minecraft:double_plant[half=lower,plant_type=large_fern]",
    "universal_minecraft:double_plant[half=upper,plant_type=large_fern]",
    "universal_minecraft:plant[plant_type=dead_bush]",
]
