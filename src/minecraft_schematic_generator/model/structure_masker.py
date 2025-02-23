import random
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from amulet import Block
from amulet_nbt import StringTag

from minecraft_schematic_generator.converter import BlockTokenConverter
from minecraft_schematic_generator.model.transformer_model import (
    TransformerMinecraftStructureGenerator,
)


class StructureMasker:
    # Represent "natural" blocks that the player is not likely to want to build with
    # These should be in universal format as we are feeding them directly into the mapping
    NATURAL_BLOCK_STRINGS = [
        "universal_minecraft:dirt",
        "universal_minecraft:stone",
        "universal_minecraft:grass_block[snowy=false]",
        "universal_minecraft:grass_block[snowy=true]",
        "universal_minecraft:water[falling=false,flowing=false,level=0]",
        "universal_minecraft:netherrack",
        "universal_minecraft:bedrock[infiniburn=false]",
        "universal_minecraft:basalt[axis=y]",
        "universal_minecraft:blackstone",
        "universal_minecraft:gravel",
        "universal_minecraft:deepslate[axis=y]",
        "universal_minecraft:tuff",
        "universal_minecraft:sand",
        "universal_minecraft:end_stone",
    ]

    BLOCK_FAMILIES = [
        {
            "base_name": "wood",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "crimson",
                    "warped",
                    "cherry",
                    # "pale_oak",
                ],
                "stripped": ["false", "true"],
            },
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "log",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "crimson",
                    "warped",
                    "cherry",
                    # "pale_oak",
                ],
                "stripped": ["false", "true"],
            },
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "stairs",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "brick",
                    "cobblestone",
                    "stone_brick",
                    "sandstone",
                    "red_sandstone",
                    "nether_brick",
                    "quartz",
                    "purpur",
                    # "prismarine",
                    "prismarine_brick",
                    "dark_prismarine",
                    "andesite",
                    "diorite",
                    # "end_stone_brick",
                    "granite",
                    "mossy_cobblestone",
                    "mossy_stone_brick",
                    "polished_andesite",
                    "polished_diorite",
                    "polished_granite",
                    "red_nether_brick",
                    "smooth_quartz",
                    "smooth_red_sandstone",
                    "smooth_sandstone",
                    "stone",
                    "crimson",
                    "warped",
                    "blackstone",
                    "polished_blackstone_brick",
                    "polished_blackstone",
                    "oxidized_cut_copper",
                    # "weathered_cut_copper",
                    # "exposed_cut_copper",
                    # "cut_copper",
                    "waxed_oxidized_cut_copper",
                    # "waxed_weathered_cut_copper",
                    # "waxed_exposed_cut_copper",
                    # "waxed_cut_copper",
                    "cobbled_deepslate",
                    "polished_deepslate",
                    "deepslate_tile",
                    "deepslate_brick",
                    "mud_brick",
                    "bamboo",
                    # "bamboo_mosaic",
                    "cherry",
                    # "tuff",
                    # "tuff_brick",
                    "polished_tuff",
                    # "pale_oak",
                    # "resin_brick",
                ]
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "half": ["bottom", "top"],
                "shape": [
                    "straight",
                    "inner_left",
                    "inner_right",
                    "outer_left",
                    "outer_right",
                ],
            },
        },
        {
            "base_name": "slab",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "petrified_oak",
                    "brick",
                    "stone",
                    "cobblestone",
                    "stone_brick",
                    "sandstone",
                    "red_sandstone",
                    "nether_brick",
                    "quartz",
                    "purpur",
                    "prismarine",
                    "prismarine_brick",
                    "dark_prismarine",
                    "andesite",
                    "cut_red_sandstone",
                    "cut_sandstone",
                    "diorite",
                    "end_stone_brick",
                    "granite",
                    "mossy_cobblestone",
                    "mossy_stone_brick",
                    "polished_andesite",
                    "polished_diorite",
                    "polished_granite",
                    "red_nether_brick",
                    "smooth_quartz",
                    "smooth_red_sandstone",
                    "smooth_sandstone",
                    "smooth_stone",
                    "crimson",
                    "warped",
                    "blackstone",
                    "polished_blackstone_brick",
                    "polished_blackstone",
                    "oxidized_cut_copper",
                    "weathered_cut_copper",
                    "exposed_cut_copper",
                    "cut_copper",
                    "waxed_oxidized_cut_copper",
                    "waxed_weathered_cut_copper",
                    "waxed_exposed_cut_copper",
                    "waxed_cut_copper",
                    "cobbled_deepslate",
                    "polished_deepslate",
                    "deepslate_tile",
                    "deepslate_brick",
                    "mud_brick",
                    "bamboo",
                    "bamboo_mosaic",
                    "cherry",
                    "tuff",
                    "tuff_brick",
                    "polished_tuff",
                    # "pale_oak",
                    # "resin_brick",
                ]
            },
            "varying_params": {"type": ["bottom", "top", "double"]},
        },
        {
            "base_name": "fence",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "nether_brick",
                    "crimson",
                    "warped",
                    "bamboo",
                    "cherry",
                    # "pale_oak",
                ]
            },
            "varying_params": {
                "north": ["true", "false"],
                "east": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
            },
        },
        {
            "base_name": "fence_gate",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    # "mangrove",
                    # "crimson",
                    "warped",
                    # "bamboo",
                    # "cherry",
                    # "pale_oak",
                ]
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "in_wall": ["true", "false"],
                "open": ["true", "false"],
            },
        },
        {
            "base_name": "door",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "iron",
                    "crimson",
                    "warped",
                    "bamboo",
                    "cherry",
                    # "copper",
                    # "exposed_copper",
                    # "oxidized_copper",
                    # "waxed_copper",
                    # "waxed_exposed_copper",
                    # "waxed_oxidized_copper",
                    # "waxed_weathered_copper",
                    # "weathered_copper",
                    "pale_oak",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "half": ["lower", "upper"],
                "hinge": ["left", "right"],
                "open": ["false", "true"],
            },
        },
        {
            "base_name": "trapdoor",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "iron",
                    "crimson",
                    "warped",
                    "bamboo",
                    "cherry",
                    # "copper",
                    # "exposed_copper",
                    # "oxidized_copper",
                    # "waxed_copper",
                    # "waxed_exposed_copper",
                    # "waxed_oxidized_copper",
                    # "waxed_weathered_copper",
                    # "weathered_copper",
                    # "pale_oak",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "open": ["false", "true"],
                "half": ["bottom", "top"],
            },
        },
        {
            "base_name": "button",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    # "acacia",
                    "dark_oak",
                    # "mangrove",
                    "stone",
                    "crimson",
                    # "warped",
                    "polished_blackstone",
                    # "bamboo",
                    # "cherry",
                    # "pale_oak",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "face": ["floor", "wall", "ceiling"],
            },
        },
        {
            "base_name": "wall",
            "separate_params": {
                "material": [
                    # "cobblestone",
                    # "mossy_cobblestone",
                    # "andesite",
                    # "brick",
                    # "diorite",
                    # "end_stone_brick",
                    # "granite",
                    # "mossy_stone_brick",
                    # "nether_brick",
                    # "prismarine",
                    # "red_nether_brick",
                    # "red_sandstone",
                    # "sandstone",
                    # "stone_brick",
                    # "blackstone",
                    # "polished_blackstone_brick",
                    # "polished_blackstone",
                    # "cobbled_deepslate",
                    # "polished_deepslate",
                    # "deepslate_tile",
                    # "deepslate_brick",
                    # "mud_brick",
                    # "tuff",
                    # "tuff_brick",
                    # "polished_tuff",
                    # "resin_brick",
                ],
            },
            "varying_params": {
                "east": ["none", "low", "tall"],
                "north": ["none", "low", "tall"],
                "south": ["none", "low", "tall"],
                "west": ["none", "low", "tall"],
                "up": ["true", "false"],
            },
        },
        {
            "base_name": "deepslate",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "basalt",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "polished_basalt",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "purpur_pillar",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "iron_bars",
            "varying_params": {
                "east": ["true", "false"],
                "north": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
            },
        },
        {
            "base_name": "chain",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "quartz_pillar",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        # {
        #     "base_name": "copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "exposed_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "weathered_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "oxidized_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "waxed_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "waxed_exposed_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "waxed_weathered_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "waxed_oxidized_copper_bulb",
        #     "varying_params": {
        #         "lit": ["false", "true"],
        #     },
        # },
        {
            "base_name": "glazed_terracotta",
            "separate_params": {
                "color": [
                    "white",
                    "light_gray",
                    "gray",
                    "black",
                    "brown",
                    "red",
                    "orange",
                    "yellow",
                    "lime",
                    "green",
                    "cyan",
                    "light_blue",
                    "blue",
                    "purple",
                    "magenta",
                    "pink",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "glass_pane",
            "varying_params": {
                "east": ["true", "false"],
                "north": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
            },
        },
        {
            "base_name": "stained_glass_pane",
            "separate_params": {
                "color": [
                    "white",
                    "light_gray",
                    "gray",
                    "black",
                    "brown",
                    "red",
                    "orange",
                    "yellow",
                    "lime",
                    "green",
                    "cyan",
                    "light_blue",
                    "blue",
                    "purple",
                    "magenta",
                    "pink",
                ],
            },
            "varying_params": {
                "east": ["true", "false"],
                "north": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
            },
        },
        {
            "base_name": "shulker_box",
            "separate_params": {
                "color": [
                    "default",
                    "white",
                    "orange",
                    "magenta",
                    "light_blue",
                    "yellow",
                    "lime",
                    "pink",
                    "gray",
                    "light_gray",
                    "cyan",
                    "purple",
                    "blue",
                    "brown",
                    "green",
                    "red",
                    "black",
                ],
            },
            "varying_params": {
                "facing": ["down", "east", "north", "south", "up", "west"],
            },
        },
        {
            "base_name": "bed",
            "separate_params": {
                "color": [
                    "white",
                    # "orange",
                    # "magenta",
                    "light_blue",
                    "yellow",
                    # "lime",
                    # "pink",
                    "gray",
                    # "light_gray",
                    # "cyan",
                    # "purple",
                    "blue",
                    "brown",
                    # "green",
                    "red",
                    "black",
                ],
            },
            "varying_params": {
                "part": ["foot", "head"],
                "facing": ["north", "east", "south", "west"],
                "occupied": ["false", "true"],
            },
        },
        {
            "base_name": "candle",
            "separate_params": {
                "color": [
                    "default",
                    "white",
                    "orange",
                    "magenta",
                    "light_blue",
                    "yellow",
                    # "lime",
                    "pink",
                    "gray",
                    "light_gray",
                    "cyan",
                    "purple",
                    "blue",
                    "brown",
                    "green",
                    "red",
                    "black",
                ],
            },
            "varying_params": {
                "candles": ["1", "2", "3", "4"],
                "lit": ["false", "true"],
            },
        },
        {
            "base_name": "banner",
            "separate_params": {
                "color": [
                    "white",
                    # "orange",
                    # "magenta",
                    # "light_blue",
                    # "yellow",
                    "lime",
                    # "pink",
                    # "gray",
                    # "light_gray",
                    # "cyan",
                    # "purple",
                    # "blue",
                    # "brown",
                    # "green",
                    "red",
                    "black",
                ],
            },
            "varying_params": {
                "rotation": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                ],
            },
        },
        {
            "base_name": "wall_banner",
            "separate_params": {
                "color": [
                    "white",
                    "orange",
                    "magenta",
                    "light_blue",
                    "yellow",
                    "lime",
                    "pink",
                    "gray",
                    "light_gray",
                    "cyan",
                    "purple",
                    "blue",
                    "brown",
                    "green",
                    "red",
                    "black",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "farmland",
            "varying_params": {
                "moisture": ["0", "1", "2", "3", "4", "5", "6", "7"],
            },
        },
        {
            "base_name": "snow",
            "varying_params": {
                "layers": ["1", "2", "3", "4", "5", "6", "7", "8"],
            },
        },
        # {
        #     "base_name": "pale_moss_carpet",
        # },
        # {
        #     "base_name": "pale_hanging_moss",
        # },
        # {
        #     "base_name": "dripstone",
        #     "varying_params": {
        #         "thickness": ["tip_merge", "tip", "frustum", "middle", "base"],
        #         "vertical_direction": ["down", "up"],
        #     },
        # },
        {
            "base_name": "bone_block",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "small_amethyst_bud",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"],
            },
        },
        {
            "base_name": "medium_amethyst_bud",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"],
            },
        },
        {
            "base_name": "large_amethyst_bud",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"],
            },
        },
        {
            "base_name": "amethyst_cluster",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"],
            },
        },
        {
            "base_name": "muddy_mangrove_roots",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        # {
        #     "base_name": "mushroom_stem",
        #     "separate_params": {
        #         "material": ["red", "brown"],
        #     },
        #     "varying_params": {
        #         "up": ["true", "false"],
        #         "down": ["true", "false"],
        #         "north": ["true", "false"],
        #         "east": ["true", "false"],
        #         "south": ["true", "false"],
        #         "west": ["true", "false"],
        #     },
        # },
        {
            "base_name": "red_mushroom_block",
            "varying_params": {
                "up": ["true", "false"],
                "down": ["true", "false"],
                "north": ["true", "false"],
                "east": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
            },
        },
        {
            "base_name": "brown_mushroom_block",
            "varying_params": {
                "up": ["true", "false"],
                "down": ["true", "false"],
                "north": ["true", "false"],
                "east": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
            },
        },
        {
            "base_name": "pink_petals",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "flower_amount": ["1", "2", "3", "4"],
            },
        },
        {
            "base_name": "bamboo",
            "varying_params": {
                "leaves": ["none", "small", "large"],
                "thickness": ["thin", "thick"],
            },
        },
        {
            "base_name": "vine",
            "varying_params": {
                "east": ["true", "false"],
                "north": ["true", "false"],
                "south": ["true", "false"],
                "west": ["true", "false"],
                "up": ["true", "false"],
            },
        },
        {
            "base_name": "double_plant",
            "separate_params": {
                "plant_type": [
                    "sunflower",
                    "lilac",
                    "rose_bush",
                    "peony",
                    "tall_grass",
                    "large_fern",
                ],
            },
            "varying_params": {
                "half": ["upper", "lower"],
            },
        },
        {
            "base_name": "pitcher_plant",
            "varying_params": {
                "half": ["upper", "lower"],
            },
        },
        {
            "base_name": "big_dripleaf",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "tilt": ["none"],
            },
        },
        {
            "base_name": "big_dripleaf_stem",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "tilt": ["none"],
            },
        },
        {
            "base_name": "small_dripleaf",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "half": ["upper", "lower"],
            },
        },
        # {
        #     "base_name": "glow_lichen",
        #     "varying_params": {
        #         "north": ["true", "false"],
        #         "east": ["true", "false"],
        #         "south": ["true", "false"],
        #         "west": ["true", "false"],
        #         "up": ["true", "false"],
        #         "down": ["true", "false"],
        #     },
        # },
        {
            "base_name": "turtle_egg",
            "varying_params": {
                "eggs": ["1", "2", "3", "4"],
                "hatch": ["0", "1", "2"],
            },
        },
        {
            "base_name": "wheat",
            "varying_params": {
                "age": ["0", "1", "2", "3", "4", "5", "6", "7"],
            },
        },
        {
            "base_name": "cocoa",
            "varying_params": {
                "age": ["0", "1", "2"],
                "facing": ["north", "east", "south", "west"],
            },
        },
        # {
        #     "base_name": "pumpkin_stem",
        #     "varying_params": {
        #         "facing": ["none", "north", "east", "south", "west"],
        #         "age": ["0", "1", "2", "3", "4", "5", "6", "7"],
        #     },
        # },
        # {
        #     "base_name": "melon_stem",
        #     "varying_params": {
        #         "facing": ["none", "north", "east", "south", "west"],
        #         "age": ["0", "1", "2", "3", "4", "5", "6", "7"],
        #     },
        # },
        # {
        #     "base_name": "beetroots",
        #     "varying_params": {
        #         "age": ["0", "1", "2", "3", "4", "5", "6", "7"],
        #     },
        # },
        # {
        #     "base_name": "pitcher_crop",
        #     "varying_params": {
        #         "half": ["upper", "lower"],
        #         "age": ["0", "1", "2", "3", "4"],
        #     },
        # },
        {
            "base_name": "cave_vines",
            "varying_params": {
                "berries": ["true", "false"],
            },
        },
        {
            "base_name": "cave_vines_plant",
            "varying_params": {
                "berries": ["true", "false"],
                "age": [
                    "0",
                    # "1",
                    # "2",
                    # "3",
                    # "4",
                    # "5",
                    # "6",
                    # "7",
                    # "8",
                    # "9",
                    # "10",
                    # "11",
                    # "12",
                    # "13",
                    # "14",
                    # "15",
                    # "16",
                    # "17",
                    # "18",
                    # "19",
                    # "20",
                    # "21",
                    # "22",
                    # "23",
                    # "24",
                    # "25",
                ],
            },
        },
        {
            "base_name": "nether_wart",
            "varying_params": {
                "age": ["0", "1", "2", "3"],
            },
        },
        {
            "base_name": "tall_seagrass",
            "common_params": {"waterlogged": "true"},
            "varying_params": {
                "half": ["upper", "lower"],
            },
        },
        # {
        #     "base_name": "sea_pickle",
        #     "varying_params": {
        #         "pickles": ["1", "2", "3", "4"],
        #         "dead": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "coral_block",
        #     "separate_params": {
        #         "color": ["tube", "brain", "bubble", "fire", "horn"],
        #     },
        #     "varying_params": {
        #         "dead": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "coral",
        #     "separate_params": {
        #         "color": ["tube", "brain", "bubble", "fire", "horn"],
        #     },
        #     "varying_params": {
        #         "dead": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "coral_fan",
        #     "separate_params": {
        #         "color": ["tube", "brain", "bubble", "fire", "horn"],
        #     },
        #     "varying_params": {
        #         "facing": ["north", "east", "south", "west", "up", "down"],
        #         "dead": ["false", "true"],
        #     },
        # },
        {
            "base_name": "sponge",
            "varying_params": {
                "wet": ["false", "true"],
            },
        },
        # {
        #     "base_name": "pumpkin",
        #     "varying_params": {
        #         "facing": ["north", "east", "south", "west"],
        #     },
        # },
        {
            "base_name": "carved_pumpkin",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "jack_o_lantern",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "hay_block",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        # {
        #     "base_name": "bee_nest",
        #     "varying_params": {
        #         "facing": ["north", "east", "south", "west"],
        #         "honey_level": ["0", "1", "2", "3", "4", "5"],
        #     },
        # },
        {
            "base_name": "ochre_froglight",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "pearlescent_froglight",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        {
            "base_name": "verdant_froglight",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        # {
        #     "base_name": "sculk_vein",
        #     "varying_params": {
        #         "down": ["true", "false"],
        #         "east": ["true", "false"],
        #         "north": ["true", "false"],
        #         "south": ["true", "false"],
        #         "up": ["true", "false"],
        #         "west": ["true", "false"],
        #     },
        # },
        # {
        #     "base_name": "sculk_catalyst",
        #     "varying_params": {
        #         "bloom": ["false", "true"],
        #     },
        # },
        # {
        #     "base_name": "bedrock",
        #     "varying_params": {
        #         "infiniburn": ["false", "true"],
        #     },
        # },
        {
            "base_name": "torch",
            "varying_params": {
                "facing": ["east", "north", "south", "up", "west"],
            },
        },
        {
            "base_name": "soul_torch",
            "varying_params": {
                "facing": ["east", "north", "south", "up", "west"],
            },
        },
        {
            "base_name": "redstone_torch",
            "varying_params": {
                "facing": ["east", "north", "south", "up", "west"],
                "lit": ["false", "true"],
            },
        },
        {
            "base_name": "lantern",
            "varying_params": {
                "hanging": ["false", "true"],
            },
        },
        {
            "base_name": "soul_lantern",
            "varying_params": {
                "hanging": ["false", "true"],
            },
        },
        {
            "base_name": "end_rod",
            "varying_params": {
                "facing": ["east", "north", "south", "up", "west", "down"],
            },
        },
        {
            "base_name": "redstone_lamp",
            "varying_params": {
                "lit": ["false", "true"],
            },
        },
        {
            "base_name": "stonecutter",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "grindstone",
            "varying_params": {
                "face": ["floor", "wall", "ceiling"],
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "loom",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "furnace",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "lit": ["false", "true"],
            },
        },
        {
            "base_name": "smoker",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "lit": ["false", "true"],
            },
        },
        {
            "base_name": "blast_furnace",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "lit": ["false", "true"],
            },
        },
        {
            "base_name": "campfire",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "lit": ["false", "true"],
                "signal_fire": ["false", "true"],
            },
        },
        # {
        #     "base_name": "soul_campfire",
        #     "varying_params": {
        #         "facing": ["north", "east", "south", "west"],
        #         "lit": ["false", "true"],
        #         "signal_fire": ["false", "true"],
        #     },
        # },
        {
            "base_name": "anvil",
            "varying_params": {
                "damage": ["normal", "chipped", "damaged"],
                "facing": ["north", "east", "south", "west"],
            },
        },
        # {
        #     "base_name": "composter",
        #     "varying_params": {
        #         "level": ["0", "1", "2", "3", "4", "5", "6", "7", "8"],
        #     },
        # },
        {
            "base_name": "brewing_stand",
            "varying_params": {
                "has_bottle_0": ["false", "true"],
                "has_bottle_1": ["false", "true"],
                "has_bottle_2": ["false", "true"],
            },
        },
        # {
        #     "base_name": "cauldron",
        #     "varying_params": {
        #         "cauldron_liquid": ["water", "lava", "powder_snow"],
        #         "level": ["0", "1", "2", "3"],
        #     },
        # },
        {
            "base_name": "bell",
            "varying_params": {
                "attachment": ["ceiling", "double_wall", "floor", "single_wall"],
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "water",
            "varying_params": {
                "level": ["0", "1", "2", "3", "4", "5", "6", "7"],
                "falling": ["false", "true"],
                "flowing": ["false", "true"],
            },
        },
        # {
        #     "base_name": "lava",
        #     "varying_params": {
        #         "level": ["0", "1", "2", "3", "4", "5", "6", "7"],
        #         "falling": ["false", "true"],
        #         "flowing": ["false", "true"],
        #     },
        # },
        {
            "base_name": "ladder",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "beehive",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "honey_level": ["0", "1", "2", "3", "4", "5"],
            },
        },
        {
            "base_name": "lightning_rod",
            "varying_params": {
                "facing": ["down", "east", "north", "south", "up", "west"],
            },
        },
        {
            "base_name": "flower_pot",
            "varying_params": {
                "plant": [
                    "dandelion",
                    "poppy",
                    "blue_orchid",
                    "allium",
                    "azure_bluet",
                    "red_tulip",
                    "orange_tulip",
                    "white_tulip",
                    "pink_tulip",
                    "oxeye_daisy",
                    "oak_sapling",
                    "spruce_sapling",
                    "birch_sapling",
                    "jungle_sapling",
                    "acacia_sapling",
                    "dark_oak_sapling",
                    "red_mushroom",
                    "brown_mushroom",
                    "fern",
                    "dead_bush",
                    "cactus",
                    "cornflower",
                    "wither_rose",
                    "lily_of_the_valley",
                    "bamboo",
                    "crimson_fungus",
                    "warped_fungus",
                    "crimson_roots",
                    "warped_roots",
                    "azalea_bush",
                    "flowering_azalea_bush",
                    "mangrove_propagule",
                    "cherry_sapling",
                    "torchflower",
                    # "pale_oak",
                    # "closed_eyeblossom",
                    # "open_eyeblossom",
                ]
            },
        },
        # {
        #     "base_name": "decorated_pot",
        #     "varying_params": {
        #         "cracked": ["false", "true"],
        #         "facing": ["north", "east", "south", "west"],
        #     },
        # },
        # {
        #     "base_name": "chiseled_bookshelf",
        #     "varying_params": {
        #         "facing": ["north", "east", "south", "west"],
        #         "slot_0_occupied": ["false", "true"],
        #         "slot_1_occupied": ["false", "true"],
        #         "slot_2_occupied": ["false", "true"],
        #         "slot_3_occupied": ["false", "true"],
        #         "slot_4_occupied": ["false", "true"],
        #         "slot_5_occupied": ["false", "true"],
        #     },
        # },
        {
            "base_name": "lectern",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "has_book": ["false", "true"],
            },
        },
        {
            "base_name": "sign",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    # "jungle",
                    "acacia",
                    "dark_oak",
                    # "mangrove",
                    # "crimson",
                    "warped",
                    # "bamboo",
                    # "cherry",
                    # "pale_oak",
                ],
            },
            "varying_params": {
                "rotation": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                ]
            },
        },
        {
            "base_name": "wall_sign",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    "acacia",
                    "dark_oak",
                    "mangrove",
                    "crimson",
                    "warped",
                    "bamboo",
                    "cherry",
                    # "pale_oak",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        # {
        #     "base_name": "hanging_sign",
        #     "separate_params": {
        #         "material": [
        #             "oak",
        #             "spruce",
        #             "birch",
        #             "jungle",
        #             "acacia",
        #             "dark_oak",
        #             "mangrove",
        #             "crimson",
        #             "warped",
        #             "bamboo",
        #             "cherry",
        #             "pale_oak",
        #         ],
        #     },
        #     "varying_params": {
        #         "connection": ["up", "up_chain"],
        #         "rotation": [
        #             "0",
        #             "1",
        #             "2",
        #             "3",
        #             "4",
        #             "5",
        #             "6",
        #             "7",
        #             "8",
        #             "9",
        #             "10",
        #             "11",
        #             "12",
        #             "13",
        #             "14",
        #             "15",
        #         ],
        #     },
        # },
        {
            "base_name": "wall_hanging_sign",
            "separate_params": {
                "material": [
                    "oak",
                    "spruce",
                    "birch",
                    "jungle",
                    # "acacia",
                    "dark_oak",
                    "mangrove",
                    "crimson",
                    "warped",
                    "bamboo",
                    "cherry",
                    # "pale_oak",
                ],
            },
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "chest",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "connection": ["none", "left", "right"],
            },
        },
        {
            "base_name": "barrel",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "ender_chest",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "respawn_anchor",
            "varying_params": {
                "charges": ["0", "1", "2", "3", "4"],
            },
        },
        {
            "base_name": "head",
            "varying_params": {
                "mob": [
                    "creeper",
                    "dragon",
                    # "piglin",
                    "player",
                    "skeleton",
                    "wither_skeleton",
                    "zombie",
                ],
                "rotation": [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                ],
            },
        },
        {
            "base_name": "end_portal_frame",
            "varying_params": {
                "eye": ["false", "true"],
                "facing": ["north", "east", "south", "west"],
            },
        },
        {
            "base_name": "nether_portal",
            "varying_params": {
                "axis": ["x", "z"],
            },
        },
        # {
        #     "base_name": "vault",
        #     "varying_params": {
        #         "facing": ["north", "east", "south", "west"],
        #         "ominous": ["false", "true"],
        #         "vault_state": ["active", "ejecting", "inactive", "unlocking"],
        #     },
        # },
        {
            "base_name": "infested_deepslate",
            "varying_params": {
                "axis": ["x", "y", "z"],
            },
        },
        # {
        #     "base_name": "redstone_wire",
        #     "varying_params": {
        #         "east": ["none", "side", "up"],
        #         "north": ["none", "side", "up"],
        #         "south": ["none", "side", "up"],
        #         "west": ["none", "side", "up"],
        #         "power": [
        #             "0",
        #             "1",
        #             "2",
        #             "3",
        #             "4",
        #             "5",
        #             "6",
        #             "7",
        #             "8",
        #             "9",
        #             "10",
        #             "11",
        #             "12",
        #             "13",
        #             "14",
        #             "15",
        #         ],
        #     },
        # },
        # {
        #     "base_name": "repeater",
        #     "varying_params": {
        #         "delay": ["1", "2", "3", "4"],
        #         "facing": ["north", "east", "south", "west"],
        #         "locked": ["false", "true"],
        #         "powered": ["false", "true"],
        #     },
        # },
        {
            "base_name": "comparator",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "mode": ["compare", "subtract"],
                "powered": ["false", "true"],
            },
        },
        {
            "base_name": "lever",
            "varying_params": {
                "face": ["floor", "wall"],
                "facing": ["north", "east", "south", "west"],
                "powered": ["false", "true"],
            },
        },
        {
            "base_name": "tripwire_hook",
            "varying_params": {
                "attached": ["false", "true"],
                "facing": ["north", "east", "south", "west"],
                "powered": ["false", "true"],
            },
        },
        {
            "base_name": "daylight_detector",
            "varying_params": {
                "inverted": ["false", "true"],
            },
        },
        {
            "base_name": "piston",
            "varying_params": {
                "extended": ["false", "true"],
                "facing": ["north", "east", "south", "west", "up", "down"],
            },
        },
        {
            "base_name": "sticky_piston",
            "varying_params": {
                "extended": ["false", "true"],
                "facing": ["north", "east", "south", "west", "up", "down"],
            },
        },
        {
            "base_name": "piston_head",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"],
                "short": ["false"],
            },
        },
        {
            "base_name": "sticky_piston_head",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"],
                "short": ["false"],
            },
        },
        {
            "base_name": "dispenser",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"]
            },
        },
        {
            "base_name": "dropper",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"]
            },
        },
        # {
        #     "base_name": "crafter",
        #     "varying_params": {
        #         "crafting": ["false", "true"],
        #         "orientation": [
        #             "down_east",
        #             "down_north",
        #             "down_south",
        #             "down_west",
        #             "up_east",
        #             "up_north",
        #             "up_south",
        #             "up_west",
        #             "west_up",
        #             "east_up",
        #             "north_up",
        #             "south_up",
        #         ],
        #         "triggered": ["false", "true"],
        #     },
        # },
        {
            "base_name": "hopper",
            "varying_params": {"facing": ["north", "east", "south", "west", "down"]},
        },
        {
            "base_name": "trapped_chest",
            "varying_params": {
                "facing": ["north", "east", "south", "west"],
                "connection": ["none", "left", "right"],
            },
        },
        {
            "base_name": "observer",
            "varying_params": {
                "facing": ["north", "east", "south", "west", "up", "down"]
            },
        },
        {
            "base_name": "rail",
            "varying_params": {
                "shape": [
                    "north_south",
                    "east_west",
                    "ascending_east",
                    "ascending_west",
                    "ascending_north",
                    "ascending_south",
                    "south_east",
                    "south_west",
                    "north_west",
                    "north_east",
                ],
            },
        },
        {
            "base_name": "powered_rail",
            "varying_params": {
                "shape": [
                    "north_south",
                    "east_west",
                    "ascending_east",
                    "ascending_west",
                    "ascending_north",
                    "ascending_south",
                    # "south_east",
                    # "south_west",
                    # "north_west",
                    # "north_east",
                ],
                "powered": ["false", "true"],
            },
        },
        {
            "base_name": "detector_rail",
            "varying_params": {
                "shape": [
                    "north_south",
                    "east_west",
                    "ascending_east",
                    "ascending_west",
                    "ascending_north",
                    "ascending_south",
                    # "south_east",
                    # "south_west",
                    # "north_west",
                    # "north_east",
                ],
                "powered": ["false", "true"],
            },
        },
        {
            "base_name": "activator_rail",
            "varying_params": {
                "shape": [
                    "north_south",
                    "east_west",
                    "ascending_east",
                    "ascending_west",
                    "ascending_north",
                    "ascending_south",
                    # "south_east",
                    # "south_west",
                    # "north_west",
                    # "north_east",
                ],
                "powered": ["false", "true"],
            },
        },
        {
            "base_name": "carrots",
            "varying_params": {
                "age": ["0", "1", "2", "3", "4", "5", "6", "7"],
            },
        },
        {
            "base_name": "potatoes",
            "varying_params": {
                "age": ["0", "1", "2", "3", "4", "5", "6", "7"],
            },
        },
        {
            "base_name": "cake",
            "varying_params": {
                "bites": ["0", "1", "2", "3", "4", "5", "6"],
            },
        },
        # {
        #     "base_name": "command_block",
        #     "separate_params": {
        #         "mode": ["impulse", "chain", "repeating"],
        #     },
        #     "varying_params": {
        #         "conditional": ["false", "true"],
        #         "facing": ["north", "east", "south", "west", "up", "down"],
        #     },
        # },
        # {
        #     "base_name": "chorus_plant",
        #     "varying_params": {
        #         "down": ["false", "true"],
        #         "east": ["false", "true"],
        #         "north": ["false", "true"],
        #         "south": ["false", "true"],
        #         "up": ["false", "true"],
        #         "west": ["false", "true"],
        #     },
        # },
    ]

    BLOCK_TRANSFORMS = [
        {
            # Dirt
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "input": "universal_minecraft:dirt",
            "outputs": [
                {
                    "base_name": "grass_block",
                    "varying_params": {
                        "snowy": ["false", "true"],
                    },
                },
                {
                    "base_name": "mycelium",
                    "varying_params": {
                        "snowy": ["false", "true"],
                    },
                },
            ],
        },
        {
            # Netherrack
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "input": "universal_minecraft:netherrack",
            "outputs": [
                {
                    "base_name": "warped_nylium",
                },
                {
                    "base_name": "crimson_nylium",
                },
            ],
        },
    ]

    BLOCK_PROPERTY_TRANSFORMS = [
        {
            # Fence north connection
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "fence",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "nether_brick",
                        "crimson",
                        "warped",
                        "bamboo",
                        "cherry",
                        "pale_oak",
                    ],
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "north",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Fence east connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "fence",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "nether_brick",
                        "crimson",
                        "warped",
                        "bamboo",
                        "cherry",
                        "pale_oak",
                    ],
                    "north": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "east",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Fence south connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "fence",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "nether_brick",
                        "crimson",
                        "warped",
                        "bamboo",
                        "cherry",
                        "pale_oak",
                    ],
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "south",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Fence west connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "fence",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "nether_brick",
                        "crimson",
                        "warped",
                        "bamboo",
                        "cherry",
                        "pale_oak",
                    ],
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                },
                "param": "west",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Glass pane north connection
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "glass_pane",
                "varying_params": {
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "north",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Glass pane east connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "glass_pane",
                "varying_params": {
                    "north": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "east",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Glass pane south connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "glass_pane",
                "varying_params": {
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "south",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Glass pane west connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "glass_pane",
                "varying_params": {
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                },
                "param": "west",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Stained glass pane north connection
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stained_glass_pane",
                "varying_params": {
                    "color": [
                        "white",
                        "light_gray",
                        "gray",
                        "black",
                        "brown",
                        "red",
                        "orange",
                        "yellow",
                        "lime",
                        "green",
                        "cyan",
                        "light_blue",
                        "blue",
                        "purple",
                        "magenta",
                        "pink",
                    ],
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "north",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Stained glass pane east connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stained_glass_pane",
                "varying_params": {
                    "color": [
                        "white",
                        "light_gray",
                        "gray",
                        "black",
                        "brown",
                        "red",
                        "orange",
                        "yellow",
                        "lime",
                        "green",
                        "cyan",
                        "light_blue",
                        "blue",
                        "purple",
                        "magenta",
                        "pink",
                    ],
                    "north": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "east",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Stained glass pane south connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stained_glass_pane",
                "varying_params": {
                    "color": [
                        "white",
                        "light_gray",
                        "gray",
                        "black",
                        "brown",
                        "red",
                        "orange",
                        "yellow",
                        "lime",
                        "green",
                        "cyan",
                        "light_blue",
                        "blue",
                        "purple",
                        "magenta",
                        "pink",
                    ],
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "south",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Stained glass pane west connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stained_glass_pane",
                "varying_params": {
                    "color": [
                        "white",
                        "light_gray",
                        "gray",
                        "black",
                        "brown",
                        "red",
                        "orange",
                        "yellow",
                        "lime",
                        "green",
                        "cyan",
                        "light_blue",
                        "blue",
                        "purple",
                        "magenta",
                        "pink",
                    ],
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                },
                "param": "west",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Iron bars north connection
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "iron_bars",
                "varying_params": {
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "north",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Iron bars east connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "iron_bars",
                "varying_params": {
                    "north": ["false", "true"],
                    "south": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "east",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Iron bars south connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "iron_bars",
                "varying_params": {
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "west": ["false", "true"],
                },
                "param": "south",
                "input": "true",
                "output": "false",
            },
        },
        {
            # Iron bars west connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "iron_bars",
                "varying_params": {
                    "north": ["false", "true"],
                    "east": ["false", "true"],
                    "south": ["false", "true"],
                },
                "param": "west",
                "input": "true",
                "output": "false",
            },
        },
        # TODO: Walls: Similar logic to fences/panes/bars, but a lot more complicated
        # TODO: Fence gate: Remove in_wall property if non wall on both sides
        {
            # Chest left north
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["north"],
                },
                "param": "connection",
                "input": "left",
                "output": "none",
            },
        },
        {
            # Chest right north
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["north"],
                },
                "param": "connection",
                "input": "right",
                "output": "none",
            },
        },
        {
            # Chest left east
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["east"],
                },
                "param": "connection",
                "input": "left",
                "output": "none",
            },
        },
        {
            # Chest right east
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["east"],
                },
                "param": "connection",
                "input": "right",
                "output": "none",
            },
        },
        {
            # Chest left south
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["south"],
                },
                "param": "connection",
                "input": "left",
                "output": "none",
            },
        },
        {
            # Chest right south
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["south"],
                },
                "param": "connection",
                "input": "right",
                "output": "none",
            },
        },
        {
            # Chest left west
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["west"],
                },
                "param": "connection",
                "input": "left",
                "output": "none",
            },
        },
        {
            # Chest right west
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "chest",
                "varying_params": {
                    "facing": ["west"],
                },
                "param": "connection",
                "input": "right",
                "output": "none",
            },
        },
        {
            # Stairs north front
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["north"],
                    "shape": ["outer_left", "outer_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs north back
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["north"],
                    "shape": ["inner_left", "inner_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs east front
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["east"],
                    "shape": ["outer_left", "outer_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs east back
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["east"],
                    "shape": ["inner_left", "inner_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs south front
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["south"],
                    "shape": ["outer_left", "outer_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs south back
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["south"],
                    "shape": ["inner_left", "inner_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs west front
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["west"],
                    "shape": ["outer_left", "outer_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        {
            # Stairs west back
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "stairs",
                "varying_params": {
                    "material": [
                        "oak",
                        "spruce",
                        "birch",
                        "jungle",
                        "acacia",
                        "dark_oak",
                        "mangrove",
                        "brick",
                        "cobblestone",
                        "stone_brick",
                        "sandstone",
                        "red_sandstone",
                        "nether_brick",
                        "quartz",
                        "purpur",
                        "prismarine",
                        "prismarine_brick",
                        "dark_prismarine",
                        "andesite",
                        "diorite",
                        "end_stone_brick",
                        "granite",
                        "mossy_cobblestone",
                        "mossy_stone_brick",
                        "polished_andesite",
                        "polished_diorite",
                        "polished_granite",
                        "red_nether_brick",
                        "smooth_quartz",
                        "smooth_red_sandstone",
                        "smooth_sandstone",
                        "stone",
                        "crimson",
                        "warped",
                        "blackstone",
                        "polished_blackstone_brick",
                        "polished_blackstone",
                        "oxidized_cut_copper",
                        "weathered_cut_copper",
                        "exposed_cut_copper",
                        "cut_copper",
                        "waxed_oxidized_cut_copper",
                        "waxed_weathered_cut_copper",
                        "waxed_exposed_cut_copper",
                        "waxed_cut_copper",
                        "cobbled_deepslate",
                        "polished_deepslate",
                        "deepslate_tile",
                        "deepslate_brick",
                        "mud_brick",
                        "bamboo",
                        "bamboo_mosaic",
                        "cherry",
                        "tuff",
                        "tuff_brick",
                        "polished_tuff",
                        "pale_oak",
                        "resin_brick",
                    ],
                    "half": ["bottom", "top"],
                    "facing": ["west"],
                    "shape": ["inner_left", "inner_right"],
                },
                "param": "shape",
                "output": "straight",
            },
        },
        # TODO: Mushroom blocks and stems
        {
            # Rail north ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "param": "shape",
                "input": "ascending_north",
                "output": "north_south",
            },
        },
        {
            # Rail east ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "param": "shape",
                "input": "ascending_east",
                "output": "east_west",
            },
        },
        {
            # Rail south ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "param": "shape",
                "input": "ascending_south",
                "output": "north_south",
            },
        },
        {
            # Rail west ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "param": "shape",
                "input": "ascending_west",
                "output": "east_west",
            },
        },
        {
            # Detector rail north ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "detector_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_north",
                "output": "north_south",
            },
        },
        {
            # Detector rail east ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "detector_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_east",
                "output": "east_west",
            },
        },
        {
            # Detector rail south ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            ],
            "transform": {
                "base_name": "detector_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_south",
                "output": "north_south",
            },
        },
        {
            # Detector rail west ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "detector_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_west",
                "output": "east_west",
            },
        },
        {
            # Activator rail north ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "activator_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_north",
                "output": "north_south",
            },
        },
        {
            # Activator rail east ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "activator_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_east",
                "output": "east_west",
            },
        },
        {
            # Activator rail south ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            ],
            "transform": {
                "base_name": "activator_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_south",
                "output": "north_south",
            },
        },
        {
            # Activator rail west ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "activator_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_west",
                "output": "east_west",
            },
        },
        {
            # Powered rail north ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "powered_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_north",
                "output": "north_south",
            },
        },
        {
            # Powered rail east ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "powered_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_east",
                "output": "east_west",
            },
        },
        {
            # Powered rail south ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            ],
            "transform": {
                "base_name": "powered_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_south",
                "output": "north_south",
            },
        },
        {
            # Powered rail west ascending
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "powered_rail",
                "varying_params": {
                    "powered": ["false", "true"],
                },
                "param": "shape",
                "input": "ascending_west",
                "output": "east_west",
            },
        },
        {
            # Rail north bend
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "varying_params": {
                    "shape": ["north_west", "north_east"],
                },
                "param": "shape",
                "output": "east_west",
            },
        },
        {
            # Rail east bend
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "varying_params": {
                    "shape": ["south_east", "north_east"],
                },
                "param": "shape",
                "output": "north_south",
            },
        },
        {
            # Rail south bend
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "varying_params": {
                    "shape": ["south_west", "south_east"],
                },
                "param": "shape",
                "output": "east_west",
            },
        },
        {
            # Rail west bend
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "rail",
                "varying_params": {
                    "shape": ["south_west", "north_west"],
                },
                "param": "shape",
                "output": "north_south",
            },
        },
        # TODO: Chorus plants
        {
            # Redstone north side connection
            "kernel": [
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "east": ["none", "side", "up"],
                    "south": ["none", "side", "up"],
                    "west": ["none", "side", "up"],
                },
                "param": "north",
                "input": "side",
                "output": "none",
            },
        },
        {
            # Redstone east side connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "north": ["none", "side", "up"],
                    "south": ["none", "side", "up"],
                    "west": ["none", "side", "up"],
                },
                "param": "east",
                "input": "side",
                "output": "none",
            },
        },
        {
            # Redstone south side connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "east": ["none", "side", "up"],
                    "north": ["none", "side", "up"],
                    "west": ["none", "side", "up"],
                },
                "param": "south",
                "input": "side",
                "output": "none",
            },
        },
        {
            # Redstone west side connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "east": ["none", "side", "up"],
                    "south": ["none", "side", "up"],
                    "north": ["none", "side", "up"],
                },
                "param": "west",
                "input": "side",
                "output": "none",
            },
        },
        {
            # Redstone north side up connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "east": ["none", "side", "up"],
                    "south": ["none", "side", "up"],
                    "west": ["none", "side", "up"],
                },
                "param": "north",
                "input": "up",
                "output": "none",
            },
        },
        {
            # Redstone east side up connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "north": ["none", "side", "up"],
                    "south": ["none", "side", "up"],
                    "west": ["none", "side", "up"],
                },
                "param": "east",
                "input": "up",
                "output": "none",
            },
        },
        {
            # Redstone south side up connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "east": ["none", "side", "up"],
                    "north": ["none", "side", "up"],
                    "west": ["none", "side", "up"],
                },
                "param": "south",
                "input": "up",
                "output": "none",
            },
        },
        {
            # Redstone west side up connection
            "kernel": [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            "transform": {
                "base_name": "redstone_wire",
                "varying_params": {
                    "power": [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                    ],
                    "east": ["none", "side", "up"],
                    "south": ["none", "side", "up"],
                    "north": ["none", "side", "up"],
                },
                "param": "west",
                "input": "up",
                "output": "none",
            },
        },
    ]

    def __init__(
        self,
        ignore_natural_blocks_chance: float = 0.75,
        mask_full_blocks_chance: float = 0.75,
        treat_family_as_single_block_chance: float = 0.5,
        percentage_block_types_to_mask: float = 0.5,
        remove_block_type_chance: float = 0.25,
        min_mask_bias: float = 0.001,
        max_mask_bias: float = 2.0,
        invert_mask_chance: float = 0.5,
        mask_origin_extra_area_percent: float = 0.5,
        flat_mask_chance: float = 0.25,
        max_mask_edge_min_from_center: int = 1,
        mask_shell_chance: float = 0.75,
        skip_block_transform_chance: float = 0.5,
        random_block_transform_chance: float = 0.5,
        skip_block_property_transform_chance: float = 0.5,
        random_block_property_transform_chance: float = 0.5,
        add_blocks_chance: float = 0.5,
        min_autoregressive_start_radius: int = 0,
        max_autoregressive_start_radius: int = 2,
        add_blocks_amount_first_dist_chance: float = 0.75,
        add_blocks_amount_first_dist_amount: int = 50,
        add_blocks_amount_first_dist_beta: int = 5,
        add_blocks_amount_second_dist_beta: int = 3,
    ):
        """Initialize the StructureMasker.

        Args:
            ignore_natural_blocks_chance (float, optional): The probability that a natural block will be ignored. Defaults to 0.75.
            mask_full_blocks_chance (float, optional): The probability that the mask will be applied to the full set of blocks in a structure. Defaults to 0.75.
            treat_family_as_single_block_chance (float, optional): The probability that blocks in a family (all stairs of a certain material, the same block in different orientations, etc..) will be treated as a single block for masking. Defaults to 0.5.
            percentage_block_types_to_mask (float, optional): The percentage of unique block types to apply an individual mask for. Defaults to 0.5.
            remove_block_type_chance (float, optional): The probability that a block type will be removed entirely instead of the normal mask. Defaults to 0.25.
            min_mask_bias (float, optional): The minimum bias to use when applying the mask operation. Defaults to 0.001.
            max_mask_bias (float, optional): The maximum bias to use when applying the mask operation. Defaults to 2.0.
            invert_mask_chance (float, optional): The probability that the mask will be inverted. Defaults to 0.5.
            mask_origin_extra_area_percent (float, optional): The amount of extra area where the mask can originate from beyond the structure bounds represented as a percentage of the structure size. Defaults to 0.5.
            flat_mask_chance (float, optional): The probability that the flat variant of the mask will be used. Defaults to 0.25.
            max_mask_edge_min_from_center (int, optional): The maximum distance from the center the edge of the mask can be. Defaults to 1.
            mask_shell_chance (float, optional): The probability that the mask will have a rabdin sized solid shell around the random mask area. Defaults to 0.75.
            skip_block_transform_chance (float, optional): The probability that a block transform will be skipped. Defaults to 0.5.
            random_block_transform_chance (float, optional): The probability that a block transform will be randomised. Defaults to 0.5.
            skip_block_property_transform_chance (float, optional): The probability that a block property transform will be skipped. Defaults to 0.5.
            random_block_property_transform_chance (float, optional): The probability that a block property transform will be randomised. Defaults to 0.5.
            add_blocks_chance (float, optional): The probability that blocks will be added back to the structure after masking. Defaults to 0.9.
            min_autoregressive_start_radius (int, optional): The minimum radius to use when starting the fake autoregressive inference. Defaults to 0.
            max_autoregressive_start_radius (int, optional): The maximum radius to use when starting the fake autoregressive inference. Defaults to 2.
            add_blocks_amount_first_dist_chance (float, optional): The probability that the first distribution will be used to determine the number of blocks to add back. Defaults to 0.5.
            add_blocks_amount_first_dist_amount (int, optional): The amount to use for the first distribution. Defaults to 50.
            add_blocks_amount_first_dist_beta (int, optional): The beta value to use for the first distribution. Defaults to 5.
            add_blocks_amount_second_dist_beta (int, optional): The beta value to use for the second distribution. Defaults to 3.
        """
        self._ignore_natural_blocks_chance = ignore_natural_blocks_chance
        self._mask_full_blocks_chance = mask_full_blocks_chance
        self._treat_family_as_single_block_chance = treat_family_as_single_block_chance
        self._percentage_block_types_to_mask = percentage_block_types_to_mask
        self._remove_block_type_chance = remove_block_type_chance
        self._min_mask_bias = min_mask_bias
        self._max_mask_bias = max_mask_bias
        self._invert_mask_chance = invert_mask_chance
        self._mask_origin_extra_area_percent = mask_origin_extra_area_percent
        self._flat_mask_chance = flat_mask_chance
        self._max_mask_edge_min_from_center = max_mask_edge_min_from_center
        self._mask_shell_chance = mask_shell_chance
        self._skip_block_transform_chance = skip_block_transform_chance
        self._random_block_transform_chance = random_block_transform_chance
        self._skip_block_property_transform_chance = (
            skip_block_property_transform_chance
        )
        self._random_block_property_transform_chance = (
            random_block_property_transform_chance
        )
        self._add_blocks_chance = add_blocks_chance
        self._min_autoregressive_start_radius = min_autoregressive_start_radius
        self._max_autoregressive_start_radius = max_autoregressive_start_radius
        self._add_blocks_amount_first_dist_chance = add_blocks_amount_first_dist_chance
        self._add_blocks_amount_first_dist_amount = add_blocks_amount_first_dist_amount
        self._add_blocks_amount_first_dist_beta = add_blocks_amount_first_dist_beta
        self._add_blocks_amount_second_dist_beta = add_blocks_amount_second_dist_beta

    def setup(self, block_token_converter: BlockTokenConverter):
        self._natural_block_tokens = torch.tensor(
            [
                block_token_converter.universal_str_to_token(block)
                for block in self.NATURAL_BLOCK_STRINGS
            ]
        )
        self._air_block_token = block_token_converter.universal_str_to_token(
            "universal_minecraft:air"
        )
        self._multi_block_tokens, self._multi_block_token_to_group_idx = (
            self._get_multi_block_tokens(block_token_converter)
        )
        self._block_transforms = self._get_block_transforms(block_token_converter)
        self._block_property_transforms = self._get_block_property_transforms(
            block_token_converter
        )

    @staticmethod
    def _build_token_set(
        block_token_converter: BlockTokenConverter,
        namespace: str,
        base_name: str,
        common_params: dict,
        varying_params: dict,
    ) -> set[int]:
        """
        Build a set of block tokens for all combinations in 'varying_params'.
        """
        param_keys = list(varying_params.keys())

        all_tokens = set()
        for combo in product(*(varying_params[k] for k in param_keys)):
            # Build up the block parameters
            params = dict(common_params)  # copy any common
            for key, val in zip(param_keys, combo):
                if val is not None:
                    params[key] = StringTag(val)
            # Construct the block, get its token
            block = Block(namespace, base_name, params)
            try:
                token = block_token_converter.universal_block_to_token(block)
            except KeyError:
                print(f"Block {block.blockstate} not found in mapping. Skipping.")
                continue
            all_tokens.add(token)

        return all_tokens

    @staticmethod
    def _build_token_tensor(
        block_token_converter: BlockTokenConverter,
        namespace: str,
        base_name: str,
        common_params: dict,
        varying_params: dict,
    ) -> torch.Tensor:
        """
        Build a tensor of block tokens for all combinations in 'varying_params'.
        """
        all_tokens = StructureMasker._build_token_set(
            block_token_converter, namespace, base_name, common_params, varying_params
        )
        return torch.tensor(list(all_tokens), dtype=torch.long)

    @staticmethod
    def _expand_block_family(
        block_token_converter: BlockTokenConverter,
        family_def: dict,
    ) -> set[torch.Tensor]:
        """
        Given a single family definition, expand it into one or more
        groups of block tokens. Returns a list of Tensors, each element
        in the list is a family group.
        """
        namespace = family_def.get("namespace", "universal_minecraft")
        base_name = family_def["base_name"]
        common_params = family_def.get("common_params", {})
        common_params = {k: StringTag(v) for k, v in common_params.items()}
        varying_params = family_def.get("varying_params", {})
        separate_params = family_def.get("separate_params", {})

        if separate_params:
            token_groups = set()
            param_keys = list(separate_params.keys())
            for combo in product(*(separate_params[k] for k in param_keys)):
                # Build up the block parameters
                params = dict(common_params)
                for key, val in zip(param_keys, combo):
                    params[key] = StringTag(val)
                token_groups.add(
                    StructureMasker._build_token_tensor(
                        block_token_converter,
                        namespace,
                        base_name,
                        params,
                        varying_params,
                    )
                )
            return token_groups
        else:
            return [
                StructureMasker._build_token_tensor(
                    block_token_converter,
                    namespace,
                    base_name,
                    common_params,
                    varying_params,
                )
            ]

    @staticmethod
    def _get_multi_block_tokens(
        block_token_converter: BlockTokenConverter,
    ) -> tuple[torch.Tensor, dict[int, int]]:
        multi_block_tokens = []

        for family in StructureMasker.BLOCK_FAMILIES:
            token_groups = StructureMasker._expand_block_family(
                block_token_converter, family
            )
            multi_block_tokens.extend(token_groups)

        # Create lookup dict: token -> group_idx in _multi_block_tokens
        token_to_group_idx = {
            token.item(): idx
            for idx, group in enumerate(multi_block_tokens)
            for token in group
        }

        return multi_block_tokens, token_to_group_idx

    def _get_extended_unique_blocks(
        self, unique_blocks: torch.Tensor
    ) -> list[torch.Tensor]:
        result = set(
            self._multi_block_tokens[self._multi_block_token_to_group_idx[block.item()]]
            if block.item() in self._multi_block_token_to_group_idx
            else torch.tensor([block])
            for block in unique_blocks
        )

        return list(result)

    @staticmethod
    def _expand_block_property_transform(
        block_token_converter: BlockTokenConverter,
        transform_def: dict,
    ) -> dict[int, int]:
        """
        Given a single block transform definition, expand it into a lookup table
        that maps input block tokens to output block tokens.
        """
        namespace = transform_def.get("namespace", "universal_minecraft")
        base_name = transform_def["base_name"]
        varying_params = transform_def.get("varying_params", {})
        param = transform_def["param"]
        input_val = transform_def.get("input", None)
        output_val = transform_def["output"]
        lut = torch.arange(block_token_converter.get_unused_token() + 1)

        param_keys = list(varying_params.keys())
        for combo in product(*(varying_params[k] for k in param_keys)):
            # Build up the block parameters
            params = {k: StringTag(v) for k, v in zip(param_keys, combo)}

            # Construct the blocks and get their tokens
            if input_val:
                params[param] = StringTag(input_val)
            input_block = Block(namespace, base_name, params)
            try:
                input_token = block_token_converter.universal_block_to_token(
                    input_block
                )
            except KeyError:
                # print(
                #     f"Input block {input_block.blockstate} not found in mapping. Skipping."
                # )
                continue

            params[param] = StringTag(output_val)
            output_block = Block(namespace, base_name, params)
            try:
                output_token = block_token_converter.universal_block_to_token(
                    output_block
                )
            except KeyError:
                # print(
                #     f"Output block {output_block.blockstate} not found in mapping. Skipping."
                # )
                continue

            # Update the lookup table
            lut[input_token] = output_token

        return lut

    @staticmethod
    def _get_block_transforms(
        block_token_converter: BlockTokenConverter,
    ) -> list[tuple[torch.Tensor, int, set[int]]]:
        result = []

        for transform in StructureMasker.BLOCK_TRANSFORMS:
            kernel = torch.tensor(transform["kernel"]).unsqueeze(0).unsqueeze(0)
            input_block_str = transform["input"]
            input_token = block_token_converter.universal_str_to_token(input_block_str)
            outputs = set()
            for family in transform["outputs"]:
                namespace = family.get("namespace", "universal_minecraft")
                base_name = family["base_name"]
                common_params = family.get("common_params", {})
                varying_params = family.get("varying_params", {})
                outputs.update(
                    StructureMasker._build_token_set(
                        block_token_converter,
                        namespace,
                        base_name,
                        common_params,
                        varying_params,
                    )
                )
            result.append((kernel, input_token, outputs))

        return result

    @staticmethod
    def _get_block_property_transforms(
        block_token_converter: BlockTokenConverter,
    ) -> list[tuple[torch.Tensor, dict]]:
        result = []

        for transform in StructureMasker.BLOCK_PROPERTY_TRANSFORMS:
            kernel = torch.tensor(transform["kernel"]).unsqueeze(0).unsqueeze(0)
            transform_def = transform["transform"]
            lut = StructureMasker._expand_block_property_transform(
                block_token_converter, transform_def
            )
            result.append((kernel, lut))

        return result

    @staticmethod
    def _create_point_noise(
        tensor: torch.Tensor,
        point: tuple[int, int, int],
        bias: float,
        min_radius: int,
        max_radius: int,
        flat: bool,
    ) -> torch.Tensor:
        side_length = tensor.shape[0]

        # Create a grid of coordinates
        d_coords, h_coords, w_coords = torch.meshgrid(
            torch.arange(side_length),
            torch.arange(side_length),
            torch.arange(side_length),
            indexing="ij",
        )

        # Calculate the distance from the point
        if flat:
            center = side_length // 2
            differences = [abs(coord - center) for coord in point]
            axis = differences.index(max(differences))
            distances = torch.abs((d_coords, h_coords, w_coords)[axis] - point[axis])
        else:
            distances = torch.sqrt(
                (d_coords - point[0]) ** 2
                + (h_coords - point[1]) ** 2
                + (w_coords - point[2]) ** 2
            )

        # Apply min and max radius
        mask = (distances >= min_radius) & (distances <= max_radius)

        # Normalize distances to range [0, 1] within the valid radius
        max_distance = min(
            max_radius,
            torch.sqrt(
                torch.tensor(
                    side_length**2 + side_length**2 + side_length**2,
                    dtype=torch.float32,
                )
            ),
        )
        normalized_distances = torch.where(
            mask, distances / max_distance, torch.ones_like(distances)
        )

        # Apply bias to the normalized distances
        if bias >= 0:
            biased_distances = normalized_distances**bias
        else:
            biased_distances = 1 - (1 - normalized_distances) ** abs(bias)

        # Calculate probabilities based on biased distances
        probabilities = 1 - biased_distances

        # Clamp probabilities to [0, 1] range
        probabilities = torch.clamp(probabilities, 0, 1)

        # Additional safeguard: replace any NaN or inf values with 0.5
        probabilities = torch.nan_to_num(probabilities, nan=0.5, posinf=0.5, neginf=0.5)

        # Generate noise based on the biased distances, only within the valid radius
        try:
            noise = torch.where(
                mask, torch.bernoulli(probabilities), torch.zeros_like(probabilities)
            ).bool()
        except RuntimeError as e:
            print(f"Error in torch.bernoulli: {e}")
            print(f"probabilities shape: {probabilities.shape}")
            print(f"probabilities unique values: {torch.unique(probabilities)}")
            raise

        # Set values less than the min radius to True if bias is negative, or greater than the max radius if bias is positive
        if bias < 0:
            noise[distances < min_radius] = True
        else:
            noise[distances > max_radius] = True

        return noise

    @staticmethod
    def _mask_operation(
        structure: torch.Tensor,
        position: torch.Tensor,
        block_types: torch.Tensor,
        bias: float,
        min_radius: int,
        max_radius: int,
        flat: bool,
    ) -> torch.Tensor:
        # Create noise around the point
        mask = StructureMasker._create_point_noise(
            structure, position, bias, min_radius, max_radius, flat
        )

        # Find the block types to mask
        if block_types is not None:
            mask &= torch.isin(structure, block_types)

        # Mask the structure
        structure[mask] = 0

        return structure

    def _mask_blocks(
        self, structure: torch.Tensor, block_types: torch.Tensor
    ) -> torch.Tensor:
        # print(f'block_types: {block_types}')

        # Compute bias
        bias = random.uniform(self._min_mask_bias, self._max_mask_bias)

        # Decide whether to take away a section of the structure or only keep a section
        if random.random() < self._invert_mask_chance:
            bias = -bias

        # Choose random position from an area slightly larger than the structure
        side_length = structure.shape[0]
        extra_area = int(side_length * self._mask_origin_extra_area_percent)
        min_position = -extra_area
        max_position = side_length + extra_area
        position = (
            random.randint(min_position, max_position),
            random.randint(min_position, max_position),
            random.randint(min_position, max_position),
        )

        # Choose a radius that will put the edge of the sphere right around the center
        center = side_length // 2
        flat = random.random() < self._flat_mask_chance
        if flat:
            distance_from_center = max(
                abs(position[0] - center),
                abs(position[1] - center),
                abs(position[2] - center),
            )
        else:
            distance_from_center = (
                (position[0] - center) ** 2
                + (position[1] - center) ** 2
                + (position[2] - center) ** 2
            ) ** 0.5
        max_radius = max(
            random.uniform(
                distance_from_center - self._max_mask_edge_min_from_center,
                distance_from_center + self._max_mask_edge_min_from_center,
            ),
            1,
        )

        # Either set the min radius to the max radius or choose a random value
        if random.random() < self._mask_shell_chance:
            min_radius = random.uniform(0, max_radius)
        else:
            min_radius = max_radius

        # Apply the mask to the structure
        masked_structure = StructureMasker._mask_operation(
            structure,
            position,
            block_types,
            bias,
            min_radius,
            max_radius,
            flat,
        )

        return masked_structure

    def _remove_blocks(self, structure: torch.Tensor) -> torch.Tensor:
        # Decide if we should ignore natural blocks
        if random.random() < self._ignore_natural_blocks_chance:
            # print('Ignoring natural blocks')
            ignore_blocks = self._natural_block_tokens
        else:
            # print('Not ignoring natural blocks')
            ignore_blocks = torch.tensor([])

        # Get unique blocks, excluding natural blocks and air
        unique_blocks = torch.tensor(
            [
                block.item()
                for block in torch.unique(structure)
                if block != self._air_block_token
                and not torch.isin(block, ignore_blocks)
            ]
        )

        # Apply for all blocks
        if random.random() < self._mask_full_blocks_chance:
            structure = self._mask_blocks(structure, unique_blocks)

        # Decide if we want to treat block families as a single block
        if random.random() < self._treat_family_as_single_block_chance:
            extended_unique_blocks = self._get_extended_unique_blocks(unique_blocks)
        else:
            extended_unique_blocks = [torch.tensor([block]) for block in unique_blocks]

        # Apply again for sampling of block types
        max_block_types = max(
            int(self._percentage_block_types_to_mask * len(extended_unique_blocks)), 1
        )
        num_block_types = min(
            random.randint(0, max_block_types), len(extended_unique_blocks)
        )
        for block_type in random.sample(extended_unique_blocks, num_block_types):
            # Decide whether to apply the mask or remove the block entirely
            if random.random() < self._remove_block_type_chance:
                structure[torch.isin(structure, block_type)] = 0
            else:
                structure = self._mask_blocks(structure, block_type)

        return structure

    def _fake_autoregressive_inference(
        self,
        structure: torch.Tensor,
        original_structure: torch.Tensor,
        start_radius: int,
        max_iterations: int,
        max_blocks: int,
    ) -> torch.Tensor:
        # Initialize tensor to track filled positions
        filled_positions = torch.zeros_like(structure, dtype=torch.bool)

        # Fill the center up to the start radius to pretend the model has already filled it
        z_mid, y_mid, x_mid = (
            structure.shape[0] // 2,
            structure.shape[1] // 2,
            structure.shape[2] // 2,
        )
        filled_positions[
            z_mid - start_radius : z_mid + start_radius + 1,
            y_mid - start_radius : y_mid + start_radius + 1,
            x_mid - start_radius : x_mid + start_radius + 1,
        ] = 1

        filled_blocks = 0

        # Each iteration adds on a "layer" of blocks
        for _ in range(max_iterations):
            # print(f"Iteration {iteration}/{max_iterations}")

            valid_positions = (
                TransformerMinecraftStructureGenerator._get_valid_positions(
                    structure, filled_positions
                )
            )
            if valid_positions is None:
                if filled_positions.all():
                    # print("No more elements to update")
                    break

                # Fill all remaining positions
                filled_positions[:] = True
                continue

            # Process each position
            for pos in valid_positions:
                z, y, x = pos
                predicted_token = original_structure[z, y, x]
                structure[z, y, x] = predicted_token
                filled_blocks += 1

                if predicted_token != self._air_block_token:
                    filled_positions[z, y, x] = 1
                    # print(f"Filled {filled_blocks}/{max_blocks} solid blocks")

                if filled_blocks >= max_blocks:
                    break

            if filled_blocks >= max_blocks:
                break

        return structure

    def _add_blocks(
        self, structure: torch.Tensor, original_structure: torch.Tensor
    ) -> torch.Tensor:
        # Find how many blocks that were removed from the original, excluding air blocks not adjacent to a solid block
        mask = TransformerMinecraftStructureGenerator.generate_neighbor_mask(
            original_structure.unsqueeze(0)
        ).squeeze(0)
        removed_blocks = (structure == 0) & (
            mask | (original_structure != self._air_block_token)
        )
        max_blocks = removed_blocks.sum().item()

        start_radius = random.randint(
            self._min_autoregressive_start_radius, self._max_autoregressive_start_radius
        )
        if np.random.rand() < self._add_blocks_amount_first_dist_chance:
            num_blocks = min(
                max_blocks,
                int(
                    self._add_blocks_amount_first_dist_amount
                    * np.random.beta(1, self._add_blocks_amount_first_dist_beta)
                ),
            )
        else:
            num_blocks = int(
                max_blocks * np.random.beta(1, self._add_blocks_amount_second_dist_beta)
            )
        return self._fake_autoregressive_inference(
            structure, original_structure, start_radius, 100, num_blocks
        )

    def _apply_block_transformations(self, structure: torch.Tensor) -> torch.Tensor:
        # Consider masked and air positions
        masked_positions = (
            ((structure == 0) | (structure == self._air_block_token))
            .long()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        for kernel, input_token, output_tokens in self._block_transforms:
            # Chance to skip the transformation
            if random.random() < self._skip_block_transform_chance:
                continue

            # Apply the convolution
            transform_mask = (
                F.conv3d(masked_positions, kernel, padding=1)
                .bool()
                .squeeze(0)
                .squeeze(0)
            )

            # Limit the mask to the input token
            transform_mask &= structure == input_token

            # Chance to only apply the transformation to a subset of the blocks
            if random.random() < self._random_block_transform_chance:
                mask_chance = random.uniform(0, 1)

                # Remove a random portion of the mask positions based on the chance
                transform_mask &= (
                    torch.rand_like(transform_mask, dtype=torch.float32) < mask_chance
                )

            # Replace the input tokens with the output token if it exists in the structure
            for output_token in output_tokens:
                if output_token in structure:
                    structure[transform_mask] = output_token

        # Consider only actual masked positions
        masked_positions = (structure == 0).long().unsqueeze(0).unsqueeze(0)

        for kernel, lut in self._block_property_transforms:
            # Chance to skip the transformation
            if random.random() < self._skip_block_property_transform_chance:
                continue

            # Apply the convolution
            transform_mask = (
                F.conv3d(masked_positions, kernel, padding=1)
                .bool()
                .squeeze(0)
                .squeeze(0)
            )

            # Chance to only apply the transformation to a subset of the blocks
            if random.random() < self._random_block_property_transform_chance:
                mask_chance = random.uniform(0, 1)

                # Remove a random portion of the mask positions based on the chance
                transform_mask &= (
                    torch.rand_like(transform_mask, dtype=torch.float32) < mask_chance
                )

            # Apply the lookup table
            structure = torch.where(transform_mask, lut[structure], structure)

        return structure

    def mask_structure(self, structure: torch.Tensor) -> torch.Tensor:
        masked_structure = structure.clone()

        # Remove blocks to represent a partially built structure
        masked_structure = self._remove_blocks(masked_structure)

        # Fill back in middle since infererence is triggered on a block position
        mid_d = structure.shape[0] // 2
        mid_h = structure.shape[1] // 2
        mid_w = structure.shape[2] // 2
        masked_structure[mid_d, mid_h, mid_w] = structure[mid_d, mid_h, mid_w]

        # Apply block transformations so the blocks will accurately represent how they should be with the other blocks removed
        masked_structure = self._apply_block_transformations(masked_structure)

        # Remove all air since this is what's done at inference time
        masked_structure[masked_structure == self._air_block_token] = 0

        # Add some blocks back to represent a partial stage in the auto-regressive inference
        if random.random() < self._add_blocks_chance:
            masked_structure = self._add_blocks(masked_structure, structure)

        return masked_structure
