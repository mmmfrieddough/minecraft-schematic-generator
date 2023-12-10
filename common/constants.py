REGION_SIZE = (8, 8, 8)

AIR = {
    'id': 'minecraft:air',
    'names': {
        'singular': ['air'],
        'plural': ['air']
    }
}

# Simple, solid blocks
SIMPLE_BLOCK_TYPES = [
    {
        'id': 'minecraft:stone',
        'names': {
            'singular': ['stone', 'regular stone', 'normal stone'],
            'plural': ['stone', 'regular stone', 'normal stone']
        }
    },
    {
        'id': 'minecraft:granite',
        'names': {
            'singular': ['granite', 'regular granite', 'normal granite'],
            'plural': ['granite', 'regular granite', 'normal granite']
        }
    },
    {
        'id': 'minecraft:diorite',
        'names': {
            'singular': ['diorite', 'regular diorite', 'normal diorite'],
            'plural': ['diorite', 'regular diorite', 'normal diorite']
        }
    },
    {
        'id': 'minecraft:polished_diorite',
        'names': {
            'singular': ['polished diorite'],
            'plural': ['polished diorite']
        }
    },
    {
        'id': 'minecraft:andesite',
        'names': {
            'singular': ['andesite', 'regular andesite', 'normal andesite'],
            'plural': ['andesite', 'regular andesite', 'normal andesite']
        }
    },
    {
        'id': 'minecraft:polished_andesite',
        'names': {
            'singular': ['polished andesite'],
            'plural': ['polished andesite']
        }
    },
    {
        'id': 'minecraft:dirt',
        'names': {
            'singular': ['dirt', 'regular dirt', 'normal dirt'],
            'plural': ['dirt', 'regular dirt', 'normal dirt']
        }
    },
    {
        'id': 'minecraft:coarse_dirt',
        'names': {
            'singular': ['coarse dirt'],
            'plural': ['coarse dirt']
        }
    },
    {
        'id': 'minecraft:cobblestone',
        'names': {
            'singular': ['cobblestone'],
            'plural': ['cobblestone']
        }
    },
    {
        'id': 'minecraft:oak_planks',
        'names': {
            'singular': ['oak plank'],
            'plural': ['oak planks', 'oak plank']
        }
    },
    {
        'id': 'minecraft:spruce_planks',
        'names': {
            'singular': ['spruce plank'],
            'plural': ['spruce planks', 'spruce plank']
        }
    },
    {
        'id': 'minecraft:birch_planks',
        'names': {
            'singular': ['birch plank'],
            'plural': ['birch planks', 'birch plank']
        }
    },
    {
        'id': 'minecraft:jungle_planks',
        'names': {
            'singular': ['jungle plank'],
            'plural': ['jungle planks', 'jungle plank']
        }
    },
    {
        'id': 'minecraft:acacia_planks',
        'names': {
            'singular': ['acacia plank'],
            'plural': ['acacia planks', 'acacia plank']
        }
    },
    {
        'id': 'minecraft:dark_oak_planks',
        'names': {
            'singular': ['dark oak plank'],
            'plural': ['dark oak planks', 'dark oak plank']
        }
    },
    {
        'id': 'minecraft:bedrock',
        'names': {
            'singular': ['bedrock'],
            'plural': ['bedrock']
        }
    },
    {
        'id': 'minecraft:sand',
        'names': {
            'singular': ['sand', 'regular sand', 'normal sand'],
            'plural': ['sand', 'regular sand', 'normal sand']
        }
    },
    {
        'id': 'minecraft:red_sand',
        'names': {
            'singular': ['red sand'],
            'plural': ['red sand']
        }
    },
    {
        'id': 'minecraft:gravel',
        'names': {
            'singular': ['gravel'],
            'plural': ['gravel']
        }
    },
    {
        'id': 'minecraft:gold_ore',
        'names': {
            'singular': ['gold ore'],
            'plural': ['gold ore']
        }
    },
    {
        'id': 'minecraft:iron_ore',
        'names': {
            'singular': ['iron ore'],
            'plural': ['iron ore']
        }
    },
    {
        'id': 'minecraft:coal_ore',
        'names': {
            'singular': ['coal ore'],
            'plural': ['coal ore']
        }
    },
    {
        'id': 'minecraft:sponge',
        'names': {
            'singular': ['sponge', 'dry sponge'],
            'plural': ['sponges', 'sponge', 'dry sponges', 'dry sponge']
        }
    },
    {
        'id': 'minecraft:wet_sponge',
        'names': {
            'singular': ['wet sponge', 'sponge'],
            'plural': ['wet sponges', 'wet sponge', 'sponges', 'sponge']
        }
    },
    {
        'id': 'minecraft:glass',
        'names': {
            'singular': ['glass', 'regular glass', 'normal glass', 'clear glass'],
            'plural': ['glass', 'regular glass', 'normal glass', 'clear glass']
        }
    },
    {
        'id': 'minecraft:lapis_ore',
        'names': {
            'singular': ['lapis lazuli ore'],
            'plural': ['lapis lazuli ore']
        }
    },
    {
        'id': 'minecraft:lapis_block',
        'names': {
            'singular': ['lapis lazuli block'],
            'plural': ['lapis lazuli blocks', 'lapis lazuli block']
        }
    },
    {
        'id': 'minecraft:sandstone',
        'names': {
            'singular': ['sandstone', 'normal sandstone', 'regular sandstone'],
            'plural': ['sandstone', 'normal sandstone', 'regular sandstone']
        }
    },
    {
        'id': 'minecraft:chiseled_sandstone',
        'names': {
            'singular': ['chiseled sandstone'],
            'plural': ['chiseled sandstone']
        }
    },
    {
        'id': 'minecraft:cut_sandstone',
        'names': {
            'singular': ['cut sandstone'],
            'plural': ['cut sandstone']
        }
    },
    {
        'id': 'minecraft:note_block',
        'names': {
            'singular': ['note block'],
            'plural': ['note blocks', 'note block']
        }
    },
    {
        'id': 'minecraft:gold_block',
        'names': {
            'singular': ['gold block'],
            'plural': ['gold blocks', 'gold block']
        }
    },
    {
        'id': 'minecraft:iron_block',
        'names': {
            'singular': ['iron block'],
            'plural': ['iron blocks', 'iron block']
        }
    },
    {
        'id': 'minecraft:smooth_quartz',
        'names': {
            'singular': ['smooth quartz'],
            'plural': ['smooth quartz']
        }
    },
    {
        'id': 'minecraft:smooth_red_sandstone',
        'names': {
            'singular': ['smooth red sandstone'],
            'plural': ['smooth red sandstone']
        }
    },
    {
        'id': 'minecraft:smooth_sandstone',
        'names': {
            'singular': ['smooth sandstone'],
            'plural': ['smooth sandstone']
        }
    },
    {
        'id': 'minecraft:smooth_stone',
        'names': {
            'singular': ['smooth stone'],
            'plural': ['smooth stone']
        }
    },
    {
        'id': 'minecraft:bricks',
        'names': {
            'singular': ['brick'],
            'plural': ['bricks', 'brick']
        }
    },
    {
        'id': 'minecraft:tnt',
        'names': {
            'singular': ['TNT'],
            'plural': ['TNT']
        }
    },
    {
        'id': 'minecraft:bookshelf',
        'names': {
            'singular': ['bookshelf'],
            'plural': ['bookshelves', 'bookshelf']
        }
    },
    {
        'id': 'minecraft:mossy_cobblestone',
        'names': {
            'singular': ['mossy cobblestone'],
            'plural': ['mossy cobblestone']
        }
    },
    {
        'id': 'minecraft:obsidian',
        'names': {
            'singular': ['obsidian'],
            'plural': ['obsidian']
        }
    },
    {
        'id': 'minecraft:purpur_block',
        'names': {
            'singular': ['purpur block'],
            'plural': ['purpur blocks', 'purpur block']
        }
    },
    {
        'id': 'minecraft:diamond_ore',
        'names': {
            'singular': ['diamond ore'],
            'plural': ['diamond ore']
        }
    },
    {
        'id': 'minecraft:diamond_block',
        'names': {
            'singular': ['diamond block'],
            'plural': ['diamond blocks', 'diamond block']
        }
    },
    {
        'id': 'minecraft:crafting_table',
        'names': {
            'singular': ['crafting table'],
            'plural': ['crafting tables', 'crafting table']
        }
    },
    {
        'id': 'minecraft:ice',
        'names': {
            'singular': ['ice'],
            'plural': ['ice']
        }
    },
    {
        'id': 'minecraft:snow_block',
        'names': {
            'singular': ['snow block'],
            'plural': ['snow blocks', 'snow block']
        }
    },
    {
        'id': 'minecraft:clay',
        'names': {
            'singular': ['clay'],
            'plural': ['clay']
        }
    },
    {
        'id': 'minecraft:pumpkin',
        'names': {
            'singular': ['pumpkin'],
            'plural': ['pumpkins', 'pumpkin']
        }
    },
    {
        'id': 'minecraft:netherrack',
        'names': {
            'singular': ['netherrack'],
            'plural': ['netherrack']
        }
    },
    {
        'id': 'minecraft:soul_sand',
        'names': {
            'singular': ['soul sand'],
            'plural': ['soul sand']
        }
    },
    {
        'id': 'minecraft:glowstone',
        'names': {
            'singular': ['glowstone'],
            'plural': ['glowstone']
        }
    },
    {
        'id': 'minecraft:infested_stone',
        'names': {
            'singular': ['infested stone', 'silverfish stone'],
            'plural': ['infested stone', 'silverfish stone']
        }
    },
    {
        'id': 'minecraft:infested_cobblestone',
        'names': {
            'singular': ['infested cobblestone', 'silverfish cobblestone'],
            'plural': ['infested cobblestone', 'silverfish cobblestone']
        }
    },
    {
        'id': 'minecraft:infested_stone_bricks',
        'names': {
            'singular': ['infested stone bricks', 'silverfish stone bricks'],
            'plural': ['infested stone bricks', 'silverfish stone bricks']
        }
    },
    {
        'id': 'minecraft:infested_mossy_stone_bricks',
        'names': {
            'singular': ['infested mossy stone bricks', 'silverfish mossy stone bricks'],
            'plural': ['infested mossy stone bricks', 'silverfish mossy stone bricks']
        }
    },
    {
        'id': 'minecraft:infested_cracked_stone_bricks',
        'names': {
            'singular': ['infested cracked stone bricks', 'silverfish cracked stone bricks'],
            'plural': ['infested cracked stone bricks', 'silverfish cracked stone bricks']
        }
    },
    {
        'id': 'minecraft:infested_chiseled_stone_bricks',
        'names': {
            'singular': ['infested chiseled stone bricks', 'silverfish chiseled stone bricks'],
            'plural': ['infested chiseled stone bricks', 'silverfish chiseled stone bricks']
        }
    },
    {
        'id': 'minecraft:stone_bricks',
        'names': {
            'singular': ['stone brick'],
            'plural': ['stone bricks', 'stone brick']
        }
    },
    {
        'id': 'minecraft:mossy_stone_bricks',
        'names': {
            'singular': ['mossy stone brick'],
            'plural': ['mossy stone bricks', 'mossy stone brick']
        }
    },
    {
        'id': 'minecraft:cracked_stone_bricks',
        'names': {
            'singular': ['cracked stone brick'],
            'plural': ['cracked stone bricks', 'cracked stone brick']
        }
    },
    {
        'id': 'minecraft:chiseled_stone_bricks',
        'names': {
            'singular': ['chiseled stone brick'],
            'plural': ['chiseled stone bricks', 'chiseled stone brick']
        }
    },
    {
        'id': 'minecraft:melon',
        'names': {
            'singular': ['melon'],
            'plural': ['melons', 'melon']
        }
    },
    {
        'id': 'minecraft:nether_bricks',
        'names': {
            'singular': ['nether brick'],
            'plural': ['nether bricks', 'nether brick']
        }
    },
    {
        'id': 'minecraft:end_stone',
        'names': {
            'singular': ['end stone'],
            'plural': ['end stone']
        }
    },
    {
        'id': 'minecraft:end_stone_bricks',
        'names': {
            'singular': ['end stone brick'],
            'plural': ['end stone bricks', 'end stone brick']
        }
    },
    {
        'id': 'minecraft:emerald_ore',
        'names': {
            'singular': ['emerald ore'],
            'plural': ['emerald ore']
        }
    },
    {
        'id': 'minecraft:emerald_block',
        'names': {
            'singular': ['emerald block'],
            'plural': ['emerald blocks', 'emerald block']
        }
    },
    {
        'id': 'minecraft:redstone_block',
        'names': {
            'singular': ['redstone block'],
            'plural': ['redstone blocks', 'redstone block']
        }
    },
    {
        'id': 'minecraft:nether_quartz_ore',
        'names': {
            'singular': ['nether quartz ore'],
            'plural': ['nether quartz ore']
        }
    },
    {
        'id': 'minecraft:chiseled_quartz_block',
        'names': {
            'singular': ['chiseled quartz block'],
            'plural': ['chiseled quartz blocks', 'chiseled quartz block']
        }
    },
    {
        'id': 'minecraft:quartz_block',
        'names': {
            'singular': ['quartz block'],
            'plural': ['quartz blocks', 'quartz block']
        }
    },
    {
        'id': 'minecraft:white_terracotta',
        'names': {
            'singular': ['white terracotta'],
            'plural': ['white terracotta']
        }
    },
    {
        'id': 'minecraft:orange_terracotta',
        'names': {
            'singular': ['orange terracotta'],
            'plural': ['orange terracotta']
        }
    },
    {
        'id': 'minecraft:magenta_terracotta',
        'names': {
            'singular': ['magenta terracotta'],
            'plural': ['magenta terracotta']
        }
    },
    {
        'id': 'minecraft:light_blue_terracotta',
        'names': {
            'singular': ['light blue terracotta'],
            'plural': ['light blue terracotta']
        }
    },
    {
        'id': 'minecraft:yellow_terracotta',
        'names': {
            'singular': ['yellow terracotta'],
            'plural': ['yellow terracotta']
        }
    },
    {
        'id': 'minecraft:lime_terracotta',
        'names': {
            'singular': ['lime terracotta'],
            'plural': ['lime terracotta']
        }
    },
    {
        'id': 'minecraft:pink_terracotta',
        'names': {
            'singular': ['pink terracotta'],
            'plural': ['pink terracotta']
        }
    },
    {
        'id': 'minecraft:gray_terracotta',
        'names': {
            'singular': ['gray terracotta'],
            'plural': ['gray terracotta']
        }
    },
    {
        'id': 'minecraft:light_gray_terracotta',
        'names': {
            'singular': ['light gray terracotta'],
            'plural': ['light gray terracotta']
        }
    },
    {
        'id': 'minecraft:cyan_terracotta',
        'names': {
            'singular': ['cyan terracotta'],
            'plural': ['cyan terracotta']
        }
    },
    {
        'id': 'minecraft:purple_terracotta',
        'names': {
            'singular': ['purple terracotta'],
            'plural': ['purple terracotta']
        }
    },
    {
        'id': 'minecraft:blue_terracotta',
        'names': {
            'singular': ['blue terracotta'],
            'plural': ['blue terracotta']
        }
    },
    {
        'id': 'minecraft:brown_terracotta',
        'names': {
            'singular': ['brown terracotta'],
            'plural': ['brown terracotta']
        }
    },
    {
        'id': 'minecraft:green_terracotta',
        'names': {
            'singular': ['green terracotta'],
            'plural': ['green terracotta']
        }
    },
    {
        'id': 'minecraft:red_terracotta',
        'names': {
            'singular': ['red terracotta'],
            'plural': ['red terracotta']
        }
    },
    {
        'id': 'minecraft:black_terracotta',
        'names': {
            'singular': ['black terracotta'],
            'plural': ['black terracotta']
        }
    },
    {
        'id': 'minecraft:terracotta',
        'names': {
            'singular': ['terracotta', 'regular terracotta', 'normal terracotta', 'plain terracotta'],
            'plural': ['terracotta', 'regular terracotta', 'normal terracotta', 'plain terracotta']
        }
    },
    {
        'id': 'minecraft:coal_block',
        'names': {
            'singular': ['coal block'],
            'plural': ['coal blocks', 'coal block']
        }
    },
    {
        'id': 'minecraft:packed_ice',
        'names': {
            'singular': ['packed ice'],
            'plural': ['packed ice']
        }
    },
    {
        'id': 'minecraft:slime_block',
        'names': {
            'singular': ['slime block'],
            'plural': ['slime blocks', 'slime block']
        }
    },
    {
        'id': 'minecraft:white_stained_glass',
        'names': {
            'singular': ['white stained glass'],
            'plural': ['white stained glass']
        }
    },
    {
        'id': 'minecraft:orange_stained_glass',
        'names': {
            'singular': ['orange stained glass'],
            'plural': ['orange stained glass']
        }
    },
    {
        'id': 'minecraft:magenta_stained_glass',
        'names': {
            'singular': ['magenta stained glass'],
            'plural': ['magenta stained glass']
        }
    },
    {
        'id': 'minecraft:light_blue_stained_glass',
        'names': {
            'singular': ['light blue stained glass'],
            'plural': ['light blue stained glass']
        }
    },
    {
        'id': 'minecraft:yellow_stained_glass',
        'names': {
            'singular': ['yellow stained glass'],
            'plural': ['yellow stained glass']
        }
    },
    {
        'id': 'minecraft:lime_stained_glass',
        'names': {
            'singular': ['lime stained glass'],
            'plural': ['lime stained glass']
        }
    },
    {
        'id': 'minecraft:pink_stained_glass',
        'names': {
            'singular': ['pink stained glass'],
            'plural': ['pink stained glass']
        }
    },
    {
        'id': 'minecraft:gray_stained_glass',
        'names': {
            'singular': ['gray stained glass'],
            'plural': ['gray stained glass']
        }
    },
    {
        'id': 'minecraft:light_gray_stained_glass',
        'names': {
            'singular': ['light gray stained glass'],
            'plural': ['light gray stained glass']
        }
    },
    {
        'id': 'minecraft:cyan_stained_glass',
        'names': {
            'singular': ['cyan stained glass'],
            'plural': ['cyan stained glass']
        }
    },
    {
        'id': 'minecraft:purple_stained_glass',
        'names': {
            'singular': ['purple stained glass'],
            'plural': ['purple stained glass']
        }
    },
    {
        'id': 'minecraft:blue_stained_glass',
        'names': {
            'singular': ['blue stained glass'],
            'plural': ['blue stained glass']
        }
    },
    {
        'id': 'minecraft:brown_stained_glass',
        'names': {
            'singular': ['brown stained glass'],
            'plural': ['brown stained glass']
        }
    },
    {
        'id': 'minecraft:green_stained_glass',
        'names': {
            'singular': ['green stained glass'],
            'plural': ['green stained glass']
        }
    },
    {
        'id': 'minecraft:red_stained_glass',
        'names': {
            'singular': ['red stained glass'],
            'plural': ['red stained glass']
        }
    },
    {
        'id': 'minecraft:black_stained_glass',
        'names': {
            'singular': ['black stained glass'],
            'plural': ['black stained glass']
        }
    },
    {
        'id': 'minecraft:prismarine',
        'names': {
            'singular': ['prismarine'],
            'plural': ['prismarine']
        }
    },
    {
        'id': 'minecraft:prismarine_bricks',
        'names': {
            'singular': ['prismarine brick'],
            'plural': ['prismarine bricks', 'prismarine brick']
        }
    },
    {
        'id': 'minecraft:dark_prismarine',
        'names': {
            'singular': ['dark prismarine'],
            'plural': ['dark prismarine']
        }
    },
    {
        'id': 'minecraft:sea_lantern',
        'names': {
            'singular': ['sea lantern'],
            'plural': ['sea lanterns', 'sea lantern']
        }
    },
    {
        'id': 'minecraft:red_sandstone',
        'names': {
            'singular': ['red sandstone'],
            'plural': ['red sandstone']
        }
    },
    {
        'id': 'minecraft:chiseled_red_sandstone',
        'names': {
            'singular': ['chiseled red sandstone'],
            'plural': ['chiseled red sandstone']
        }
    },
    {
        'id': 'minecraft:cut_red_sandstone',
        'names': {
            'singular': ['cut red sandstone'],
            'plural': ['cut red sandstone']
        }
    },
    {
        'id': 'minecraft:magma_block',
        'names': {
            'singular': ['magma block'],
            'plural': ['magma blocks', 'magma block']
        }
    },
    {
        'id': 'minecraft:nether_wart_block',
        'names': {
            'singular': ['nether wart block'],
            'plural': ['nether wart blocks', 'nether wart block']
        }
    },
    {
        'id': 'minecraft:red_nether_bricks',
        'names': {
            'singular': ['red nether brick'],
            'plural': ['red nether bricks', 'red nether brick']
        }
    },
    {
        'id': 'minecraft:white_concrete',
        'names': {
            'singular': ['white concrete'],
            'plural': ['white concrete']
        }
    },
    {
        'id': 'minecraft:orange_concrete',
        'names': {
            'singular': ['orange concrete'],
            'plural': ['orange concrete']
        }
    },
    {
        'id': 'minecraft:magenta_concrete',
        'names': {
            'singular': ['magenta concrete'],
            'plural': ['magenta concrete']
        }
    },
    {
        'id': 'minecraft:light_blue_concrete',
        'names': {
            'singular': ['light blue concrete'],
            'plural': ['light blue concrete']
        }
    },
    {
        'id': 'minecraft:yellow_concrete',
        'names': {
            'singular': ['yellow concrete'],
            'plural': ['yellow concrete']
        }
    },
    {
        'id': 'minecraft:lime_concrete',
        'names': {
            'singular': ['lime concrete'],
            'plural': ['lime concrete']
        }
    },
    {
        'id': 'minecraft:pink_concrete',
        'names': {
            'singular': ['pink concrete'],
            'plural': ['pink concrete']
        }
    },
    {
        'id': 'minecraft:gray_concrete',
        'names': {
            'singular': ['gray concrete'],
            'plural': ['gray concrete']
        }
    },
    {
        'id': 'minecraft:light_gray_concrete',
        'names': {
            'singular': ['light gray concrete'],
            'plural': ['light gray concrete']
        }
    },
    {
        'id': 'minecraft:cyan_concrete',
        'names': {
            'singular': ['cyan concrete'],
            'plural': ['cyan concrete']
        }
    },
    {
        'id': 'minecraft:purple_concrete',
        'names': {
            'singular': ['purple concrete'],
            'plural': ['purple concrete']
        }
    },
    {
        'id': 'minecraft:blue_concrete',
        'names': {
            'singular': ['blue concrete'],
            'plural': ['blue concrete']
        }
    },
    {
        'id': 'minecraft:brown_concrete',
        'names': {
            'singular': ['brown concrete'],
            'plural': ['brown concrete']
        }
    },
    {
        'id': 'minecraft:green_concrete',
        'names': {
            'singular': ['green concrete'],
            'plural': ['green concrete']
        }
    },
    {
        'id': 'minecraft:red_concrete',
        'names': {
            'singular': ['red concrete'],
            'plural': ['red concrete']
        }
    },
    {
        'id': 'minecraft:black_concrete',
        'names': {
            'singular': ['black concrete'],
            'plural': ['black concrete']
        }
    },
    {
        'id': 'minecraft:white_concrete_powder',
        'names': {
            'singular': ['white concrete powder'],
            'plural': ['white concrete powder']
        }
    },
    {
        'id': 'minecraft:orange_concrete_powder',
        'names': {
            'singular': ['orange concrete powder'],
            'plural': ['orange concrete powder']
        }
    },
    {
        'id': 'minecraft:magenta_concrete_powder',
        'names': {
            'singular': ['magenta concrete powder'],
            'plural': ['magenta concrete powder']
        }
    },
    {
        'id': 'minecraft:light_blue_concrete_powder',
        'names': {
            'singular': ['light blue concrete powder'],
            'plural': ['light blue concrete powder']
        }
    },
    {
        'id': 'minecraft:yellow_concrete_powder',
        'names': {
            'singular': ['yellow concrete powder'],
            'plural': ['yellow concrete powder']
        }
    },
    {
        'id': 'minecraft:lime_concrete_powder',
        'names': {
            'singular': ['lime concrete powder'],
            'plural': ['lime concrete powder']
        }
    },
    {
        'id': 'minecraft:pink_concrete_powder',
        'names': {
            'singular': ['pink concrete powder'],
            'plural': ['pink concrete powder']
        }
    },
    {
        'id': 'minecraft:gray_concrete_powder',
        'names': {
            'singular': ['gray concrete powder'],
            'plural': ['gray concrete powder']
        }
    },
    {
        'id': 'minecraft:light_gray_concrete_powder',
        'names': {
            'singular': ['light gray concrete powder'],
            'plural': ['light gray concrete powder']
        }
    },
    {
        'id': 'minecraft:cyan_concrete_powder',
        'names': {
            'singular': ['cyan concrete powder'],
            'plural': ['cyan concrete powder']
        }
    },
    {
        'id': 'minecraft:purple_concrete_powder',
        'names': {
            'singular': ['purple concrete powder'],
            'plural': ['purple concrete powder']
        }
    },
    {
        'id': 'minecraft:blue_concrete_powder',
        'names': {
            'singular': ['blue concrete powder'],
            'plural': ['blue concrete powder']
        }
    },
    {
        'id': 'minecraft:brown_concrete_powder',
        'names': {
            'singular': ['brown concrete powder'],
            'plural': ['brown concrete powder']
        }
    },
    {
        'id': 'minecraft:green_concrete_powder',
        'names': {
            'singular': ['green concrete powder'],
            'plural': ['green concrete powder']
        }
    },
    {
        'id': 'minecraft:red_concrete_powder',
        'names': {
            'singular': ['red concrete powder'],
            'plural': ['red concrete powder']
        }
    },
    {
        'id': 'minecraft:black_concrete_powder',
        'names': {
            'singular': ['black concrete powder'],
            'plural': ['black concrete powder']
        }
    },
    {
        'id': 'minecraft:dead_tube_coral_block',
        'names': {
            'singular': ['dead tube coral block'],
            'plural': ['dead tube coral blocks']
        }
    },
    {
        'id': 'minecraft:dead_brain_coral_block',
        'names': {
            'singular': ['dead brain coral block'],
            'plural': ['dead brain coral blocks']
        }
    },
    {
        'id': 'minecraft:dead_bubble_coral_block',
        'names': {
            'singular': ['dead bubble coral block'],
            'plural': ['dead bubble coral blocks']
        }
    },
    {
        'id': 'minecraft:dead_fire_coral_block',
        'names': {
            'singular': ['dead fire coral block'],
            'plural': ['dead fire coral blocks']
        }
    },
    {
        'id': 'minecraft:dead_horn_coral_block',
        'names': {
            'singular': ['dead horn coral block'],
            'plural': ['dead horn coral blocks']
        }
    },
    {
        'id': 'minecraft:tube_coral_block',
        'names': {
            'singular': ['tube coral block'],
            'plural': ['tube coral blocks']
        }
    },
    {
        'id': 'minecraft:brain_coral_block',
        'names': {
            'singular': ['brain coral block'],
            'plural': ['brain coral blocks']
        }
    },
    {
        'id': 'minecraft:bubble_coral_block',
        'names': {
            'singular': ['bubble coral block'],
            'plural': ['bubble coral blocks']
        }
    },
    {
        'id': 'minecraft:fire_coral_block',
        'names': {
            'singular': ['fire coral block'],
            'plural': ['fire coral blocks']
        }
    },
    {
        'id': 'minecraft:horn_coral_block',
        'names': {
            'singular': ['horn coral block'],
            'plural': ['horn coral blocks']
        }
    },
    {
        'id': 'minecraft:blue_ice',
        'names': {
            'singular': ['blue ice'],
            'plural': ['blue ice']
        }
    },
    {
        'id': 'minecraft:honey_block',
        'names': {
            'singular': ['honey block'],
            'plural': ['honey blocks']
        }
    },
    {
        'id': 'minecraft:honeycomb_block',
        'names': {
            'singular': ['honeycomb block'],
            'plural': ['honeycomb blocks']
        }
    },
    {
        'id': 'minecraft:crimson_planks',
        'names': {
            'singular': ['crimson planks'],
            'plural': ['crimson planks']
        }
    },
    {
        'id': 'minecraft:warped_planks',
        'names': {
            'singular': ['warped planks'],
            'plural': ['warped planks']
        }
    },
    {
        'id': 'minecraft:nether_gold_ore',
        'names': {
            'singular': ['nether gold ore'],
            'plural': ['nether gold ores']
        }
    },
    {
        'id': 'minecraft:soul_soil',
        'names': {
            'singular': ['soul soil'],
            'plural': ['soul soil']
        }
    },
    {
        'id': 'minecraft:cracked_nether_bricks',
        'names': {
            'singular': ['cracked nether brick'],
            'plural': ['cracked nether bricks', 'cracked nether brick']
        }
    },
    {
        'id': 'minecraft:chiseled_nether_bricks',
        'names': {
            'singular': ['chiseled nether brick'],
            'plural': ['chiseled nether bricks', 'chiseled nether brick']
        }
    },
    {
        'id': 'minecraft:quartz_bricks',
        'names': {
            'singular': ['quartz brick'],
            'plural': ['quartz bricks', 'quartz brick']
        }
    },
    {
        'id': 'minecraft:warped_wart_block',
        'names': {
            'singular': ['warped wart block'],
            'plural': ['warped wart blocks', 'warped wart block']
        }
    },
    {
        'id': 'minecraft:shroomlight',
        'names': {
            'singular': ['shroomlight'],
            'plural': ['shroomlights', 'shroomlight']
        }
    },
    {
        'id': 'minecraft:netherite_block',
        'names': {
            'singular': ['netherite block'],
            'plural': ['netherite blocks', 'netherite block']
        }
    },
    {
        'id': 'minecraft:ancient_debris',
        'names': {
            'singular': ['ancient debris'],
            'plural': ['ancient debris']
        }
    },
    {
        'id': 'minecraft:target',
        'names': {
            'singular': ['target block'],
            'plural': ['target blocks', 'target block']
        }
    },
    {
        'id': 'minecraft:crying_obsidian',
        'names': {
            'singular': ['crying obsidian'],
            'plural': ['crying obsidian']
        }
    },
    {
        'id': 'minecraft:blackstone',
        'names': {
            'singular': ['blackstone', 'regular blackstone', 'normal blackstone'],
            'plural': ['blackstone', 'regular blackstone', 'normal blackstone']
        }
    },
    {
        'id': 'minecraft:gilded_blackstone',
        'names': {
            'singular': ['gilded blackstone'],
            'plural': ['gilded blackstone']
        }
    },
    {
        'id': 'minecraft:polished_blackstone',
        'names': {
            'singular': ['polished blackstone'],
            'plural': ['polished blackstone']
        }
    },
    {
        'id': 'minecraft:chiseled_polished_blackstone',
        'names': {
            'singular': ['chiseled polished blackstone'],
            'plural': ['chiseled polished blackstone']
        }
    },
    {
        'id': 'minecraft:polished_blackstone_bricks',
        'names': {
            'singular': ['polished blackstone bricks'],
            'plural': ['polished blackstone bricks']
        }
    },
    {
        'id': 'minecraft:amethyst_block',
        'names': {
            'singular': ['amethyst block'],
            'plural': ['amethyst blocks', 'amethyst block']
        }
    },
    {
        'id': 'minecraft:calcite',
        'names': {
            'singular': ['calcite'],
            'plural': ['calcite']
        }
    },
    {
        'id': 'minecraft:chiseled_deepslate',
        'names': {
            'singular': ['chiseled deepslate'],
            'plural': ['chiseled deepslate']
        }
    },
    {
        'id': 'minecraft:cobbled_deepslate',
        'names': {
            'singular': ['cobbled deepslate'],
            'plural': ['cobbled deepslate']
        }
    },
    {
        'id': 'minecraft:copper_block',
        'names': {
            'singular': ['copper block'],
            'plural': ['copper blocks', 'copper block']
        }
    },
    {
        'id': 'minecraft:copper_ore',
        'names': {
            'singular': ['copper ore'],
            'plural': ['copper ore']
        }
    },
    {
        'id': 'minecraft:cracked_deepslate_bricks',
        'names': {
            'singular': ['cracked deepslate brick'],
            'plural': ['cracked deepslate bricks', 'cracked deepslate brick']
        }
    },
    {
        'id': 'minecraft:cracked_deepslate_tiles',
        'names': {
            'singular': ['cracked deepslate tile'],
            'plural': ['cracked deepslate tiles', 'cracked deepslate tile']
        }
    },
    {
        'id': 'minecraft:cut_copper',
        'names': {
            'singular': ['cut copper'],
            'plural': ['cut copper']
        }
    },
    {
        'id': 'minecraft:deepslate_bricks',
        'names': {
            'singular': ['deepslate brick'],
            'plural': ['deepslate bricks', 'deepslate brick']
        }
    },
    {
        'id': 'minecraft:deepslate_coal_ore',
        'names': {
            'singular': ['deepslate coal ore'],
            'plural': ['deepslate coal ore']
        }
    },
    {
        'id': 'minecraft:deepslate_copper_ore',
        'names': {
            'singular': ['deepslate copper ore'],
            'plural': ['deepslate copper ore']
        }
    },
    {
        'id': 'minecraft:deepslate_diamond_ore',
        'names': {
            'singular': ['deepslate diamond ore'],
            'plural': ['deepslate diamond ore']
        }
    },
    {
        'id': 'minecraft:deepslate_emerald_ore',
        'names': {
            'singular': ['deepslate emerald ore'],
            'plural': ['deepslate emerald ore']
        }
    },
    {
        'id': 'minecraft:deepslate_gold_ore',
        'names': {
            'singular': ['deepslate gold ore'],
            'plural': ['deepslate gold ore']
        }
    },
    {
        'id': 'minecraft:deepslate_iron_ore',
        'names': {
            'singular': ['deepslate iron ore'],
            'plural': ['deepslate iron ore']
        }
    },
    {
        'id': 'minecraft:deepslate_lapis_ore',
        'names': {
            'singular': ['deepslate lapis ore'],
            'plural': ['deepslate lapis ore']
        }
    },
    {
        'id': 'minecraft:deepslate_tiles',
        'names': {
            'singular': ['deepslate tile'],
            'plural': ['deepslate tiles', 'deepslate tile']
        }
    },
    {
        'id': 'minecraft:dripstone_block',
        'names': {
            'singular': ['dripstone block'],
            'plural': ['dripstone blocks', 'dripstone block']
        }
    },
    {
        'id': 'minecraft:exposed_copper',
        'names': {
            'singular': ['exposed copper'],
            'plural': ['exposed copper']
        }
    },
    {
        'id': 'minecraft:exposed_cut_copper',
        'names': {
            'singular': ['exposed cut copper'],
            'plural': ['exposed cut copper']
        }
    },
    {
        'id': 'minecraft:moss_block',
        'names': {
            'singular': ['moss block'],
            'plural': ['moss blocks', 'moss block']
        }
    },
    {
        'id': 'minecraft:oxidized_copper',
        'names': {
            'singular': ['oxidized copper'],
            'plural': ['oxidized copper']
        }
    },
    {
        'id': 'minecraft:polished_deepslate',
        'names': {
            'singular': ['polished deepslate'],
            'plural': ['polished deepslate']
        }
    },
    {
        'id': 'minecraft:raw_copper_block',
        'names': {
            'singular': ['raw copper block'],
            'plural': ['raw copper blocks', 'raw copper block']
        }
    },
    {
        'id': 'minecraft:raw_gold_block',
        'names': {
            'singular': ['raw gold block'],
            'plural': ['raw gold blocks', 'raw gold block']
        }
    },
    {
        'id': 'minecraft:raw_iron_block',
        'names': {
            'singular': ['raw iron block'],
            'plural': ['raw iron blocks', 'raw iron block']
        }
    },
    {
        'id': 'minecraft:rooted_dirt',
        'names': {
            'singular': ['rooted dirt'],
            'plural': ['rooted dirt']
        }
    },
    {
        'id': 'minecraft:smooth_basalt',
        'names': {
            'singular': ['smooth basalt'],
            'plural': ['smooth basalt']
        }
    },
    {
        'id': 'minecraft:tinted_glass',
        'names': {
            'singular': ['tinted glass'],
            'plural': ['tinted glass']
        }
    },
    {
        'id': 'minecraft:tuff',
        'names': {
            'singular': ['tuff'],
            'plural': ['tuff']
        }
    },
    {
        'id': 'minecraft:waxed_copper_block',
        'names': {
            'singular': ['waxed copper block'],
            'plural': ['waxed copper blocks', 'waxed copper block']
        }
    },
    {
        'id': 'minecraft:waxed_cut_copper',
        'names': {
            'singular': ['waxed cut copper'],
            'plural': ['waxed cut copper']
        }
    },
    {
        'id': 'minecraft:waxed_exposed_copper',
        'names': {
            'singular': ['waxed exposed copper'],
            'plural': ['waxed exposed copper']
        }
    },
    {
        'id': 'minecraft:waxed_exposed_cut_copper',
        'names': {
            'singular': ['waxed exposed cut copper'],
            'plural': ['waxed exposed cut copper']
        }
    },
    {
        'id': 'minecraft:waxed_oxidized_copper',
        'names': {
            'singular': ['waxed oxidized copper'],
            'plural': ['waxed oxidized copper']
        }
    },
    {
        'id': 'minecraft:waxed_weathered_copper',
        'names': {
            'singular': ['waxed weathered copper'],
            'plural': ['waxed weathered copper']
        }
    },
    {
        'id': 'minecraft:waxed_weathered_cut_copper',
        'names': {
            'singular': ['waxed weathered cut copper'],
            'plural': ['waxed weathered cut copper']
        }
    },
    {
        'id': 'minecraft:weathered_copper',
        'names': {
            'singular': ['weathered copper'],
            'plural': ['weathered copper']
        }
    },
    {
        'id': 'minecraft:weathered_cut_copper',
        'names': {
            'singular': ['weathered cut copper'],
            'plural': ['weathered cut copper']
        }
    },
    {
        'id': 'minecraft:white_wool',
        'names': {
            'singular': ['white wool'],
            'plural': ['white wool']
        }
    },
    {
        'id': 'minecraft:orange_wool',
        'names': {
            'singular': ['orange wool'],
            'plural': ['orange wool']
        }
    },
    {
        'id': 'minecraft:magenta_wool',
        'names': {
            'singular': ['magenta wool'],
            'plural': ['magenta wool']
        }
    },
    {
        'id': 'minecraft:light_blue_wool',
        'names': {
            'singular': ['light blue wool'],
            'plural': ['light blue wool']
        }
    },
    {
        'id': 'minecraft:yellow_wool',
        'names': {
            'singular': ['yellow wool'],
            'plural': ['yellow wool']
        }
    },
    {
        'id': 'minecraft:lime_wool',
        'names': {
            'singular': ['lime wool'],
            'plural': ['lime wool']
        }
    },
    {
        'id': 'minecraft:pink_wool',
        'names': {
            'singular': ['pink wool'],
            'plural': ['pink wool']
        }
    },
    {
        'id': 'minecraft:gray_wool',
        'names': {
            'singular': ['gray wool'],
            'plural': ['gray wool']
        }
    },
    {
        'id': 'minecraft:light_gray_wool',
        'names': {
            'singular': ['light gray wool'],
            'plural': ['light gray wool']
        }
    },
    {
        'id': 'minecraft:cyan_wool',
        'names': {
            'singular': ['cyan wool'],
            'plural': ['cyan wool']
        }
    },
    {
        'id': 'minecraft:purple_wool',
        'names': {
            'singular': ['purple wool'],
            'plural': ['purple wool']
        }
    },
    {
        'id': 'minecraft:blue_wool',
        'names': {
            'singular': ['blue wool'],
            'plural': ['blue wool']
        }
    },
    {
        'id': 'minecraft:brown_wool',
        'names': {
            'singular': ['brown wool'],
            'plural': ['brown wool']
        }
    },
    {
        'id': 'minecraft:green_wool',
        'names': {
            'singular': ['green wool'],
            'plural': ['green wool']
        }
    },
    {
        'id': 'minecraft:red_wool',
        'names': {
            'singular': ['red wool'],
            'plural': ['red wool']
        }
    },
    {
        'id': 'minecraft:black_wool',
        'names': {
            'singular': ['black wool'],
            'plural': ['black wool']
        }
    }
]

complex_block_types = [
    'minecraft:grass_block[snowy=false]',
    'minecraft:podzol[snowy=false]',
    'minecraft:oak_log[axis=x]',
    'minecraft:spruce_log[axis=x]',
    'minecraft:birch_log[axis=x]',
    'minecraft:jungle_log[axis=x]',
    'minecraft:acacia_log[axis=x]',
    'minecraft:dark_oak_log[axis=x]',
    'minecraft:stripped_oak_log[axis=x]',
    'minecraft:stripped_spruce_log[axis=x]',
    'minecraft:stripped_birch_log[axis=x]',
    'minecraft:stripped_jungle_log[axis=x]',
    'minecraft:stripped_acacia_log[axis=x]',
    'minecraft:stripped_dark_oak_log[axis=x]',
    'minecraft:stripped_oak_wood[axis=x]',
    'minecraft:stripped_spruce_wood[axis=x]',
    'minecraft:stripped_birch_wood[axis=x]',
    'minecraft:stripped_jungle_wood[axis=x]',
    'minecraft:stripped_acacia_wood[axis=x]',
    'minecraft:stripped_dark_oak_wood[axis=x]',
    'minecraft:oak_wood[axis=x]',
    'minecraft:spruce_wood[axis=x]',
    'minecraft:birch_wood[axis=x]',
    'minecraft:jungle_wood[axis=x]',
    'minecraft:acacia_wood[axis=x]',
    'minecraft:dark_oak_wood[axis=x]',
    'minecraft:oak_leaves[distance=7,persistent=true,waterlogged=false]',
    'minecraft:spruce_leaves[distance=7,persistent=true,waterlogged=false]',
    'minecraft:birch_leaves[distance=7,persistent=true,waterlogged=false]',
    'minecraft:jungle_leaves[distance=7,persistent=true,waterlogged=false]',
    'minecraft:acacia_leaves[distance=7,persistent=true,waterlogged=false]',
    'minecraft:dark_oak_leaves[distance=7,persistent=true,waterlogged=false]',
    'minecraft:dispenser[facing=north,triggered=false]',
    'minecraft:purpur_pillar[axis=y]',
    'minecraft:furnace[facing=north,lit=false]',
    'minecraft:redstone_ore[lit=false]',
    'minecraft:jukebox[has_record=false]',
    'minecraft:carved_pumpkin[facing=north]',
    'minecraft:jack_o_lantern[facing=north]',
    'minecraft:redstone_lamp[lit=false]',
    'minecraft:quartz_pillar[axis=y]',
    'minecraft:frosted_ice[age=0]',
    'minecraft:deepslate_redstone_ore[lit=false]',
    'minecraft:mycelium[snowy=false]',
    'minecraft:deepslate[axis=y]'
]

# all_block_types = AIR + complex_block_types
