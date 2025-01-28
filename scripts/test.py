import amulet

world = amulet.load_level("New World (1)")
chunk = world.level_wrapper.load_chunk(0, 0, "minecraft:overworld")
for block in chunk.block_palette.blocks:
    print(block)
