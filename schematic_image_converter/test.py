from litemapy import Schematic, Region, BlockState

# # Shortcut to create a schematic with a single region
# reg = Region(0, 0, 0, 21, 21, 21)
# schem = reg.as_schematic(name="Planet", author="SmylerMC",
#                          description="Made with litemapy")

# # Create the block state we are going to use
# block = BlockState("minecraft:light_blue_concrete")

# # Build the planet
# for x, y, z in reg.allblockpos():
#     if round(((x-10)**2 + (y-10)**2 + (z-10)**2)**.5) <= 10:
#         reg.setblock(x, y, z, block)

# # Save the schematic
# schem.save("planet.litematic")

# Load the schematic and get its first region
schem = Schematic.load("v7makd4rS9-LightHouse.litematic")
reg = list(schem.regions.values())[0]

# Print out every block in the region
for x, y, z in reg.allblockpos():
    b = reg.getblock(x, y, z)
    if b.blockid == "minecraft:smooth_quartz_stairs":
        print('stairs')
    if b.blockid == "minecraft:air":
        continue
    print(f"Block at ({x}, {y}, {z}): {b.blockid}")

# Print out the basic shape
# for x in reg.xrange():
#     for z in reg.zrange():
#         b = reg.getblock(x, 20, z)
#         if b.blockid == "minecraft:air":
#             print(" ", end="")
#         else:
#             print("#", end='')
#     print()
