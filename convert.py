import litemapy

# Import schematic and convert to sponge format
schematic = litemapy.Schematic.load('KbvChvZGXS-Hardcore House.litematic')
region = next(iter(schematic.regions.values()))
sponge_nbt = region.to_sponge_nbt()
sponge_nbt.save('KbvChvZGXS-Hardcore House.schem')