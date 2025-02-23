from amulet import Block

block_properties_to_remove = {
    # Plants
    "sugar_cane": ["age"],
    "cactus": ["age"],
    "chorus_flower": ["age"],
    "kelp": ["age"],
    "kelp_plant": ["age"],
    "cave_vines": ["age"],
    "twisting_vines": ["age"],
    "weeping_vines": ["age"],
    "bamboo": ["stage"],
    "bamboo_sapling": ["stage"],
    "sapling": ["stage"],
    "mangrove_propagule": ["stage"],
    "leaves": ["check_decay", "distance", "persistent"],
    # Redstone
    "door": ["powered"],
    "trapdoor": ["powered"],
    "fence_gate": ["powered"],
    "button": ["powered"],
    "observer": ["powered"],
    "lectern": ["powered"],
    "tripwire": ["powered"],
    "tripwire_hook": ["powered"],
    "lightning_rod": ["powered"],
    "waxed_copper_bulb": ["powered"],
    "waxed_exposed_copper_bulb": ["powered"],
    "waxed_weathered_copper_bulb": ["powered"],
    "waxed_oxidized_copper_bulb": ["powered"],
    "note_block": ["instrument", "note", "powered"],
    "bell": ["powered", "toggle"],
    "head": ["powered", "no_drop"],
    "wall_head": ["powered", "no_drop"],
    "pressure_plate": ["powered"],
    "light_weighted_pressure_plate": ["power"],
    "heavy_weighted_pressure_plate": ["power"],
    "daylight_detector": ["power"],
    "target": ["power"],
    "sculk_sensor": ["power", "sculk_sensor_phase"],
    "calibrated_sculk_sensor": ["power", "sculk_sensor_phase"],
    "dispenser": ["triggered"],
    "dropper": ["triggered"],
    "hopper": ["enabled"],
    # Misc
    "fire": ["age"],
    "soul_fire": ["age"],
    "barrel": ["open"],
    "tnt": ["unstable", "underwater"],
    "scaffolding": ["bottom", "distance", "stability_checked"],
    "jukebox": ["has_record"],
}

block_names_to_change = {
    "cave_air": "air",
}


def clean_block_properties(block: Block) -> Block:
    if not block.properties or block.base_name not in block_properties_to_remove:
        return block
    block_properties: dict = block.properties
    for property in block_properties_to_remove[block.base_name]:
        block_properties.pop(property, None)
    if block_properties != block.properties:
        return Block(block.namespace, block.base_name, block_properties)
    return block


def clean_block(block: Block) -> Block:
    if block.base_name in block_names_to_change:
        block = Block(
            block.namespace, block_names_to_change[block.base_name], block.properties
        )
    block = clean_block_properties(block)
    return block


def main():
    test_block_states = {
        "universal_minecraft:cave_air": "universal_minecraft:air",
        # Plants
        "universal_minecraft:sugar_cane[age=9]": "universal_minecraft:sugar_cane",
        "universal_minecraft:cactus[age=0]": "universal_minecraft:cactus",
        "universal_minecraft:chorus_flower[age=5]": "universal_minecraft:chorus_flower",
        "universal_minecraft:kelp[age=25,waterlogged=true]": "universal_minecraft:kelp[waterlogged=true]",
        "universal_minecraft:kelp_plant[age=25,waterlogged=true]": "universal_minecraft:kelp_plant[waterlogged=true]",
        "universal_minecraft:cave_vines[age=1,berries=true]": "universal_minecraft:cave_vines[berries=true]",
        "universal_minecraft:twisting_vines[age=8]": "universal_minecraft:twisting_vines",
        "universal_minecraft:twisting_vines_plant[age=25]": "universal_minecraft:twisting_vines_plant",
        "universal_minecraft:weeping_vines[age=13]": "universal_minecraft:weeping_vines",
        "universal_minecraft:bamboo[leaves=none,stage=0,thickness=thick]": "universal_minecraft:bamboo[leaves=none,thickness=thick]",
        "universal_minecraft:bamboo_sapling[stage=0]": "universal_minecraft:bamboo_sapling",
        "universal_minecraft:sapling[material=jungle,stage=1]": "universal_minecraft:sapling[material=jungle]",
        "universal_minecraft:mangrove_propagule[age=4,hanging=true,stage=0]": "universal_minecraft:mangrove_propagule[age=4,hanging=true]",
        "universal_minecraft:leaves[check_decay=false,distance=7,material=azalea,persistent=true]": "universal_minecraft:leaves[material=azalea]",
        "universal_minecraft:flower_pot[plant=spruce_sapling,update=false]": "universal_minecraft:flower_pot[plant=spruce_sapling]",
        # Redstone
        "universal_minecraft:door[facing=east,half=lower,hinge=left,material=spruce,open=false,powered=false]": "universal_minecraft:door[facing=east,half=lower,hinge=left,material=spruce,open=false]",
        "universal_minecraft:trapdoor[facing=north,half=bottom,material=oak,open=true,powered=false]": "universal_minecraft:trapdoor[facing=north,half=bottom,material=oak,open=true]",
        "universal_minecraft:fence_gate[facing=south,in_wall=false,material=oak,open=false,powered=false]": "universal_minecraft:fence_gate[facing=south,in_wall=false,material=oak,open=false]",
        "universal_minecraft:button[face=wall,facing=west,material=oak,powered=false]": "universal_minecraft:button[face=wall,facing=west,material=oak]",
        "universal_minecraft:observer[facing=west,powered=false]": "universal_minecraft:observer[facing=west]",
        "universal_minecraft:lectern[facing=east,has_book=false,powered=false]": "universal_minecraft:lectern[facing=east,has_book=false]",
        "universal_minecraft:tripwire[attached=false,disarmed=false,east=false,north=false,powered=false,south=false,suspended=true,west=false]": "universal_minecraft:tripwire[attached=false,disarmed=false,east=false,north=false,south=false,suspended=true,west=false]",
        "universal_minecraft:tripwire_hook[attached=true,facing=west,powered=true]": "universal_minecraft:tripwire_hook[attached=true,facing=west]",
        "universal_minecraft:lightning_rod[facing=up,powered=false]": "universal_minecraft:lightning_rod[facing=up]",
        "universal_minecraft:waxed_copper_bulb[lit=true,powered=false]": "universal_minecraft:waxed_copper_bulb[lit=true]",
        "universal_minecraft:waxed_exposed_copper_bulb[lit=true,powered=false]": "universal_minecraft:waxed_exposed_copper_bulb[lit=true]",
        "universal_minecraft:waxed_weathered_copper_bulb[lit=true,powered=false]": "universal_minecraft:waxed_weathered_copper_bulb[lit=true]",
        "universal_minecraft:waxed_oxidized_copper_bulb[lit=true,powered=false]": "universal_minecraft:waxed_oxidized_copper_bulb[lit=true]",
        "universal_minecraft:note_block[instrument=basedrum,note=0,powered=false]": "universal_minecraft:note_block",
        "universal_minecraft:bell[attachment=ceiling,facing=north,powered=false,toggle=0]": "universal_minecraft:bell[attachment=ceiling,facing=north]",
        "universal_minecraft:head[mob=zombie,no_drop=false,powered=false,rotation=5]": "universal_minecraft:head[mob=zombie,rotation=5]",
        "universal_minecraft:wall_head[facing=west,mob=dragon,no_drop=false,powered=false]": "universal_minecraft:wall_head[facing=west,mob=dragon]",
        "universal_minecraft:daylight_detector[inverted=false,power=1]": "universal_minecraft:daylight_detector[inverted=false]",
        "universal_minecraft:target[power=0]": "universal_minecraft:target",
        "universal_minecraft:sculk_sensor[power=0,sculk_sensor_phase=cooldown,waterlogged=true]": "universal_minecraft:sculk_sensor[waterlogged=true]",
        "universal_minecraft:calibrated_sculk_sensor[facing=south,power=0,sculk_sensor_phase=inactive,waterlogged=true]": "universal_minecraft:calibrated_sculk_sensor[facing=south,waterlogged=true]",
        "universal_minecraft:dispenser[facing=east,triggered=false]": "universal_minecraft:dispenser[facing=east]",
        "universal_minecraft:hopper[enabled=true,facing=down]": "universal_minecraft:hopper[facing=down]",
        "universal_minecraft:dropper[facing=up,triggered=false]": "universal_minecraft:dropper[facing=up]",
        "universal_minecraft:pressure_plate[material=stone,powered=false]": "universal_minecraft:pressure_plate[material=stone]",
        "universal_minecraft:light_weighted_pressure_plate[power=0]": "universal_minecraft:light_weighted_pressure_plate",
        "universal_minecraft:heavy_weighted_pressure_plate[power=0]": "universal_minecraft:heavy_weighted_pressure_plate",
        # Misc
        "universal_minecraft:fire[age=0,east=false,north=false,south=false,up=false,west=false]": "universal_minecraft:fire[east=false,north=false,south=false,up=false,west=false]",
        "universal_minecraft:soul_fire[age=0]": "universal_minecraft:soul_fire",
        "universal_minecraft:barrel[facing=up,open=false]": "universal_minecraft:barrel[facing=up]",
        "universal_minecraft:tnt[underwater=false,unstable=false]": "universal_minecraft:tnt",
        "universal_minecraft:scaffolding[bottom=true,distance=1,stability_checked=true,waterlogged=true]": "universal_minecraft:scaffolding[waterlogged=true]",
        "universal_minecraft:jukebox[has_record=true]": "universal_minecraft:jukebox",
    }
    for block_state in test_block_states:
        block = Block.from_string_blockstate(block_state)
        block = clean_block(block)
        if block.blockstate != test_block_states[block_state]:
            print(f"Failed for {block_state}")


if __name__ == "__main__":
    main()
