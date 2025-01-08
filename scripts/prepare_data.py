from minecraft_schematic_generator.data_preparer import load_schematics

schematics_dir = "data/schematics/"
hdf5_path = "data/data_v2.h5"
load_schematics(
    schematics_dir,
    hdf5_path,
    (0.8, 0.10, 0.10),
    validation_only_datasets=["holdout"],
)
