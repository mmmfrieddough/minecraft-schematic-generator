from minecraft_schematic_generator.data_preparer import load_schematics

schematics_dir = "data/test_schematics"
hdf5_path = "data/data_test_large.h5"


def main():
    load_schematics(
        schematics_dir,
        hdf5_path,
        (0.8, 0.10, 0.10),
        # validation_only_datasets=["holdout"],
        num_workers=30,
    )


if __name__ == "__main__":
    main()
