from minecraft_schematic_generator.data_sampler import WorldSampler


def main():
    world_sampler = WorldSampler(
        schematic_directory="data/schematics",
        temp_directory="data/temp",
        progress_save_period=30,
        chunk_mark_radius=2,
        sample_check_size=9,
        sample_overlap_proportion=0.7,
        sample_target_block_threshold=0.03,
        sample_minimum_air_threshold=0.10,
        sample_collection_size=15,
        sampling_purge_interval=3,
        resource_usage_limit=0.8,
        save_schematics=False,
        # sample_limit=200,
        save_to_hdf5=True,
        hdf5_path="data/data_v6_test.h5",
        # max_workers=1,
    )
    dir = "data/worlds"
    world_sampler.sample_directory(dir)


if __name__ == "__main__":
    main()
