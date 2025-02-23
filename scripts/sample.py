from minecraft_schematic_generator.data_sampler import WorldSampler


def main():
    world_sampler = WorldSampler(
        schematic_directory="data/schematics",
        temp_directory="data/temp",
        progress_save_period=30,
        chunk_mark_radius=1,
        sample_overlap_proportion=0.75,
        sample_size=11,
        sample_target_block_threshold=0.04,
        sample_minimum_air_threshold=0.15,
        sampling_purge_interval=3,
        resource_usage_limit=0.8,
        save_schematics=False,
        save_to_hdf5=True,
        hdf5_path="data/data_v3.h5",
    )
    dir = "data"
    world_sampler.sample_directory(dir)


if __name__ == "__main__":
    main()
