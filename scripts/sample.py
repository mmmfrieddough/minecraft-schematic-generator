from minecraft_schematic_generator.data_sampler import WorldSampler


def main():
    world_sampler = WorldSampler(
        schematic_directory="data/schematics/temp",
        temp_directory="data/temp",
        chunk_progress_save_interval=10000,
        chunk_mark_radius=2,
        sample_offset=7,
        sample_size=11,
        sample_interested_block_threshold=50,
        sample_minimum_air_threshold=200,
        sample_progress_save_interval=1000,
        sampling_purge_interval=3,
        num_workers=30,
        clear_worker_directories=False,
        chunk_search_limit=20000,
        sample_search_limit=2000,
        sample_limit=300,
    )
    world_sampler.clear_directory("data/worlds/temp")
    world_sampler.sample_directory("data/worlds/temp")


if __name__ == "__main__":
    main()
