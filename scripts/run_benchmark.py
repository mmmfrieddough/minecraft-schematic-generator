from minecraft_schematic_generator.block_benchmark import run_benchmark
from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    DictBlockTokenMapper,
)
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

experiment_name = "diamond_v1"
model_version = "version_1"
checkpoint_path = (
    f"lightning_logs/{experiment_name}/{model_version}/checkpoints/last.ckpt"
)
model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    checkpoint_path, map_location="cpu"
).model
model.eval()

block_token_mapper = DictBlockTokenMapper(model.block_str_mapping)
block_token_converter = BlockTokenConverter(block_token_mapper)

# Run all benchmarks
results = run_benchmark(
    model,
    block_token_converter=block_token_converter,
    schematic_size=11,
    num_runs=1,
    save_debug_schematics=True,
    base_seed=0,
    batch_size=50,
)
print(results)
