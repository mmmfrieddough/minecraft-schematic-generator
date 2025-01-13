from minecraft_schematic_generator.block_benchmark import run_benchmark
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
)

experiment_name = "mini_model"
model_version = 8
# experiment_name = "center_data"
# model_version = 12
checkpoint_path = (
    f"lightning_logs/{experiment_name}/version_{model_version}/checkpoints/last.ckpt"
)
model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    checkpoint_path
)
model.eval()

# Run all benchmarks
results = run_benchmark(model, num_runs=1, save_debug_schematics=True, base_seed=0)
print(results)
