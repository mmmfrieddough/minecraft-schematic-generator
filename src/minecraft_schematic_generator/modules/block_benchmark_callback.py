from lightning.pytorch.callbacks import Callback

from minecraft_schematic_generator.block_benchmark import run_benchmark
from minecraft_schematic_generator.converter import BlockTokenConverter


class BlockBenchmarkCallback(Callback):
    def __init__(
        self,
        block_token_converter: BlockTokenConverter,
        num_runs: int = 100,
        save_debug_schematics: bool = False,
        base_seed: int = 0,
    ):
        self._block_token_converter = block_token_converter
        self._num_runs = num_runs
        self._save_debug_schematics = save_debug_schematics
        self._base_seed = base_seed

    def on_validation_epoch_end(self, trainer, pl_module):
        results = run_benchmark(
            pl_module.model,
            block_token_converter=self._block_token_converter,
            num_runs=self._num_runs,
            save_debug_schematics=self._save_debug_schematics,
            base_seed=self._base_seed,
            batch_size=trainer.val_dataloaders[0].batch_size,
            show_progress=(trainer.global_rank == 0),
        )
        pl_module.log("benchmark", results.average, sync_dist=True)
        for category in results.category_results:
            pl_module.log(
                f"benchmark/{category.name}", category.average, sync_dist=True
            )
            for benchmark in category.benchmark_results:
                pl_module.log(
                    f"benchmark/{category.name}/{benchmark.name}",
                    benchmark.average,
                    sync_dist=True,
                )
