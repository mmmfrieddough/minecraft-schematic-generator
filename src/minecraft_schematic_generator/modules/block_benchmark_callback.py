from lightning.pytorch.callbacks import Callback

from minecraft_schematic_generator.block_benchmark import run_benchmark
from minecraft_schematic_generator.converter import BlockTokenConverter


class BlockBenchmarkCallback(Callback):
    def __init__(
        self,
        block_token_converter: BlockTokenConverter,
        schematic_size: int,
        num_runs: int = 100,
        batch_size: int = 0,
        save_debug_schematics: bool = False,
        base_seed: int = 0,
    ):
        self._block_token_converter = block_token_converter
        self._schematic_size = schematic_size
        self._num_runs = num_runs
        self._batch_size = batch_size
        self._save_debug_schematics = save_debug_schematics
        self._base_seed = base_seed

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if trainer.global_rank == 0:
            results = run_benchmark(
                pl_module.model,
                block_token_converter=self._block_token_converter,
                schematic_size=self._schematic_size,
                num_runs=self._num_runs,
                save_debug_schematics=self._save_debug_schematics,
                base_seed=self._base_seed,
                batch_size=self._batch_size,
            )
            # Create a dictionary of all metrics
            metrics = {"benchmark": results.average}
            for category in results.category_results:
                metrics[f"benchmark/{category.name}"] = category.average
                for benchmark in category.benchmark_results:
                    metrics[f"benchmark/{category.name}/{benchmark.name}"] = (
                        benchmark.average
                    )
        else:
            # Initialize empty metrics on other ranks
            metrics = {}

        # Broadcast metrics from rank 0 to all other ranks
        if trainer.world_size > 1:
            metrics = trainer.strategy.broadcast(metrics, 0)

        # Now log metrics with sync_dist=True since all ranks have the same values
        for name, value in metrics.items():
            pl_module.log(name, value, sync_dist=True, on_epoch=True)
