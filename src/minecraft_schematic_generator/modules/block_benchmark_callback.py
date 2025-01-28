from lightning.pytorch.callbacks import Callback

from minecraft_schematic_generator.block_benchmark import run_benchmark


class BlockBenchmarkCallback(Callback):
    def __init__(self, num_runs=100, save_debug_schematics=False, base_seed=0):
        self.num_runs = num_runs
        self.save_debug_schematics = save_debug_schematics
        self.base_seed = base_seed

    def on_validation_epoch_end(self, trainer, pl_module):
        results = run_benchmark(
            pl_module.model,
            num_runs=self.num_runs,
            save_debug_schematics=self.save_debug_schematics,
            base_seed=self.base_seed,
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
