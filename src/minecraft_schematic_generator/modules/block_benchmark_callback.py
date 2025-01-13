from pytorch_lightning.callbacks import Callback


class BlockBenchmarkCallback(Callback):
    def __init__(self, benchmark_dataloader):
        self.benchmark_dataloader = benchmark_dataloader

    def on_epoch_end(self, trainer, pl_module):
        print("\nRunning benchmark...")
        pl_module.benchmark(self.benchmark_dataloader)
