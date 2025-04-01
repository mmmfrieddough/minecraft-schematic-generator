from collections import deque
import time
from lightning import Callback


class RollingBatchTimeCallback(Callback):
    def __init__(self, warmup_batches=50, window_size=100, print_every=10):
        super().__init__()
        self.warmup_batches = warmup_batches
        self.window_size = window_size
        self.print_every = print_every

        self.batch_times = deque(maxlen=window_size)
        self.last_batch_time = None
        self.batch_count = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.is_global_zero:
            self.last_batch_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1

        # Skip measurements during warmup
        if self.batch_count <= self.warmup_batches:
            if self.batch_count == self.warmup_batches and trainer.is_global_zero:
                print(
                    f"\nWarmup complete after {self.warmup_batches} batches. Starting measurements..."
                )
            return

        # Only track timing on the main process
        if not trainer.is_global_zero:
            return

        self.batch_times.append(time.time())

        # Print rolling average every N batches
        if self.batch_count % self.print_every == 0 and len(self.batch_times) > 0:
            oldest_time = self.batch_times[0]
            elapsed_time = time.time() - oldest_time
            batches_per_sec = len(self.batch_times) / elapsed_time
            print(f"Batch {self.batch_count}: {batches_per_sec:.2f} batches/sec")
