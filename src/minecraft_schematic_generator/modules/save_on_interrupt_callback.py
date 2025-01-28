from lightning import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer


class SaveOnInterruptCallback(Callback):
    def __init__(self, checkpoint_callback):
        self.checkpoint_callback = checkpoint_callback

    def on_exception(
        self, trainer: Trainer, lightning_module: LightningModule, exception: Exception
    ):
        if (
            isinstance(exception, KeyboardInterrupt)
            and self.checkpoint_callback is not None
        ):
            print("\nSaving checkpoint on interrupt...")
            self.checkpoint_callback.on_validation_end(trainer, lightning_module)
