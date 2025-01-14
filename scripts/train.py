import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import wandb
from minecraft_schematic_generator.modules import (
    LightningTransformerMinecraftStructureGenerator,
    MinecraftDataModule,
)


def main():
    torch.set_float32_matmul_precision("medium")

    experiment_name = "center_data"
    experiment_version = 13
    checkpoint_dir = "lightning_logs"
    tensorboard_logger = TensorBoardLogger(
        checkpoint_dir, name=experiment_name, version=experiment_version
    )
    wandb_logger = WandbLogger(
        name=experiment_name,
        project="minecraft-structure-generator",
        version=str(experiment_version),
    )

    lightning_model = LightningTransformerMinecraftStructureGenerator(
        num_classes=20000,
        max_sequence_length=1331,
        embedding_dropout=0.1,
        model_dim=768,
        num_heads=6,
        num_layers=6,
        decoder_dropout=0.1,
        max_learning_rate=1e-6,
        warmup_steps=100000,
    )

    hdf5_file = "data/data_v2.h5"
    batch_size = 20
    data_module = MinecraftDataModule(
        file_path=hdf5_file,
        batch_size=batch_size,
        combine_datasets=True,
        separate_validation_datasets=["holdout"],
    )

    latest_checkpoint_callback = ModelCheckpoint(save_last=True)
    best_model_checkpoint_callback = ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min"
    )
    lr_monitor_callback = LearningRateMonitor()

    ddp = DDPStrategy(process_group_backend="gloo")

    trainer = Trainer(
        strategy=ddp,
        devices=2,
        max_epochs=5000,
        logger=[tensorboard_logger, wandb_logger],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=5000,
        limit_val_batches=1000,
        num_sanity_val_steps=0,
        callbacks=[
            latest_checkpoint_callback,
            best_model_checkpoint_callback,
            lr_monitor_callback,
        ],
    )

    trainer.fit(lightning_model, datamodule=data_module, ckpt_path="last")

    wandb.finish()


if __name__ == "__main__":
    main()
