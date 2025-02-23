import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import wandb
from minecraft_schematic_generator.constants import MAX_STRUCTURE_SIZE
from minecraft_schematic_generator.model.structure_masker import StructureMasker
from minecraft_schematic_generator.modules import (
    BlockBenchmarkCallback,
    LightningTransformerMinecraftStructureGenerator,
    MinecraftDataModule,
    SaveOnInterruptCallback,
)


def main():
    seed_everything(0)

    torch.set_float32_matmul_precision("medium")

    experiment_name = "new_data_augmentation"
    experiment_version = 2
    checkpoint_dir = "lightning_logs"
    tensorboard_logger = TensorBoardLogger(
        checkpoint_dir, name=experiment_name, version=experiment_version
    )
    wandb_logger = WandbLogger(
        name=experiment_name,
        project="minecraft-structure-generator",
        version=str(experiment_version),
    )

    structure_masker = StructureMasker()
    data_module = MinecraftDataModule(
        file_path="data/data_v3.h5",
        structure_masker=structure_masker,
        batch_size=77,
        num_workers=8,
        separate_validation_datasets=["holdout\\holdout1\\overworld"],
    )

    lightning_model = LightningTransformerMinecraftStructureGenerator(
        num_classes=13000,
        block_str_mapping=data_module.block_str_mapping,
        max_structure_size=MAX_STRUCTURE_SIZE,
        embedding_dropout=0.1,
        embedding_dim=128,
        model_dim=192,
        num_heads=2,
        num_layers=2,
        decoder_dropout=0.1,
        max_learning_rate=5e-4,
        warmup_proportion=0.1,
    )

    latest_checkpoint_callback = ModelCheckpoint(save_last=True)
    best_model_checkpoint_callback = ModelCheckpoint(
        filename="best", monitor="val_loss", mode="min"
    )
    save_on_interrupt_callback = SaveOnInterruptCallback(
        checkpoint_callback=latest_checkpoint_callback
    )
    lr_monitor_callback = LearningRateMonitor()
    block_benchmark_callback = BlockBenchmarkCallback(
        block_token_converter=data_module.get_block_token_converter(),
        schematic_size=11,
        num_runs=200,
    )

    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False)

    trainer = Trainer(
        strategy=ddp,
        max_epochs=10,
        logger=[tensorboard_logger, wandb_logger],
        val_check_interval=0.1,
        limit_val_batches=0.2,
        accumulate_grad_batches=4,
        precision="bf16-mixed",
        callbacks=[
            latest_checkpoint_callback,
            best_model_checkpoint_callback,
            save_on_interrupt_callback,
            lr_monitor_callback,
            block_benchmark_callback,
        ],
    )

    trainer.fit(lightning_model, datamodule=data_module, ckpt_path="last")

    wandb.finish()


if __name__ == "__main__":
    main()
