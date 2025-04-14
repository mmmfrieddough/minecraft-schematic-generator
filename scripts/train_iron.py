import os

import torch
import torch._dynamo.config
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from minecraft_schematic_generator.constants import MAX_STRUCTURE_SIZE
from minecraft_schematic_generator.model.structure_masker import StructureMasker
from minecraft_schematic_generator.model.structure_transformer import (
    StructureTransformer,
)
from minecraft_schematic_generator.modules import (
    BlockBenchmarkCallback,
    LightningTransformerMinecraftStructureGenerator,
    MinecraftDataModule,
    SaveOnInterruptCallback,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main():
    torch.set_float32_matmul_precision("medium")

    experiment_name = "iron_v1"
    experiment_version = 11
    checkpoint_dir = "lightning_logs"
    tensorboard_logger = TensorBoardLogger(
        checkpoint_dir, name=experiment_name, version=experiment_version
    )
    # wandb_logger = WandbLogger(
    #     name=experiment_name,
    #     project="minecraft-structure-generator",
    #     version=str(experiment_version),
    # )

    structure_masker = StructureMasker()
    structure_transformer = StructureTransformer()
    data_module = MinecraftDataModule(
        file_path="/mnt/windows/data_v7.h5",
        structure_masker=structure_masker,
        structure_transformer=structure_transformer,
        # crop_sizes={7: 550, 9: 264, 11: 143, 13: 85, 15: 57},
        # crop_sizes={7: 430},
        crop_sizes={15: 43, 13: 80, 11: 130, 9: 240, 7: 430},
        num_workers=20,
        persistent_workers=True,
        separate_validation_datasets=["hermitcraft\\hermitcraft6\\overworld"],
    )

    lightning_model = LightningTransformerMinecraftStructureGenerator(
        num_classes=15000,
        block_str_mapping=data_module.get_block_str_mapping(),
        max_structure_size=MAX_STRUCTURE_SIZE,
        embedding_dropout=0.1,
        embedding_dim=256,
        model_dim=384,
        num_heads=6,
        num_layers=4,
        decoder_dropout=0.1,
        max_learning_rate=1e-4,
        warmup_proportion=0.05,
    )
    torch._dynamo.config.cache_size_limit = 16
    lightning_model = torch.compile(lightning_model)

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
        batch_size=35,
    )

    trainer = Trainer(
        max_epochs=20,
        logger=[tensorboard_logger],
        val_check_interval=0.1,
        limit_val_batches=0.2,
        accumulate_grad_batches=8,
        precision="bf16-mixed",
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
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
