import os

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from common.file_paths import BASE_DIR
from modules import (GenerateSchematicCallback,
                     LightningTransformerMinecraftStructureGenerator,
                     MinecraftDataModule)
import wandb


def main():
    torch.set_float32_matmul_precision('medium')
    seed_everything(0, workers=True)

    experiment_name = "real_run"
    experiment_version = 12
    checkpoint_dir = "lightning_logs"
    tensorboard_logger = TensorBoardLogger(
        checkpoint_dir, name=experiment_name, version=experiment_version)
    wandb_logger = WandbLogger(
        name=experiment_name, project='minecraft-structure-generator', version=str(experiment_version))

    lightning_model = LightningTransformerMinecraftStructureGenerator(
        num_classes=20000,
        max_sequence_length=1331,
        embedding_dropout=0.1,
        model_dim=768,
        num_heads=12,
        num_layers=12,
        decoder_dropout=0.1,
        max_learning_rate=1e-5,
        warmup_steps=30000
    )

    hdf5_file = os.path.join(BASE_DIR, 'data.h5')
    batch_size = 12
    data_module = MinecraftDataModule(
        file_path=hdf5_file,
        batch_size=batch_size,
        # num_workers=4,
        combine_datasets=True,
        separate_validation_datasets=['validation_world']
    )
    wandb_logger.experiment.config['batch_size'] = batch_size

    latest_checkpoint_callback = ModelCheckpoint(
        save_last=True
    )
    best_model_checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=200,
        mode='min',
        verbose=True
    )
    generate_schematic_callback = GenerateSchematicCallback(
        full_path='schematic_viewer/public/schematics/full/',
        masked_path='schematic_viewer/public/schematics/masked/',
        filled_path='schematic_viewer/public/schematics/filled/',
        data_module=data_module,
        generate_train=False,
        generate_val=True,
        generate_all_datasets=False,
        generate_every_n_epochs=1,
        temperature=0.7
    )
    lr_monitor_callback = LearningRateMonitor()

    trainer = Trainer(
        max_epochs=5000,
        logger=[tensorboard_logger, wandb_logger],
        # profiler='simple',
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        val_check_interval=0.05,
        limit_val_batches=0.05,
        num_sanity_val_steps=0,
        callbacks=[
            latest_checkpoint_callback,
            best_model_checkpoint_callback,
            early_stop_callback,
            # generate_schematic_callback,
            lr_monitor_callback
        ]
    )

    trainer.fit(lightning_model, datamodule=data_module, ckpt_path='last')

    wandb.finish()


if __name__ == '__main__':
    main()
