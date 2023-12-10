import os

import mlflow
import torch
from azureml.core import Workspace
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

from common.file_paths import BASE_DIR
from modules import (LightningTransformerMinecraftStructureGenerator,
                     MinecraftDataModule)


def train():
    torch.set_float32_matmul_precision('medium')
    seed_everything(0, workers=True)

    lightning_model = LightningTransformerMinecraftStructureGenerator(
        num_classes=250,
        max_sequence_length=512,
        embedding_dim=32,
        embedding_dropout=0.2,
        num_heads=16,
        num_layers=2,
        decoder_dropout=0.2,
        freeze_encoder=True,
        learning_rate=1e-4
    )
    # lightning_model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    #     'lightning_logs/minecraft_structure_generator/version_7/checkpoints/last.ckpt')

    hdf5_file = os.path.join(BASE_DIR, 'data.h5')
    data_module = MinecraftDataModule(
        file_path=hdf5_file,
        batch_size=64,
    )

    logger = TensorBoardLogger(
        'lightning_logs', name='minecraft_structure_generator', log_graph=False)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_weights_only=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=False,
        mode='min'
    )
    # generate_schematic_callback = GenerateSchematicCallback(
    #     save_path='schematic_viewer/public/schematics/',
    #     data_module=data_module,
    #     generate_train=False,
    #     generate_val=True,
    #     generate_all_datasets=False,
    #     generate_every_n_epochs=10
    # )

    workspace = Workspace.from_config()
    mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
    mlf_logger = MLFlowLogger()

    trainer = Trainer(
        max_epochs=5000,
        loggers=[logger, mlf_logger],
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            # generate_schematic_callback
        ]
    )

    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == '__main__':
    train()
