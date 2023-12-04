import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from common.file_paths import BASE_DIR
from modules import (GenerateSchematicCallback,
                     LightningTransformerMinecraftStructureGenerator,
                     MinecraftDataModule)

seed_everything(0, workers=True)

lightning_model = LightningTransformerMinecraftStructureGenerator(
    num_classes=20,
    max_sequence_length=512,
    embedding_dim=32,
    embedding_dropout=0.1,
    decoder_dim=32,
    num_heads=2,
    num_layers=2,
    decoder_dropout=0.1,
    freeze_encoder=True,
    learning_rate=1e-3
)
# checkpoint_version = 4
# lightning_model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
#     f'lightning_logs/minecraft_structure_generator/version_{checkpoint_version}/checkpoints/last.ckpt')

hdf5_file = os.path.join(BASE_DIR, 'data.h5')
data_module = MinecraftDataModule(
    file_path=hdf5_file,
    batch_size=32,
    # num_workers=4
)

logger = TensorBoardLogger(
    'lightning_logs', name='minecraft_structure_generator', log_graph=True)
profiler = SimpleProfiler()
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    verbose=False,
    save_last=True,
    save_top_k=3,
    mode='max',
)
early_stop_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=50,
    verbose=False,
    mode='max'
)
generate_schematic_callback = GenerateSchematicCallback(
    save_path='schematic_viewer/public/schematics/',
    data_module=data_module,
    generate_train=False,
    generate_val=True,
    generate_all_datasets=False,
    generate_every_n_epochs=5,
    autoregressive=True
)

trainer = Trainer(
    max_epochs=5000,
    logger=logger,
    profiler=profiler,
    gradient_clip_val=1.0,
    log_every_n_steps=1,
    callbacks=[
        checkpoint_callback,
        early_stop_callback,
        generate_schematic_callback
    ]
)


def main():
    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == '__main__':
    main()
