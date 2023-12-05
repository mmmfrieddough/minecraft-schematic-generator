import os
from datetime import datetime

import optuna
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

from common.file_paths import BASE_DIR
from modules import (GenerateSchematicCallback,
                     LightningTransformerMinecraftStructureGenerator,
                     MinecraftDataModule)


def objective(trial: optuna.Trial):
    seed_everything(0, workers=True)

    lightning_model = LightningTransformerMinecraftStructureGenerator(
        num_classes=20,
        max_sequence_length=512,
        embedding_dim=trial.suggest_categorical(
            "embedding_dim", [16, 32, 64, 128]),
        embedding_dropout=trial.suggest_float(
            "embedding_dropout", 0.0, 0.5, step=0.1),
        decoder_dim=trial.suggest_categorical(
            "decoder_dim", [32, 64, 128, 256]),
        num_heads=trial.suggest_categorical("num_heads", [1, 2, 4, 8, 16, 32]),
        num_layers=trial.suggest_int("num_layers", 1, 4),
        decoder_dropout=trial.suggest_float(
            "decoder_dropout", 0.0, 0.5, step=0.1),
        freeze_encoder=True,
        learning_rate=trial.suggest_float(
            "learning_rate", 1e-5, 1e-2, log=True)
    )

    hdf5_file = os.path.join(BASE_DIR, 'data.h5')
    data_module = MinecraftDataModule(
        file_path=hdf5_file,
        batch_size=32,
        # num_workers=4
    )

    logger = TensorBoardLogger(
        'lightning_logs', name='minecraft_structure_generator', log_graph=False)
    profiler = SimpleProfiler()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=False,
        save_last=True,
        save_top_k=3,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=False,
        mode='min'
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
        # profiler=profiler,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        callbacks=[
            # checkpoint_callback,
            early_stop_callback,
            # generate_schematic_callback
        ]
    )

    trainer.fit(lightning_model, datamodule=data_module)

    trainable_params = sum(p.numel()
                           for p in lightning_model.parameters() if p.requires_grad)
    return trainer.callback_metrics["val_loss"], trainer.callback_metrics["val_accuracy"], trainable_params


if __name__ == "__main__":
    # study_name = "study_20231204195755"
    study_name = f"study_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    study = optuna.create_study(directions=("minimize", "maximize", "minimize"),
                                study_name=study_name, storage='sqlite:///studies.db', load_if_exists=True)
    study.optimize(objective, n_trials=100)

    print("Best hyperparameters:", study.best_trial.params)
