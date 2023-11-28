import json
import os
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from schempy import Schematic

from converter import SchematicArrayConverter


class GenerateSchematicCallback(Callback):
    def __init__(self, save_path, data_module, generate_train=False, generate_val=True, generate_every_n_epochs=1):
        self.save_path = save_path
        self.data_module = data_module
        self.generate_train = generate_train
        self.generate_val = generate_val
        self.generate_every_n_epochs = generate_every_n_epochs
        self.schematic_array_converter = SchematicArrayConverter()

    def setup(self, trainer, pl_module, stage) -> None:
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        # Delete all existing samples
        for filename in os.listdir(self.save_path):
            filepath = os.path.join(self.save_path, filename)
            os.remove(filepath)

    def generate_sample(self, module, dataloader):
        # Select a sample
        samples = next(iter(dataloader))
        features, _, description = samples
        features = features[0]

        # Move the sample to the same device as the model
        features = features.to(module.device)

        # Generate a sample using the model
        module.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            generated_sample = module(features.unsqueeze(0))
        module.train()  # Set the model back to training mode

        # Convert logits to classes
        generated_sample = torch.argmax(generated_sample, dim=1)
        generated_sample = generated_sample.squeeze(0)

        # Convert the sample to the desired format using the provided function
        schematic = self.schematic_array_converter.array_to_schematic(
            generated_sample)
        schematic.name = description[0]

        return schematic

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if not self.generate_train or (epoch + 1) % self.generate_every_n_epochs != 0:
            return

        # Get the train dataloader
        train_dataloader = self.data_module.train_dataloader()

        # Generate a sample
        schematic = self.generate_sample(module, train_dataloader)

        # Save the sample
        epoch = trainer.current_epoch
        filename = f'sample_epoch_{epoch}_train.schem'
        self.save_sample(schematic, epoch, filename)

    def on_validation_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if not self.generate_val or (epoch + 1) % self.generate_every_n_epochs != 0:
            return

        # Get all validation dataloaders
        val_dataloaders = self.data_module.val_dataloader()
        for i, val_loader in enumerate(val_dataloaders):
            # Generate a sample
            schematic = self.generate_sample(module, val_loader)

            # Save the sample
            epoch = trainer.current_epoch
            dataset_name, _ = self.data_module.val_datasets[i]
            filename = f'sample_epoch_{epoch}_dataloader_{dataset_name}.schem'
            self.save_sample(schematic, epoch, filename)

    def save_sample(self, schematic: Schematic, epoch: int, filename: str):
        filepath = os.path.join(self.save_path, filename)
        schematic.save_to_file(Path(filepath), 2)
