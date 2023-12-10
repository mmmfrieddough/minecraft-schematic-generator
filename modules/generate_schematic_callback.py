import os
from pathlib import Path
import random

import torch
from lightning.pytorch.callbacks import Callback
from schempy import Schematic

from converter import SchematicArrayConverter


class GenerateSchematicCallback(Callback):
    def __init__(self, save_path, data_module, generate_train=False, generate_val=True, generate_every_n_epochs=1, generate_all_datasets=True, temperature=1.0):
        self.save_path = save_path
        self.data_module = data_module
        self.generate_train = generate_train
        self.generate_val = generate_val
        self.generate_every_n_epochs = generate_every_n_epochs
        self.generate_all_datasets = generate_all_datasets
        self.temperature = temperature
        self.schematic_array_converter = SchematicArrayConverter()

    def setup(self, trainer, pl_module, stage) -> None:
        # Create the save directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        # Delete all existing samples
        for filename in os.listdir(self.save_path):
            filepath = os.path.join(self.save_path, filename)
            os.remove(filepath)

    def generate_sample(self, module, dataloader):
        # Pick a random sample from the dataloader
        i = random.randint(0, len(dataloader.dataset) - 1)
        _, prompt, _ = dataloader.dataset[i]

        # Move the sample to the same device as the model
        # features = features.to(module.device)

        # Generate a sample using the model
        module.eval()
        with torch.no_grad():
            generated_sample = module.generate(prompt, self.temperature)
        module.train()

        # Convert the sample to the desired format using the provided function
        schematic: Schematic = self.schematic_array_converter.array_to_schematic(
            generated_sample)
        schematic.name = prompt

        return schematic

    def _generate_and_save_sample(self, trainer, module, dataloader, dataset_name):
        # Generate a sample
        schematic = self.generate_sample(module, dataloader)

        # Save the sample
        epoch = trainer.current_epoch
        filename = f'sample_epoch_{epoch}_dataloader_{dataset_name}.schem'
        self.save_sample(schematic, epoch, filename)

    def on_train_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if not self.generate_train or (epoch + 1) % self.generate_every_n_epochs != 0:
            return

        # Get the train dataloader
        train_dataloader = self.data_module.train_dataloader()

        # Generate a sample
        self._generate_and_save_sample(
            trainer, module, train_dataloader, 'train')

    def on_validation_epoch_end(self, trainer, module):
        epoch = trainer.current_epoch
        if not self.generate_val or (epoch + 1) % self.generate_every_n_epochs != 0:
            return

        val_dataloaders = self.data_module.val_dataloader()
        if not self.generate_all_datasets:
            # Pick a random validation dataloader
            random_val_loader = random.choice(val_dataloaders)
            i = val_dataloaders.index(random_val_loader)
            name = self.data_module.val_datasets[i][0]

            # Generate a sample
            self._generate_and_save_sample(
                trainer, module, random_val_loader, name)
        else:
            # Go through all validation dataloaders
            for i, val_loader in enumerate(val_dataloaders):
                name = self.data_module.val_datasets[i][0]
                # Generate a sample
                self._generate_and_save_sample(
                    trainer, module, val_loader, name)

    def save_sample(self, schematic: Schematic, epoch: int, filename: str):
        filepath = os.path.join(self.save_path, filename)
        schematic.save_to_file(Path(filepath), 2)
