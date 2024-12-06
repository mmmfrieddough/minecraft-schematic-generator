from typing import List

import h5py
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from model import MinecraftDataset


class MinecraftDataModule(LightningDataModule):
    def __init__(self, file_path: str, batch_size: int = 32, num_workers: int = 0, combine_datasets: bool = False, separate_validation_datasets: List[str] = []):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0
        self.combine_datasets = combine_datasets
        self.separate_validation_datasets = separate_validation_datasets
        self.val_dataset_names = {}

    def get_val_dataset_name(self, dataloader_idx: int):
        return self.val_dataset_names[dataloader_idx]

    def setup(self, stage=None, index: int = None):
        with h5py.File(self.file_path, 'r') as file:
            if index is not None:
                train_keys = [list(file['train'].keys())[index]]
                val_keys = [list(file['validation'].keys())[index]]
                test_keys = [list(file['test'].keys())[index]]
            else:
                train_keys = file['train'].keys()
                val_keys = file['validation'].keys()
                test_keys = file['test'].keys()

            self.train_datasets = [(generator_type, MinecraftDataset(
                self.file_path, 'train', generator_type)) for generator_type in tqdm(train_keys, desc="Loading training datasets")]
            self.val_datasets = [(generator_type, MinecraftDataset(
                self.file_path, 'validation', generator_type)) for generator_type in tqdm(val_keys, desc="Loading validation datasets")]
            self.test_datasets = [(generator_type, MinecraftDataset(
                self.file_path, 'test', generator_type)) for generator_type in tqdm(test_keys, desc="Loading test datasets")]

    def train_dataloader(self):
        train_datasets = ConcatDataset(
            [dataset for _, dataset in self.train_datasets])
        assert len(train_datasets) > 0, "Training DataLoader is empty."
        return DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        if self.combine_datasets:
            # Filter datasets to be combined and kept separate based on `separate_validation_datasets`
            combined_val_datasets = [
                dataset for generator_type, dataset in self.val_datasets if generator_type not in self.separate_validation_datasets]
            separate_val_datasets = [
                dataset for generator_type, dataset in self.val_datasets if generator_type in self.separate_validation_datasets]
            separate_val_dataset_names = [generator_type for generator_type,
                                          dataset in self.val_datasets if generator_type in self.separate_validation_datasets]

            # Create a DataLoader for the combined datasets
            if combined_val_datasets:
                combined_val_loader = DataLoader(ConcatDataset(combined_val_datasets), batch_size=self.batch_size,
                                                 shuffle=True, drop_last=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
                val_loaders = [combined_val_loader]
                self.val_dataset_names[0] = "combined"
            else:
                val_loaders = []

            # Add separate DataLoaders for datasets not to be combined
            for i in range(len(separate_val_datasets)):
                dataloader = DataLoader(separate_val_datasets[i], batch_size=self.batch_size, shuffle=True,
                                        drop_last=False, num_workers=self.num_workers, persistent_workers=self.persistent_workers)
                val_loaders.append(dataloader)
                self.val_dataset_names[i + 1] = separate_val_dataset_names[i]
        else:
            val_loaders = [DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                      num_workers=self.num_workers, persistent_workers=self.persistent_workers) for _, dataset in self.val_datasets]
        for i, loader in enumerate(val_loaders):
            assert len(
                loader) > 0, f"Validation DataLoader at index {i} is empty."
        return val_loaders

    def test_dataloader(self):
        if self.combine_datasets:
            test_datasets = ConcatDataset(
                [dataset for _, dataset in self.test_datasets])
            test_loader = DataLoader(test_datasets, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                     num_workers=self.num_workers, persistent_workers=self.persistent_workers)
            test_loaders = [test_loader]
        else:
            test_loaders = [DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False,
                                       num_workers=self.num_workers, persistent_workers=self.persistent_workers) for _, dataset in self.test_datasets]
        for i, loader in enumerate(test_loaders):
            assert len(loader) > 0, f"Test DataLoader at index {i} is empty."
        return test_loaders
