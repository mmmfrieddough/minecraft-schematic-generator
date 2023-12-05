import h5py
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from model import MinecraftDataset


class MinecraftDataModule(LightningDataModule):
    def __init__(self, file_path: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0

    def setup(self, stage=None):
        with h5py.File(self.file_path, 'r') as file:
            self.train_datasets = [(generator_type, MinecraftDataset(
                self.file_path, 'train', generator_type)) for generator_type in file['train'].keys()]
            self.val_datasets = [(generator_type, MinecraftDataset(
                self.file_path, 'validation', generator_type)) for generator_type in file['validation'].keys()]
            self.test_datasets = [(generator_type, MinecraftDataset(
                self.file_path, 'test', generator_type)) for generator_type in file['test'].keys()]

    def train_dataloader(self):
        train_datasets = ConcatDataset(
            [dataset for _, dataset in self.train_datasets])
        assert len(train_datasets) > 0, "Training DataLoader is empty."
        return DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        val_loaders = [DataLoader(dataset, batch_size=self.batch_size, drop_last=False, num_workers=self.num_workers,
                                  persistent_workers=self.persistent_workers) for _, dataset in self.val_datasets]
        for i, loader in enumerate(val_loaders):
            assert len(
                loader) > 0, f"Validation DataLoader at index {i} is empty."
        return val_loaders

    def test_dataloader(self):
        test_loaders = [DataLoader(dataset, batch_size=self.validation_batch_size, drop_last=False,
                                   num_workers=self.num_workers, persistent_workers=self.persistent_workers) for _, dataset in self.test_datasets]
        for i, loader in enumerate(test_loaders):
            assert len(loader) > 0, f"Test DataLoader at index {i} is empty."
        return test_loaders
