import h5py
from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

from model import MinecraftDataset


class MinecraftDataModule(LightningDataModule):
    def __init__(self, file_path: str, batch_size: int = 32):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []

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
        return DataLoader(train_datasets, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, drop_last=True) for _, dataset in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(dataset, batch_size=self.batch_size, drop_last=True) for _, dataset in self.test_datasets]
