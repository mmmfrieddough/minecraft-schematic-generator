import h5py
import torch
from torch.utils.data import Dataset


class MinecraftDataset(Dataset):
    def __init__(self, file_path: str, split: str, generator: str):
        self.file_path: str = file_path
        self.split: str = split
        self.generator: str = generator
        with h5py.File(file_path, 'r') as file:
            self.length: int = len(file[split][generator]['names'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.split][self.generator]
            name = group['names'][idx].decode('utf-8')
            description = group['prompts'][idx].decode('utf-8')
            data = torch.tensor(group['structures'][idx]).long()
        return name, description, data
