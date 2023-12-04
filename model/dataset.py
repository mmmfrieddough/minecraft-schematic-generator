import random
import h5py
import torch
from torch.utils.data import Dataset


class MinecraftDataset(Dataset):
    def __init__(self, file_path: str, split: str, generator: str):
        self.file_path: str = file_path
        self.split: str = split
        self.generator: str = generator
        with h5py.File(self.file_path, 'r') as file:
            self.names = list(file[split][generator].keys())
            self.length: int = len(self.names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.names[idx]
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.split][self.generator][name]
            prompt = random.choice(group['prompts']).decode('utf-8')
            structure = torch.from_numpy(group['structure'][:]).long()
        return name, prompt, structure
