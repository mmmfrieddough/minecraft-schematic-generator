import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class MinecraftDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        with h5py.File(self.hdf5_file, 'r') as hf:
            self.keys = list(hf.keys())  # Store the keys in a list

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as hf:
            return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as hf:
            group_key = self.keys[idx]  # Use the stored key
            group = hf[group_key]
            input_embedding = torch.tensor(group['input_embedding'][:])
            input_embedding = input_embedding.float()
            target_tensor = torch.tensor(group['target_tensor'][:])
        return input_embedding, target_tensor, group_key
