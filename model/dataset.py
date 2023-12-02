import h5py
import torch
from torch.utils.data import Dataset


class MinecraftDataset(Dataset):
    def __init__(self, file_path, split_type, generator_type):
        self.file_path = file_path
        self.split_type = split_type
        self.generator_type = generator_type
        with h5py.File(file_path, 'r') as file:
            self.sample_names = list(
                file[self.split_type][self.generator_type].keys())
            self.length = len(self.sample_names)
        # Set to only load a single sample for debugging
        # test_pos = 5
        # self.sample_names = self.sample_names[test_pos:test_pos+1]
        # self.length = len(self.sample_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.split_type][self.generator_type][self.sample_names[idx]]
            features = torch.tensor(group['features'][:]).float()
            target = torch.tensor(group['target'][:]).long()
            description = group['description'][()].decode('utf-8')
        return features, target, description
