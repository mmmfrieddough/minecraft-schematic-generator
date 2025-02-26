import h5py
import torch
from torch.utils.data import Dataset


class MinecraftDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str,
        path: str,
    ):
        self._file_path = file_path
        self._split = split
        self._path = path

        # Get the dataset length
        with h5py.File(file_path, "r") as file:
            group = file[split][path]
            self.length = len(group["names"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self._file_path, "r") as file:
            structure = torch.from_numpy(
                file[self._split][self._path]["structures"][idx][()]
            ).long()

        return structure
