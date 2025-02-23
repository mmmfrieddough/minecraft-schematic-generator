import h5py
import torch
from torch.profiler import record_function
from torch.utils.data import Dataset

from .structure_masker import StructureMasker


class MinecraftDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        split: str,
        generator: str,
        structure_masker: StructureMasker,
    ):
        self._file_path = file_path
        self._split = split
        self._generator = generator
        self._structure_masker = structure_masker

        # Get the dataset length
        with h5py.File(file_path, "r") as file:
            group = file[split][generator]
            self.length = len(group["names"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with record_function("getitem_total"):
            with record_function("read_structure"):
                # Load single structure from disk when needed
                with h5py.File(self._file_path, "r") as file:
                    # Read directly into a torch tensor with the correct dtype
                    structure = torch.from_numpy(
                        file[self._split][self._generator]["structures"][idx][()]
                    ).long()

            with record_function("mask_structure"):
                masked_structure = self._structure_masker.mask_structure(structure)

            return structure, masked_structure
