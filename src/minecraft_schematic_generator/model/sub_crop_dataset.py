from torch.profiler import record_function
from torch.utils.data import Dataset

from .structure_masker import StructureMasker


class SubCropDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        structure_masker: StructureMasker,
        crop_size: int,
        indices: list[int],
    ):
        self._dataset = dataset
        self._structure_masker = structure_masker
        self._crop_size = crop_size
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        with record_function("Get item"):
            with record_function("Read structure"):
                structure = self._dataset[self._indices[idx]]

            with record_function("Crop structure"):
                # Check if the structure can be cropped
                z_size, y_size, x_size = structure.shape
                if (
                    z_size < self._crop_size
                    or y_size < self._crop_size
                    or x_size < self._crop_size
                ):
                    raise ValueError(
                        "Structure is too small to be cropped. "
                        f"Structure size: {z_size}x{y_size}x{x_size}, "
                        f"Crop size: {self._crop_size}x{self._crop_size}x{self._crop_size}"
                    )

                # Crop the structure to the correct size
                z_start = (z_size - self._crop_size) // 2
                y_start = (y_size - self._crop_size) // 2
                x_start = (x_size - self._crop_size) // 2
                structure = structure[
                    z_start : z_start + self._crop_size,
                    y_start : y_start + self._crop_size,
                    x_start : x_start + self._crop_size,
                ]

            with record_function("Mask structure"):
                masked_structure = self._structure_masker.mask_structure(structure)

        return structure, masked_structure
