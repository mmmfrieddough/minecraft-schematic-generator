import json
from typing import Any, List

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler
from tqdm import tqdm

from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    DictBlockTokenMapper,
)
from minecraft_schematic_generator.model import (
    CombinedDataLoader,
    MinecraftDataset,
    ResumableDataLoader,
    SubCropDataset,
)
from minecraft_schematic_generator.model.structure_masker import StructureMasker
from minecraft_schematic_generator.model.structure_transformer import (
    StructureTransformer,
)


class MinecraftDataModule(LightningDataModule):
    def __init__(
        self,
        file_path: str,
        structure_masker: StructureMasker,
        structure_transformer: StructureTransformer,
        crop_sizes: dict[int, int],
        num_workers: int = 0,
        persistent_workers: bool = False,
        separate_validation_datasets: List[str] = [],
    ):
        super().__init__()
        self._file_path = file_path
        self._crop_sizes = crop_sizes
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._separate_validation_datasets = separate_validation_datasets
        self._rng_state = None
        self._train_loaded_from_checkpoint = False

        self._train_dataset = None
        self._train_assigned_sizes = None
        self._train_dataloader = None

        self._val_datasets = {}
        self._val_shuffled_indices = {}
        self._val_assigned_sizes = {}
        self._val_dataloaders = {}

        self._test_dataset = None
        self._test_shuffled_indices = None
        self._test_assigned_sizes = None
        self._test_dataloader = None

        with h5py.File(self._file_path, "r") as file:
            # Load the mapping
            if "mapping" in file:
                mapping_group = file["mapping"]
                mapping_str = mapping_group["block_to_token"][()]
                mapping_json = json.loads(mapping_str)
                self._block_str_mapping = dict(mapping_json)
            else:
                raise ValueError("Mapping not found in HDF5 file.")

            self._structure_masker = structure_masker
            self._structure_transformer = structure_transformer
            block_token_mapper = DictBlockTokenMapper(self._block_str_mapping)
            block_token_converter = BlockTokenConverter(block_token_mapper)
            self._block_token_converter = block_token_converter
            self._structure_masker.setup(block_token_converter)
            self._structure_transformer.setup(block_token_converter)

    def get_block_str_mapping(self):
        return self._block_str_mapping

    def get_block_token_converter(self):
        return self._block_token_converter

    def save_split_state_dict(
        self,
        assigned_sizes: list[int],
        dataloader: CombinedDataLoader,
        shuffled_indices: list[int] | None = None,
    ) -> dict:
        # Save the assigned sizes and dataloader state
        state_dict = {
            "assigned_sizes": assigned_sizes,
            "dataloader_state": dataloader.state_dict(),
            "shuffled_indices": shuffled_indices,
        }

        return state_dict

    def state_dict(self) -> dict:
        """Save datamodule state."""
        state_dict = {}
        if self._train_dataloader:
            state_dict["train"] = self.save_split_state_dict(
                self._train_assigned_sizes, self._train_dataloader, None
            )

        val_states = {}
        for key, dataloader in self._val_dataloaders.items():
            if dataloader:
                val_states[key] = self.save_split_state_dict(
                    self._val_assigned_sizes[key],
                    dataloader,
                    self._val_shuffled_indices[key],
                )
        if val_states:
            state_dict["val"] = val_states

        if self._test_dataloader:
            state_dict["test"] = self.save_split_state_dict(
                self._test_assigned_sizes,
                self._test_dataloader,
                self._test_shuffled_indices,
            )
        return state_dict

    def load_split_state_dict(
        self, dataset: Dataset, state_dict: dict, train: bool
    ) -> None:
        # Load information from the state_dict
        assigned_sizes = state_dict["assigned_sizes"]
        dataloader_state = state_dict["dataloader_state"]
        shuffled_indices = state_dict.get("shuffled_indices", None)

        # Create dataloaders and load their states
        if shuffled_indices is not None:
            dataset = torch.utils.data.Subset(dataset, shuffled_indices)
        dataloader = self._get_cropped_dataloader(dataset, train, assigned_sizes)
        dataloader.load_state_dict(dataloader_state)

        return assigned_sizes, dataloader, shuffled_indices

    def load_state_dict(self, state_dict: dict) -> None:
        """Load datamodule state."""
        if "train" in state_dict:
            self._train_assigned_sizes, self._train_dataloader, _ = (
                self.load_split_state_dict(
                    self._train_dataset, state_dict["train"], train=True
                )
            )
            self._train_loaded_from_checkpoint = True

        if "val" in state_dict:
            val_states = state_dict["val"]
            for key, val_dataset in self._val_datasets.items():
                if key not in val_states:
                    continue

                assigned_sizes, dataloader, shuffled_indices = (
                    self.load_split_state_dict(
                        val_dataset, val_states[key], train=False
                    )
                )
                self._val_assigned_sizes[key] = assigned_sizes
                self._val_dataloaders[key] = dataloader
                self._val_shuffled_indices[key] = shuffled_indices

        if "test" in state_dict:
            (
                self._test_assigned_sizes,
                self._test_dataloader,
                self._test_shuffled_indices,
            ) = self.load_split_state_dict(
                self._test_dataset, state_dict["test"], train=False
            )

    def _get_cropped_dataloader(
        self, dataset: Dataset, train: bool, assigned_sizes: list[int]
    ) -> CombinedDataLoader:
        dataloaders = []
        for crop_size, batch_size in self._crop_sizes.items():
            indices = [i for i, size in enumerate(assigned_sizes) if size == crop_size]
            crop_dataset = SubCropDataset(
                dataset,
                self._structure_masker,
                self._structure_transformer,
                crop_size,
                indices,
            )
            sampler = DistributedSampler(
                dataset=crop_dataset,
                shuffle=train,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
            )
            dataloader = ResumableDataLoader(
                crop_dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=train,
                num_workers=int(self._num_workers / len(self._crop_sizes)),
                persistent_workers=self._persistent_workers,
            )

            dataloaders.append(dataloader)

        return CombinedDataLoader(dataloaders)

    def setup(self, stage: Any = None, index: int | None = None):
        with h5py.File(self._file_path, "r") as file:
            train_paths = list(file["train"].keys())
            val_paths = list(file["validation"].keys())
            test_paths = list(file["test"].keys())

        if index is not None:
            train_paths = [train_paths[index]]
            val_paths = [val_paths[index]]
            test_paths = [test_paths[index]]

        if stage == "fit" or stage is None:
            train_datasets = [
                MinecraftDataset(self._file_path, "train", path)
                for path in tqdm(train_paths, desc="Creating training datasets")
                if path not in self._separate_validation_datasets
            ]
            self._train_dataset = ConcatDataset(train_datasets)

            combined_val_datasets = [
                MinecraftDataset(self._file_path, "validation", path)
                for path in tqdm(val_paths, desc="Creating validation datasets")
                if path not in self._separate_validation_datasets
            ]
            self._val_datasets["combined"] = ConcatDataset(combined_val_datasets)
            self._val_assigned_sizes["combined"] = None
            self._val_dataloaders["combined"] = None

            # Collect separate validation datasets
            for separate_val_path in self._separate_validation_datasets:
                datasets = (
                    [
                        MinecraftDataset(self._file_path, "train", separate_val_path)
                        for path in train_paths
                        if path == separate_val_path
                    ]
                    + [
                        MinecraftDataset(
                            self._file_path, "validation", separate_val_path
                        )
                        for path in val_paths
                        if path == separate_val_path
                    ]
                    + [
                        MinecraftDataset(self._file_path, "test", separate_val_path)
                        for path in test_paths
                        if path == separate_val_path
                    ]
                )
                assert len(datasets) > 0, f"No datasets found for {separate_val_path}"
                self._val_datasets[separate_val_path] = ConcatDataset(datasets)
                self._val_assigned_sizes[separate_val_path] = None
                self._val_dataloaders[separate_val_path] = None

        if stage == "test" or stage is None:
            test_datasets = [
                MinecraftDataset(self._file_path, "test", path)
                for path in tqdm(test_paths, desc="Creating test datasets")
            ]
            self._test_dataset = ConcatDataset(test_datasets)

    def get_val_dataset_name(self, dataloader_idx: int):
        return list(self._val_dataloaders.keys())[dataloader_idx]

    def train_dataloader(self):
        if not self._train_loaded_from_checkpoint or self._train_dataloader is None:
            # Generate new sizes
            self._train_assigned_sizes = np.random.choice(
                list(self._crop_sizes.keys()), len(self._train_dataset)
            )

            # Create the dataloader
            self._train_dataloader = self._get_cropped_dataloader(
                self._train_dataset,
                train=True,
                assigned_sizes=self._train_assigned_sizes,
            )

        return self._train_dataloader

    def on_fit_start(self):
        self._train_loaded_from_checkpoint = False

    def val_dataloader(self):
        for key in self._val_dataloaders:
            if self._val_dataloaders[key] is None:
                # Shuffle the dataset
                self._val_shuffled_indices[key] = torch.randperm(
                    len(self._val_datasets[key])
                )
                dataset = torch.utils.data.Subset(
                    self._val_datasets[key], self._val_shuffled_indices[key]
                )

                # Generate sizes
                self._val_assigned_sizes[key] = np.random.choice(
                    list(self._crop_sizes.keys()), len(dataset)
                )

                # Create the dataloader
                self._val_dataloaders[key] = self._get_cropped_dataloader(
                    dataset,
                    train=False,
                    assigned_sizes=self._val_assigned_sizes[key],
                )

        return list(self._val_dataloaders.values())

    def test_dataloader(self):
        if self._test_dataloader is None:
            # Shuffle the dataset
            self._test_shuffled_indices = torch.randperm(len(self._test_dataset))
            dataset = torch.utils.data.Subset(
                self._test_dataset, self._test_shuffled_indices
            )

            # Generate sizes
            self._test_assigned_sizes = np.random.choice(
                list(self._crop_sizes.keys()), len(dataset)
            )

            # Create the dataloader
            self._test_dataloader = self._get_cropped_dataloader(
                dataset,
                train=False,
                assigned_sizes=self._test_assigned_sizes,
            )

        return self._test_dataloader
