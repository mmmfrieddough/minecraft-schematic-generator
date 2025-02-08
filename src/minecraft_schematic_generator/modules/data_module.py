import json
from typing import Any, List

import h5py
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    DictBlockTokenMapper,
)
from minecraft_schematic_generator.model import MinecraftDataset, ResumableDataLoader


class MinecraftDataModule(LightningDataModule):
    def __init__(
        self,
        file_path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        combine_datasets: bool = False,
        separate_validation_datasets: List[str] = [],
    ):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.num_workers = num_workers
        self.persistent_workers = num_workers > 0
        self.combine_datasets = combine_datasets
        self.separate_validation_datasets = separate_validation_datasets
        self.val_dataset_names = {}
        self._rng_state = None

        with h5py.File(self.file_path, "r") as file:
            # Load the mapping
            if "mapping" in file:
                mapping_group = file["mapping"]
                mapping_str = mapping_group["block_to_token"][()]
                mapping_json = json.loads(mapping_str)
                self.block_str_mapping = dict(mapping_json)
            else:
                raise ValueError("Mapping not found in HDF5 file.")

            block_token_mapper = DictBlockTokenMapper(self.block_str_mapping)
            self.block_token_converter = BlockTokenConverter(block_token_mapper)

    def state_dict(self) -> dict:
        """Save datamodule state."""
        return {
            "rng_state": torch.get_rng_state(),
            "val_dataset_names": self.val_dataset_names,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load datamodule state."""
        self._rng_state = state_dict.get("rng_state")
        self.val_dataset_names = state_dict.get("val_dataset_names", {})
        if self._rng_state is not None:
            torch.set_rng_state(self._rng_state)

    def get_val_dataset_name(self, dataloader_idx: int):
        return self.val_dataset_names[dataloader_idx]

    def setup(self, stage: Any = None, index: int | None = None):
        with h5py.File(self.file_path, "r") as file:
            if index is not None:
                train_keys = [list(file["train"].keys())[index]]
                val_keys = [list(file["validation"].keys())[index]]
                test_keys = [list(file["test"].keys())[index]]
            else:
                train_keys = file["train"].keys()
                val_keys = file["validation"].keys()
                test_keys = file["test"].keys()

            self.train_datasets = [
                (
                    generator_type,
                    MinecraftDataset(
                        self.file_path,
                        "train",
                        generator_type,
                        self.block_token_converter,
                    ),
                )
                for generator_type in tqdm(train_keys, desc="Loading training datasets")
            ]
            self.val_datasets = [
                (
                    generator_type,
                    MinecraftDataset(
                        self.file_path,
                        "validation",
                        generator_type,
                        self.block_token_converter,
                    ),
                )
                for generator_type in tqdm(val_keys, desc="Loading validation datasets")
            ]
            self.test_datasets = [
                (
                    generator_type,
                    MinecraftDataset(
                        self.file_path,
                        "test",
                        generator_type,
                        self.block_token_converter,
                    ),
                )
                for generator_type in tqdm(test_keys, desc="Loading test datasets")
            ]

    def train_dataloader(self):
        train_datasets = ConcatDataset([dataset for _, dataset in self.train_datasets])
        assert len(train_datasets) > 0, "Training DataLoader is empty."

        if self._rng_state is not None:
            torch.set_rng_state(self._rng_state)

        return ResumableDataLoader(
            train_datasets,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        if self.combine_datasets:
            # Filter datasets to be combined and kept separate based on `separate_validation_datasets`
            combined_val_datasets = [
                dataset
                for generator_type, dataset in self.val_datasets
                if generator_type not in self.separate_validation_datasets
            ]
            separate_val_datasets = [
                dataset
                for generator_type, dataset in self.val_datasets
                if generator_type in self.separate_validation_datasets
            ]
            separate_val_dataset_names = [
                generator_type
                for generator_type, dataset in self.val_datasets
                if generator_type in self.separate_validation_datasets
            ]

            # Create a DataLoader for the combined datasets
            if combined_val_datasets:
                combined_dataset = ConcatDataset(combined_val_datasets)

                # Shuffle the combined dataset once
                indices = torch.randperm(len(combined_dataset))
                combined_dataset = torch.utils.data.Subset(combined_dataset, indices)

                combined_val_loader = ResumableDataLoader(
                    combined_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers,
                )
                val_loaders = [combined_val_loader]
                self.val_dataset_names[0] = "combined"
            else:
                val_loaders = []

            # Add separate DataLoaders for datasets not to be combined
            for i in range(len(separate_val_datasets)):
                dataloader = ResumableDataLoader(
                    separate_val_datasets[i],
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers,
                )
                val_loaders.append(dataloader)
                self.val_dataset_names[i + 1] = separate_val_dataset_names[i]
        else:
            val_loaders = [
                ResumableDataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers,
                )
                for _, dataset in self.val_datasets
            ]
        for i, loader in enumerate(val_loaders):
            assert len(loader) > 0, f"Validation DataLoader at index {i} is empty."
        return val_loaders

    def test_dataloader(self):
        if self.combine_datasets:
            test_datasets = ConcatDataset(
                [dataset for _, dataset in self.test_datasets]
            )
            test_loader = ResumableDataLoader(
                test_datasets,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
            test_loaders = [test_loader]
        else:
            test_loaders = [
                ResumableDataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers,
                )
                for _, dataset in self.test_datasets
            ]
        for i, loader in enumerate(test_loaders):
            assert len(loader) > 0, f"Test DataLoader at index {i} is empty."
        return test_loaders
