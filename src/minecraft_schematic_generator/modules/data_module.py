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
from minecraft_schematic_generator.model.structure_masker import StructureMasker


class MinecraftDataModule(LightningDataModule):
    def __init__(
        self,
        file_path: str,
        structure_masker: StructureMasker,
        batch_size: int = 32,
        num_workers: int = 0,
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

            self._structure_masker = structure_masker
            block_token_mapper = DictBlockTokenMapper(self.block_str_mapping)
            block_token_converter = BlockTokenConverter(block_token_mapper)
            self._block_token_converter = block_token_converter
            self._structure_masker.setup(block_token_converter)

    def get_block_token_converter(self):
        return self._block_token_converter

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
                train_keys = list(file["train"].keys())
                val_keys = list(file["validation"].keys())
                test_keys = list(file["test"].keys())

            # Collect separate validation datasets from all splits
            separate_datasets = []
            for generator_type in self.separate_validation_datasets:
                # Check each split for the generator type
                for split in ["train", "validation", "test"]:
                    if generator_type in file[split]:
                        separate_datasets.append(
                            (
                                generator_type,
                                MinecraftDataset(
                                    self.file_path,
                                    split,
                                    generator_type,
                                    self._structure_masker,
                                ),
                            )
                        )
            if self.separate_validation_datasets and not separate_datasets:
                raise ValueError("Separate validation datasets not found")

            # Filter out separate_validation_datasets from regular splits
            train_keys = [
                k for k in train_keys if k not in self.separate_validation_datasets
            ]
            val_keys = [
                k for k in val_keys if k not in self.separate_validation_datasets
            ]
            test_keys = [
                k for k in test_keys if k not in self.separate_validation_datasets
            ]

            self.train_datasets = [
                (
                    generator_type,
                    MinecraftDataset(
                        self.file_path,
                        "train",
                        generator_type,
                        self._structure_masker,
                    ),
                )
                for generator_type in tqdm(train_keys, desc="Loading training datasets")
            ]

            # Regular validation datasets plus the separate ones collected from all splits
            self.val_datasets = [
                (
                    generator_type,
                    MinecraftDataset(
                        self.file_path,
                        "validation",
                        generator_type,
                        self._structure_masker,
                    ),
                )
                for generator_type in tqdm(val_keys, desc="Loading validation datasets")
            ] + separate_datasets

            self.test_datasets = [
                (
                    generator_type,
                    MinecraftDataset(
                        self.file_path,
                        "test",
                        generator_type,
                        self._structure_masker,
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
        # Create a DataLoader for the combined datasets
        combined_val_datasets = [
            dataset
            for generator_type, dataset in self.val_datasets
            if generator_type not in self.separate_validation_datasets
        ]
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
        for i, generator_type in enumerate(self.separate_validation_datasets):
            datasets = [
                dataset
                for dataset_name, dataset in self.val_datasets
                if dataset_name == generator_type
            ]
            combined_dataset = ConcatDataset(datasets)
            dataloader = ResumableDataLoader(
                combined_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
            val_loaders.append(dataloader)
            self.val_dataset_names[i + 1] = generator_type

        for i, loader in enumerate(val_loaders):
            assert len(loader) > 0, f"Validation DataLoader at index {i} is empty."
        return val_loaders

    def test_dataloader(self):
        test_datasets = ConcatDataset([dataset for _, dataset in self.test_datasets])
        test_loader = ResumableDataLoader(
            test_datasets,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        test_loaders = [test_loader]

        for i, loader in enumerate(test_loaders):
            assert len(loader) > 0, f"Test DataLoader at index {i} is empty."
        return test_loaders
