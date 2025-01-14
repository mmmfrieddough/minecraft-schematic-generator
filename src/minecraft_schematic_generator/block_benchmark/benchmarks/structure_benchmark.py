from abc import abstractmethod
from typing import Set, Tuple

import numpy as np
import torch
from schempy import Schematic

from minecraft_schematic_generator.converter import SchematicArrayConverter

from .base_benchmark import BaseBenchmark


class StructureBenchmark(BaseBenchmark):
    """Base class for benchmarks that test structure generation"""

    SCHEMATIC_SIZE = 11
    SCHEMATIC_MIDDLE = SCHEMATIC_SIZE // 2

    def __init__(
        self,
        name: str,
        save_debug_schematics=False,
        debug_output_dir="debug_schematics",
    ):
        super().__init__(name, save_debug_schematics, debug_output_dir)
        self.schematic_array_converter = SchematicArrayConverter()

    def create_schematics(self):
        """Create empty complete and partial schematics"""
        complete_schematic = Schematic(
            self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE
        )
        partial_schematic = Schematic(
            self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE, self.SCHEMATIC_SIZE
        )
        return complete_schematic, partial_schematic

    def convert_to_model_input(self, partial_schematic):
        """Convert schematic to model input format"""
        partial_structure = self.schematic_array_converter.schematic_to_array(
            partial_schematic
        )
        partial_structure = (
            torch.from_numpy(partial_structure.astype(np.int64)).long().contiguous()
        )
        # Remove air blocks (ID 1)
        partial_structure[partial_structure == 1] = 0
        return partial_structure

    def compare_structures(
        self, complete_structure, generated_structure, removed_positions
    ):
        """Compare original and generated structures at removed positions"""
        correct_predictions = 0
        total_predictions = len(removed_positions)

        for pos in removed_positions:
            original_value = complete_structure[pos[2], pos[1], pos[0]]
            generated_value = generated_structure[pos[2], pos[1], pos[0]]

            if original_value == generated_value:
                correct_predictions += 1

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        return accuracy

    def get_model_inputs(
        self, seed: int
    ) -> Tuple[torch.Tensor, Tuple[np.ndarray, Set[Tuple[int, int, int]]]]:
        """
        Prepare structure inputs for model inference.
        Returns:
            model_input: Tensor of partial structure
            comparison_data: Tuple of (complete_structure, removed_positions)
        """
        # Create schematics
        complete_schematic, partial_schematic = self.create_schematics()

        # Build the structure
        removed_positions = self.build_structure(
            complete_schematic, partial_schematic, seed
        )

        # Save debug schematics
        self.save_debug_schematic(
            complete_schematic, f"{self.name}/complete_{seed}.schem"
        )
        self.save_debug_schematic(
            partial_schematic, f"{self.name}/partial_{seed}.schem"
        )

        # Convert to model input
        partial_structure = self.convert_to_model_input(partial_schematic)
        complete_structure = self.schematic_array_converter.schematic_to_array(
            complete_schematic
        )

        return partial_structure, (complete_structure, removed_positions)

    def compare_model_output(
        self,
        model_output: torch.Tensor,
        comparison_data: Tuple[np.ndarray, Set[Tuple[int, int, int]]],
        seed: int,
    ) -> float:
        """Compare generated structure with original"""
        complete_structure, removed_positions = comparison_data

        # Save generated result
        generated_schematic = self.schematic_array_converter.array_to_schematic(
            model_output
        )
        self.save_debug_schematic(
            generated_schematic, f"{self.name}/generated_{seed}.schem"
        )

        return self.compare_structures(
            complete_structure, model_output, removed_positions
        )

    @abstractmethod
    def build_structure(
        self, complete_schematic: Schematic, partial_schematic: Schematic, seed: int
    ) -> set:
        """
        Build the complete and partial structures for testing.

        Args:
            complete_schematic: The schematic to build the complete structure in
            partial_schematic: The schematic to build the partial structure in
            seed: Random seed for structure generation

        Returns:
            set of positions where blocks were removed from the partial structure
        """
        pass
