from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    name: str
    scores: List[float]

    @property
    def average(self) -> float:
        return np.mean(self.scores)

    @property
    def std_dev(self) -> float:
        return np.std(self.scores)

    def __str__(self) -> str:
        return f"{self.name}: {self.average:.2%} (±{self.std_dev:.2%})"


class BaseBenchmark(ABC):
    def __init__(
        self,
        name: str,
        save_debug_schematics=False,
        debug_output_dir="debug_schematics",
    ):
        self.name = name
        self.save_debug_schematics = save_debug_schematics
        self.debug_output_dir = Path(debug_output_dir)
        if save_debug_schematics:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)

    def save_debug_schematic(self, schematic, filename):
        if self.save_debug_schematics:
            output_path = self.debug_output_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            schematic.save_to_file(output_path, 2)

    @abstractmethod
    def get_model_inputs(self, seed: int) -> Tuple[torch.Tensor, Any]:
        """
        Prepare inputs for model inference.
        Returns:
            model_input: Tensor for model input
            comparison_data: Any data needed for comparing results
        """
        raise NotImplementedError("Must be implemented by subclass")

    @abstractmethod
    def compare_model_output(
        self, model_output: torch.Tensor, comparison_data: Any, seed: int
    ) -> float:
        """
        Compare model output with expected results.
        Returns:
            score: float between 0 and 1
        """
        raise NotImplementedError("Must be implemented by subclass")

    def run(
        self, model, num_runs, base_seed, batch_size, show_progress
    ) -> BenchmarkResult:
        """Run multiple tests with batched model inference"""
        # Gather all inputs
        all_inputs = []
        all_comparison_data = []

        for seed in tqdm(
            range(base_seed, base_seed + num_runs),
            desc=f"{self.name} - Preparing",
            leave=False,
            disable=not show_progress,
        ):
            model_input, comparison_data = self.get_model_inputs(seed)
            all_inputs.append(model_input)
            all_comparison_data.append(comparison_data)

        # Process in batches
        scores = []
        num_batches = (num_runs + batch_size - 1) // batch_size  # Ceiling division
        for i in tqdm(
            range(0, num_runs, batch_size),
            desc=f"{self.name} - Processing",
            total=num_batches,
            leave=False,
            disable=not show_progress,
        ):
            batch_inputs = all_inputs[i : i + batch_size]
            batch_comparison = all_comparison_data[i : i + batch_size]

            # Run model on batch
            model_input_batch = torch.stack(batch_inputs)
            model_input_batch = model_input_batch.unsqueeze(1)
            model_outputs = model.one_shot_inference(model_input_batch, 0.7, True)

            # Compare batch results
            for j, output in enumerate(model_outputs):
                score = self.compare_model_output(
                    output, batch_comparison[j], base_seed + i + j
                )
                scores.append(score)

        return BenchmarkResult(self.name, scores)
