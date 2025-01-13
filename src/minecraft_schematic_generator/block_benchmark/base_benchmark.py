from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
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

    @abstractmethod
    def run_single_test(self, model, seed: int) -> float:
        """Run a single instance of the test"""
        pass

    def save_debug_schematic(self, schematic, filename):
        if self.save_debug_schematics:
            output_path = self.debug_output_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            schematic.save_to_file(output_path, 2)

    def run(self, model, num_runs=1, base_seed=0) -> BenchmarkResult:
        scores = []
        for seed in tqdm(
            range(base_seed, base_seed + num_runs), desc=f"{self.name}", leave=False
        ):
            score = self.run_single_test(model, seed)
            scores.append(score)

        return BenchmarkResult(self.name, scores)
