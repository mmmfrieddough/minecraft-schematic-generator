from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from .base_benchmark import BaseBenchmark, BenchmarkResult


@dataclass
class CategoryResult:
    name: str
    benchmark_results: List[BenchmarkResult]

    @property
    def average(self) -> float:
        all_scores = [
            score for result in self.benchmark_results for score in result.scores
        ]
        return np.mean(all_scores)

    @property
    def std_dev(self) -> float:
        all_scores = [
            score for result in self.benchmark_results for score in result.scores
        ]
        return np.std(all_scores)

    def __str__(self) -> str:
        result = f"\n{self.name} Category: {self.average:.2%} (±{self.std_dev:.2%})\n"
        for benchmark_result in self.benchmark_results:
            result += f"  {benchmark_result}\n"
        return result


@dataclass
class BenchmarkSuite:
    category_results: List[CategoryResult]

    @property
    def average(self) -> float:
        all_scores = [
            score
            for category in self.category_results
            for result in category.benchmark_results
            for score in result.scores
        ]
        return np.mean(all_scores)

    def __str__(self) -> str:
        result = f"\nOverall Average: {self.average:.2%}\n"
        for category in self.category_results:
            result += f"{category}"
        return result


class BenchmarkCategory(Enum):
    STRUCTURES = "Structures"
    CONNECTIONS = "Connections"
    PATTERNS = "Patterns"
    REDSTONE = "Redstone"


class BenchmarkRegistry:
    def __init__(self):
        self.benchmarks: Dict[BenchmarkCategory, List[BaseBenchmark]] = {
            category: [] for category in BenchmarkCategory
        }

    def register_benchmark(self, category: BenchmarkCategory, benchmark: BaseBenchmark):
        self.benchmarks[category].append(benchmark)

    def run_category(
        self, category: BenchmarkCategory, model, num_runs=1, base_seed=0
    ) -> CategoryResult:
        results = []
        with tqdm(
            total=len(self.benchmarks[category]),
            desc=f"Category: {category.value}",
            leave=False,
        ) as pbar:
            for benchmark in self.benchmarks[category]:
                results.append(benchmark.run(model, num_runs, base_seed))
                pbar.update(1)
        return CategoryResult(category.value, results)

    def run_all(self, model, num_runs=1, base_seed=0) -> BenchmarkSuite:
        category_results = []
        with tqdm(
            total=len(BenchmarkCategory),
            desc="Overall Progress",
            position=0,
            leave=True,
        ) as pbar:
            for category in BenchmarkCategory:
                category_results.append(
                    self.run_category(category, model, num_runs, base_seed)
                )
                pbar.update(1)

        return BenchmarkSuite(category_results)
