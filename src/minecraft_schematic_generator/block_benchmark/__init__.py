from .bed_benchmark import BedBenchmark
from .benchmark_registry import BenchmarkCategory, BenchmarkRegistry
from .door_benchmark import DoorBenchmark
from .portal_benchmark import PortalBenchmark
from .tall_plant_benchmark import TallPlantBenchmark

# Import other benchmark types as they're created


def create_default_registry(save_debug_schematics) -> BenchmarkRegistry:
    """Create and populate a registry with all standard benchmarks"""
    registry = BenchmarkRegistry()

    # Register Door benchmarks
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        DoorBenchmark("Doors", save_debug_schematics),
    )

    # Register Bed benchmarks
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        BedBenchmark("Beds", save_debug_schematics),
    )

    # Register Tall Plant benchmarks
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        TallPlantBenchmark("Tall Plants", save_debug_schematics),
    )

    # Register Portal benchmarks
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        PortalBenchmark("Portals", save_debug_schematics),
    )

    return registry


def run_benchmark(model, num_runs=1, save_debug_schematics=False, base_seed=0):
    """
    Run the full benchmark suite with all default tests.

    Args:
        model: The model to benchmark
        num_runs: Number of times to run each test (default: 1)
        save_debug_schematics: Whether to save debug schematics (default: False)
        base_seed: Starting seed for random number generation (default: 0)

    Returns:
        BenchmarkSuite containing all results
    """
    registry = create_default_registry(save_debug_schematics)
    results = registry.run_all(model, num_runs, base_seed)
    return results


def run_category(
    category: BenchmarkCategory,
    model,
    num_runs=1,
    save_debug_schematics=False,
    base_seed=0,
):
    """
    Run benchmarks for a specific category.

    Args:
        category: BenchmarkCategory to run
        model: The model to benchmark
        num_runs: Number of times to run each test (default: 1)
        save_debug_schematics: Whether to save debug schematics (default: False)
        base_seed: Starting seed for random number generation (default: 0)

    Returns:
        CategoryResult containing category results
    """
    registry = create_default_registry()
    results = registry.run_category(category, model, num_runs, base_seed)
    return results
