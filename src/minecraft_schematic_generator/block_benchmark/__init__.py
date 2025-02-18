from minecraft_schematic_generator.converter import (
    BlockTokenConverter,
    SchematicArrayConverter,
)

from .benchmark_registry import BenchmarkCategory, BenchmarkRegistry
from .benchmarks.bed_benchmark import BedBenchmark
from .benchmarks.door_benchmark import DoorBenchmark
from .benchmarks.pattern_benchmark import PatternBenchmark, PatternType
from .benchmarks.portal_benchmark import PortalBenchmark
from .benchmarks.redstone_power_benchmark import (
    RedstoneComponentType,
    RedstonePowerBenchmark,
)
from .benchmarks.stairs_benchmark import StairsBenchmark
from .benchmarks.tall_plant_benchmark import TallPlantBenchmark


def create_default_registry(
    block_token_converter: BlockTokenConverter,
    schematic_size: int,
    save_debug_schematics: bool,
) -> BenchmarkRegistry:
    """Create and populate a registry with all standard benchmarks"""
    registry = BenchmarkRegistry()
    schematic_array_converter = SchematicArrayConverter(block_token_converter)

    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        DoorBenchmark(
            "doors", schematic_array_converter, schematic_size, save_debug_schematics
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        BedBenchmark(
            "beds", schematic_array_converter, schematic_size, save_debug_schematics
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        TallPlantBenchmark(
            "tall_plants",
            schematic_array_converter,
            schematic_size,
            save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.STRUCTURES,
        PortalBenchmark(
            "portals", schematic_array_converter, schematic_size, save_debug_schematics
        ),
    )

    registry.register_benchmark(
        BenchmarkCategory.PATTERNS,
        PatternBenchmark(
            "plane",
            schematic_array_converter,
            schematic_size,
            PatternType.PLANE,
            save_debug_schematics=save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.PATTERNS,
        PatternBenchmark(
            "checkerboard_plane",
            schematic_array_converter,
            schematic_size,
            PatternType.PLANE,
            checkerboard=True,
            save_debug_schematics=save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.PATTERNS,
        PatternBenchmark(
            "cross",
            schematic_array_converter,
            schematic_size,
            PatternType.CROSS,
            save_debug_schematics=save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.PATTERNS,
        PatternBenchmark(
            "checkerboard_cross",
            schematic_array_converter,
            schematic_size,
            PatternType.CROSS,
            checkerboard=True,
            save_debug_schematics=save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.PATTERNS,
        StairsBenchmark(
            "stairs",
            schematic_array_converter,
            schematic_size,
            min_width=0,
            max_width=3,
            save_debug_schematics=save_debug_schematics,
        ),
    )

    registry.register_benchmark(
        BenchmarkCategory.REDSTONE,
        RedstonePowerBenchmark(
            "redstone_lamps",
            schematic_array_converter,
            schematic_size,
            RedstoneComponentType.LAMP,
            save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.REDSTONE,
        RedstonePowerBenchmark(
            "iron_doors",
            schematic_array_converter,
            schematic_size,
            RedstoneComponentType.DOOR,
            save_debug_schematics,
        ),
    )
    registry.register_benchmark(
        BenchmarkCategory.REDSTONE,
        RedstonePowerBenchmark(
            "pistons",
            schematic_array_converter,
            schematic_size,
            RedstoneComponentType.PISTON,
            save_debug_schematics,
        ),
    )

    return registry


def run_benchmark(
    model,
    block_token_converter: BlockTokenConverter,
    schematic_size: int,
    num_runs: int = 1,
    save_debug_schematics: bool = False,
    base_seed: int = 0,
    batch_size: int = 1,
    show_progress: bool = True,
):
    """
    Run the full benchmark suite with all default tests.

    Args:
        model: The model to benchmark
        num_runs: Number of times to run each test (default: 1)
        save_debug_schematics: Whether to save debug schematics (default: False)
        base_seed: Starting seed for random number generation (default: 0)
        batch_size: Batch size for model inference (default: 1)
        show_progress: Whether to show progress bars (default: True)

    Returns:
        BenchmarkSuite containing all results
    """
    registry = create_default_registry(
        block_token_converter, schematic_size, save_debug_schematics
    )
    results = registry.run_all(model, num_runs, base_seed, batch_size, show_progress)
    return results


def run_category(
    category: BenchmarkCategory,
    model,
    num_runs=1,
    save_debug_schematics=False,
    base_seed=0,
    batch_size=1,
):
    """
    Run benchmarks for a specific category.

    Args:
        category: BenchmarkCategory to run
        model: The model to benchmark
        num_runs: Number of times to run each test (default: 1)
        save_debug_schematics: Whether to save debug schematics (default: False)
        base_seed: Starting seed for random number generation (default: 0)
        batch_size: Batch size for model inference (default: 1)

    Returns:
        CategoryResult containing category results
    """
    registry = create_default_registry(save_debug_schematics)
    results = registry.run_category(category, model, num_runs, base_seed, batch_size)
    return results
