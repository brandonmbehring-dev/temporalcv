#!/usr/bin/env python3
"""
M4/M5 Benchmark Runner for temporalcv.

Runs comprehensive model comparison on M4 (6 frequencies) and M5 datasets.

Usage:
    # Quick validation (100 series per frequency) - ~1 hour
    python scripts/run_benchmark.py --quick

    # Full benchmark (1000 series per frequency) - ~8-10 hours
    python scripts/run_benchmark.py --full

    # Single frequency
    python scripts/run_benchmark.py --dataset m4_monthly --sample 1000

    # Resume interrupted run
    python scripts/run_benchmark.py --resume benchmarks/results/run_abc123

    # M5 only (requires manual download from Kaggle)
    python scripts/run_benchmark.py --dataset m5 --m5-path ~/data/m5/

    # Baseline models only (fast)
    python scripts/run_benchmark.py --quick --models baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

M4_FREQUENCIES = ["yearly", "quarterly", "monthly", "weekly", "daily", "hourly"]

# M4 competition horizons
M4_HORIZONS = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "weekly": 13,
    "daily": 14,
    "hourly": 48,
}

# Seasonal periods by frequency
SEASON_LENGTHS = {
    "yearly": 1,
    "quarterly": 4,
    "monthly": 12,
    "weekly": 52,
    "daily": 7,
    "hourly": 24,
}

# Sample sizes
QUICK_SAMPLE = 100  # Per frequency
FULL_SAMPLE = 1000  # Per frequency


# =============================================================================
# Adapter Factory
# =============================================================================


def build_adapters(
    model_set: str = "all",
    season_length: int = 1,
) -> List[Any]:
    """
    Build list of adapters for benchmarking.

    Parameters
    ----------
    model_set : {"baseline", "statsforecast", "all"}
        Which models to include
    season_length : int
        Seasonal period for the dataset

    Returns
    -------
    list[ForecastAdapter]
        List of configured adapters
    """
    from temporalcv.compare import NaiveAdapter, SeasonalNaiveAdapter

    adapters = []

    # Baseline models (always fast)
    if model_set in ("baseline", "all"):
        adapters.append(NaiveAdapter())
        if season_length > 1:
            adapters.append(SeasonalNaiveAdapter(season_length=season_length))

    # Statsforecast models (wrapped for multi-series)
    if model_set in ("statsforecast", "all"):
        try:
            from temporalcv.compare.adapters import MultiSeriesAdapter, StatsforecastAdapter

            # Determine n_jobs (use half of CPUs for safety, max 8)
            import os
            n_jobs = min(os.cpu_count() // 2, 8) or 1

            # Core models (require MultiSeriesAdapter wrapper)
            adapters.extend([
                MultiSeriesAdapter(
                    StatsforecastAdapter("AutoARIMA", season_length=season_length),
                    n_jobs=n_jobs,
                ),
                MultiSeriesAdapter(
                    StatsforecastAdapter("AutoETS", season_length=season_length),
                    n_jobs=n_jobs,
                ),
                MultiSeriesAdapter(
                    StatsforecastAdapter("AutoTheta", season_length=season_length),
                    n_jobs=n_jobs,
                ),
            ])

            # Intermittent demand models (no seasonality)
            adapters.extend([
                MultiSeriesAdapter(
                    StatsforecastAdapter("CrostonClassic"),
                    n_jobs=n_jobs,
                ),
                MultiSeriesAdapter(
                    StatsforecastAdapter("ADIDA"),
                    n_jobs=n_jobs,
                ),
                MultiSeriesAdapter(
                    StatsforecastAdapter("IMAPA"),
                    n_jobs=n_jobs,
                ),
            ])

            # Simple models
            adapters.extend([
                MultiSeriesAdapter(
                    StatsforecastAdapter("HistoricAverage"),
                    n_jobs=n_jobs,
                ),
            ])

        except ImportError:
            logger.warning(
                "statsforecast not available. Install with: pip install statsforecast"
            )

    return adapters


# =============================================================================
# Dataset Loading
# =============================================================================


def load_datasets(
    dataset_filter: Optional[str],
    sample_size: int,
    m5_path: Optional[Path],
) -> List[Any]:
    """
    Load datasets for benchmarking.

    Parameters
    ----------
    dataset_filter : str or None
        Single dataset to load (e.g., "m4_monthly"), or None for all
    sample_size : int
        Number of series to sample per dataset
    m5_path : Path or None
        Path to M5 data directory

    Returns
    -------
    list[Dataset]
        Loaded datasets
    """
    from temporalcv.benchmarks import load_m4

    datasets = []

    # Parse dataset filter
    if dataset_filter:
        if dataset_filter.startswith("m4_"):
            freq = dataset_filter.replace("m4_", "")
            if freq in M4_FREQUENCIES:
                frequencies = [freq]
            else:
                raise ValueError(f"Unknown M4 frequency: {freq}")
        elif dataset_filter == "m5":
            frequencies = []  # M5 only
        else:
            raise ValueError(f"Unknown dataset: {dataset_filter}")
    else:
        frequencies = M4_FREQUENCIES

    # Load M4 datasets
    for freq in frequencies:
        try:
            dataset = load_m4(subset=freq, sample_size=sample_size)
            logger.info(
                "Loaded M4 %s: %d series, horizon=%d",
                freq,
                dataset.metadata.n_series,
                dataset.metadata.horizon,
            )
            datasets.append(dataset)
        except Exception as e:
            logger.warning("Failed to load M4 %s: %s", freq, e)

    # Load M5 if requested and path provided
    if (dataset_filter == "m5" or dataset_filter is None) and m5_path:
        try:
            from temporalcv.benchmarks import load_m5

            m5 = load_m5(path=str(m5_path), sample_size=sample_size, aggregate=True)
            logger.info("Loaded M5: %d series", m5.metadata.n_series)
            datasets.append(m5)
        except Exception as e:
            logger.warning("Failed to load M5: %s", e)

    return datasets


# =============================================================================
# Progress Reporting
# =============================================================================


def create_progress_callback(start_time: float) -> callable:
    """Create a progress callback that shows elapsed time."""

    def callback(current: int, total: int, dataset_name: str) -> None:
        elapsed = time.perf_counter() - start_time
        elapsed_min = elapsed / 60
        avg_per_dataset = elapsed / current if current > 0 else 0
        remaining = (total - current) * avg_per_dataset / 60

        print(
            f"[{current}/{total}] {dataset_name} | "
            f"Elapsed: {elapsed_min:.1f}m | "
            f"Remaining: ~{remaining:.1f}m"
        )

    return callback


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run M4/M5 benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help=f"Quick validation mode ({QUICK_SAMPLE} series/freq)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help=f"Full benchmark mode ({FULL_SAMPLE} series/freq)",
    )
    mode_group.add_argument(
        "--resume",
        type=Path,
        metavar="DIR",
        help="Resume from checkpoint directory",
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly", "m5"],
        help="Run single dataset only",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Override sample size",
    )
    parser.add_argument(
        "--m5-path",
        type=Path,
        help="Path to M5 data directory (from Kaggle)",
    )

    # Model selection
    parser.add_argument(
        "--models",
        choices=["baseline", "statsforecast", "all"],
        default="all",
        help="Model set to run (default: all)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine sample size
    if args.sample:
        sample_size = args.sample
    elif args.quick:
        sample_size = QUICK_SAMPLE
    elif args.full:
        sample_size = FULL_SAMPLE
    else:
        sample_size = QUICK_SAMPLE  # Default to quick

    # Set up output directory
    if args.resume:
        output_dir = args.resume
        checkpoint_dir = output_dir / "checkpoints"
    else:
        run_id = str(uuid.uuid4())[:8]
        output_dir = args.output_dir / f"run_{run_id}"
        checkpoint_dir = output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Configure file logging
    file_handler = logging.FileHandler(output_dir / "benchmark.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("M4/M5 Benchmark Suite")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)
    logger.info("Sample size: %d series/dataset", sample_size)
    logger.info("Model set: %s", args.models)

    # Load datasets
    logger.info("-" * 60)
    logger.info("Loading datasets...")
    datasets = load_datasets(args.dataset, sample_size, args.m5_path)

    if not datasets:
        logger.error("No datasets loaded!")
        return 1

    logger.info("Loaded %d datasets", len(datasets))

    # Import comparison functions
    from temporalcv.compare import run_benchmark_suite
    from temporalcv.compare.results import create_run_metadata, save_benchmark_results

    # Build adapters (using first dataset's frequency for season_length)
    # For proper per-frequency adapters, we'd need to run each dataset separately
    # For simplicity, using monthly seasonality as default
    adapters = build_adapters(args.models, season_length=12)
    logger.info("Using %d adapters: %s", len(adapters), [a.model_name for a in adapters])

    if not adapters:
        logger.error("No adapters available!")
        return 1

    # Run benchmark
    logger.info("-" * 60)
    logger.info("Running benchmark...")
    start_time = time.perf_counter()

    try:
        report = run_benchmark_suite(
            datasets=datasets,
            adapters=adapters,
            primary_metric="mae",
            include_dm_test=True,
            progress_callback=create_progress_callback(start_time),
            checkpoint_dir=checkpoint_dir,
        )
    except Exception as e:
        logger.exception("Benchmark failed: %s", e)
        return 1

    elapsed = time.perf_counter() - start_time
    logger.info("Benchmark completed in %.1f minutes", elapsed / 60)

    # Save results
    logger.info("-" * 60)
    logger.info("Saving results...")

    metadata = create_run_metadata(
        models=[a.model_name for a in adapters],
        datasets=[d.metadata.name for d in datasets],
    )
    metadata["total_runtime_seconds"] = elapsed
    metadata["sample_size"] = sample_size
    metadata["model_set"] = args.models

    save_benchmark_results(report, output_dir / "results.json", metadata=metadata)

    # Generate markdown
    markdown = report.to_markdown()
    (output_dir / "results.md").write_text(markdown)

    # Print summary
    logger.info("-" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nDatasets evaluated: {len(report.results)}")
    print(f"Models compared: {len(adapters)}")
    print(f"Total runtime: {elapsed/60:.1f} minutes")

    if report.summary.get("wins_by_model"):
        print("\nModel Wins:")
        for model, wins in sorted(report.summary["wins_by_model"].items(), key=lambda x: -x[1]):
            print(f"  {model}: {wins}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - results.json (structured data)")
    print(f"  - results.md (markdown tables)")
    print(f"  - benchmark.log (run log)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
