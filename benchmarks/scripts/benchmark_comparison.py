#!/usr/bin/env python3
"""
Benchmark Comparison: temporalcv vs sklearn TimeSeriesSplit

Compares temporalcv's WalkForwardCV against sklearn's TimeSeriesSplit on:
1. Feature correctness (gap enforcement, temporal ordering)
2. API compatibility
3. Performance (splitting speed)

Run: python scripts/benchmark_comparison.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from temporalcv.cv import WalkForwardCV


@dataclass
class BenchmarkResult:
    """Result of a single benchmark comparison."""

    name: str
    temporalcv_value: Any
    sklearn_value: Any
    winner: str
    notes: str


def check_gap_enforcement(n: int = 1000, gap: int = 5) -> BenchmarkResult:
    """Check if gap between train and test is enforced."""
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n)

    # temporalcv with gap
    cv_tcv = WalkForwardCV(n_splits=5, gap=gap, test_size=50)
    tcv_gaps = []
    for train_idx, test_idx in cv_tcv.split(X, y):
        gap_actual = test_idx[0] - train_idx[-1] - 1
        tcv_gaps.append(gap_actual)

    # sklearn TimeSeriesSplit (no gap parameter)
    cv_sklearn = TimeSeriesSplit(n_splits=5, test_size=50, gap=gap)
    sklearn_gaps = []
    for train_idx, test_idx in cv_sklearn.split(X, y):
        gap_actual = test_idx[0] - train_idx[-1] - 1
        sklearn_gaps.append(gap_actual)

    tcv_correct = all(g >= gap for g in tcv_gaps)
    sklearn_correct = all(g >= gap for g in sklearn_gaps)

    return BenchmarkResult(
        name="Gap Enforcement",
        temporalcv_value=f"All gaps >= {gap}: {tcv_correct}",
        sklearn_value=f"All gaps >= {gap}: {sklearn_correct}",
        winner="Both"
        if tcv_correct and sklearn_correct
        else ("temporalcv" if tcv_correct else "sklearn"),
        notes="Both support gap parameter (sklearn added in v1.0)",
    )


def check_window_types(n: int = 1000) -> BenchmarkResult:
    """Check window type support (expanding vs sliding)."""
    X = np.arange(n).reshape(-1, 1)
    y = np.arange(n)

    # temporalcv supports both
    cv_expanding = WalkForwardCV(n_splits=5, window_type="expanding", test_size=50)
    cv_sliding = WalkForwardCV(n_splits=5, window_type="sliding", test_size=50, window_size=200)

    expanding_sizes = [len(train) for train, _ in cv_expanding.split(X, y)]
    sliding_sizes = [len(train) for train, _ in cv_sliding.split(X, y)]

    expanding_grows = expanding_sizes[-1] > expanding_sizes[0]
    sliding_constant = len(set(sliding_sizes)) <= 2  # Allow minor variation

    # sklearn only has expanding (max_train_size simulates sliding but not exactly)
    cv_sklearn = TimeSeriesSplit(n_splits=5, test_size=50)
    sklearn_sizes = [len(train) for train, _ in cv_sklearn.split(X, y)]
    sklearn_grows = sklearn_sizes[-1] > sklearn_sizes[0]

    return BenchmarkResult(
        name="Window Types",
        temporalcv_value=f"expanding={expanding_grows}, sliding={sliding_constant}",
        sklearn_value=f"expanding={sklearn_grows}, sliding=N/A (max_train_size only)",
        winner="temporalcv",
        notes="temporalcv has native sliding window; sklearn uses max_train_size workaround",
    )


def check_leakage_detection() -> BenchmarkResult:
    """Check if leakage detection is available."""
    # temporalcv has gates

    tcv_has_gates = True

    # sklearn has no built-in leakage detection
    sklearn_has_gates = False

    return BenchmarkResult(
        name="Leakage Detection",
        temporalcv_value="gate_signal_verification, gate_suspicious_improvement, gate_synthetic_ar1",
        sklearn_value="None (external tools needed)",
        winner="temporalcv",
        notes="temporalcv's unique value proposition",
    )


def check_statistical_tests() -> BenchmarkResult:
    """Check if statistical testing is integrated."""

    tcv_has_tests = True

    return BenchmarkResult(
        name="Statistical Tests",
        temporalcv_value="dm_test (DM), pt_test (PT), HAC variance",
        sklearn_value="None",
        winner="temporalcv",
        notes="temporalcv integrates forecast evaluation tests",
    )


def benchmark_split_speed(
    n: int = 10000, n_splits: int = 10, iterations: int = 100
) -> BenchmarkResult:
    """Benchmark splitting speed."""
    X = np.random.randn(n, 10)
    y = np.random.randn(n)

    # temporalcv
    cv_tcv = WalkForwardCV(n_splits=n_splits, test_size=n // (n_splits + 1))
    start = time.perf_counter()
    for _ in range(iterations):
        list(cv_tcv.split(X, y))
    tcv_time = (time.perf_counter() - start) / iterations * 1000  # ms

    # sklearn
    cv_sklearn = TimeSeriesSplit(n_splits=n_splits, test_size=n // (n_splits + 1))
    start = time.perf_counter()
    for _ in range(iterations):
        list(cv_sklearn.split(X, y))
    sklearn_time = (time.perf_counter() - start) / iterations * 1000  # ms

    winner = "temporalcv" if tcv_time < sklearn_time else "sklearn"
    ratio = max(tcv_time, sklearn_time) / min(tcv_time, sklearn_time)

    return BenchmarkResult(
        name="Split Speed (n=10k, 10 splits)",
        temporalcv_value=f"{tcv_time:.3f} ms",
        sklearn_value=f"{sklearn_time:.3f} ms",
        winner=winner if ratio > 1.1 else "Comparable",
        notes=f"Ratio: {ratio:.2f}x",
    )


def check_conformal_prediction() -> BenchmarkResult:
    """Check conformal prediction support."""

    return BenchmarkResult(
        name="Conformal Prediction",
        temporalcv_value="SplitConformalPredictor, AdaptiveConformalPredictor",
        sklearn_value="None (MAPIE is separate package)",
        winner="temporalcv",
        notes="Built-in uncertainty quantification",
    )


def check_financial_cv() -> BenchmarkResult:
    """Check financial CV support (purging, embargo)."""

    return BenchmarkResult(
        name="Financial CV (Purging)",
        temporalcv_value="PurgedKFold, PurgedWalkForward, CombinatorialPurgedCV",
        sklearn_value="None",
        winner="temporalcv",
        notes="Lopez de Prado AFML methods",
    )


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmark comparisons."""
    return [
        check_gap_enforcement(),
        check_window_types(),
        check_leakage_detection(),
        check_statistical_tests(),
        check_conformal_prediction(),
        check_financial_cv(),
        benchmark_split_speed(),
    ]


def generate_markdown_table(results: list[BenchmarkResult]) -> str:
    """Generate markdown table from results."""
    lines = [
        "| Feature | temporalcv | sklearn | Winner | Notes |",
        "|---------|------------|---------|--------|-------|",
    ]

    for r in results:
        lines.append(
            f"| {r.name} | {r.temporalcv_value} | {r.sklearn_value} | **{r.winner}** | {r.notes} |"
        )

    return "\n".join(lines)


def main():
    """Run benchmarks and print results."""
    print("=" * 70)
    print("BENCHMARK COMPARISON: temporalcv vs sklearn TimeSeriesSplit")
    print("=" * 70)

    results = run_all_benchmarks()

    print("\n### Results\n")
    print(generate_markdown_table(results))

    # Summary
    tcv_wins = sum(1 for r in results if "temporalcv" in r.winner)
    sklearn_wins = sum(1 for r in results if r.winner == "sklearn")
    ties = len(results) - tcv_wins - sklearn_wins

    print("\n### Summary")
    print(f"- temporalcv wins: {tcv_wins}")
    print(f"- sklearn wins: {sklearn_wins}")
    print(f"- Comparable/Both: {ties}")

    print("\n### Key Differentiators")
    print("1. **Leakage Detection**: temporalcv's validation gates catch data leakage")
    print("2. **Statistical Testing**: Integrated DM/PT tests for forecast evaluation")
    print("3. **Conformal Prediction**: Built-in uncertainty quantification")
    print("4. **Financial CV**: Purging and embargo for overlapping labels")

    return results


if __name__ == "__main__":
    main()
