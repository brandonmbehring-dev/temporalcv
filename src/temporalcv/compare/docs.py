"""
Documentation Generation for Benchmark Results.

Generates comprehensive markdown documentation from benchmark results.

Example
-------
>>> from temporalcv.compare.docs import generate_benchmark_docs
>>> from temporalcv.compare.results import load_benchmark_results
>>>
>>> report, metadata = load_benchmark_results("results.json")
>>> markdown = generate_benchmark_docs(report, metadata)
>>> Path("docs/benchmarks.md").write_text(markdown)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from temporalcv.compare.base import ComparisonReport, ComparisonResult


# =============================================================================
# Main Generator
# =============================================================================


def generate_benchmark_docs(
    report: ComparisonReport,
    metadata: Optional[Dict[str, Any]] = None,
    include_methodology: bool = True,
) -> str:
    """
    Generate comprehensive benchmark documentation.

    Parameters
    ----------
    report : ComparisonReport
        Benchmark results
    metadata : dict, optional
        Run metadata (timestamp, versions, etc.)
    include_methodology : bool, default=True
        Whether to include methodology section

    Returns
    -------
    str
        Markdown-formatted documentation
    """
    sections: List[str] = []

    # Header
    sections.append("# temporalcv Benchmark Results\n")

    # Metadata
    if metadata:
        sections.append(_generate_metadata_section(metadata))

    # Executive summary
    sections.append(_generate_summary_section(report))

    # Per-dataset results
    sections.append(_generate_detailed_results(report))

    # Statistical significance
    sections.append(_generate_significance_section(report))

    # Methodology
    if include_methodology:
        sections.append(_generate_methodology_section())

    return "\n\n".join(sections)


# =============================================================================
# Section Generators
# =============================================================================


def _generate_metadata_section(metadata: Dict[str, Any]) -> str:
    """Generate metadata section."""
    lines = [
        "## Run Information\n",
        f"- **Generated**: {metadata.get('timestamp', 'N/A')}",
        f"- **Version**: temporalcv {metadata.get('temporalcv_version', 'N/A')}",
        f"- **Python**: {metadata.get('python_version', 'N/A')}",
        f"- **Platform**: {metadata.get('platform', 'N/A')}",
    ]

    if "total_runtime_seconds" in metadata:
        runtime_min = metadata["total_runtime_seconds"] / 60
        lines.append(f"- **Runtime**: {runtime_min:.1f} minutes")

    if "sample_size" in metadata:
        lines.append(f"- **Sample size**: {metadata['sample_size']} series/dataset")

    return "\n".join(lines)


def _generate_summary_section(report: ComparisonReport) -> str:
    """Generate executive summary with best models table."""
    lines = [
        "## Summary\n",
        f"Evaluated **{len(report.results)}** datasets with "
        f"**{len(report.results[0].models) if report.results else 0}** models.\n",
    ]

    # Best model per dataset table
    lines.extend([
        "### Best Model by Dataset\n",
        "| Dataset | Best Model | MAE | RMSE | Runtime |",
        "|---------|------------|-----|------|---------|",
    ])

    for result in report.results:
        best_model = result.best_model
        best_result = next(
            (m for m in result.models if m.model_name == best_model),
            None
        )

        if best_result:
            mae = best_result.metrics.get("mae", float("nan"))
            rmse = best_result.metrics.get("rmse", float("nan"))
            runtime = best_result.runtime_seconds

            lines.append(
                f"| {result.dataset_name} | **{best_model}** | "
                f"{mae:.4f} | {rmse:.4f} | {runtime:.1f}s |"
            )

    # Model wins summary
    if report.summary.get("wins_by_model"):
        lines.extend([
            "\n### Model Wins\n",
            "| Model | Wins | Win Rate |",
            "|-------|------|----------|",
        ])

        total = len(report.results)
        for model, wins in sorted(
            report.summary["wins_by_model"].items(),
            key=lambda x: -x[1]
        ):
            rate = wins / total * 100 if total > 0 else 0
            lines.append(f"| {model} | {wins} | {rate:.0f}% |")

    return "\n".join(lines)


def _generate_detailed_results(report: ComparisonReport) -> str:
    """Generate detailed per-dataset results."""
    lines = ["## Detailed Results\n"]

    for result in report.results:
        lines.append(f"### {result.dataset_name}\n")
        lines.append(f"Best model: **{result.best_model}**\n")

        # Ranking table
        lines.extend([
            "| Rank | Model | MAE | RMSE | MAPE | Direction Acc | Runtime |",
            "|------|-------|-----|------|------|---------------|---------|",
        ])

        ranking = result.get_ranking()
        for rank, (model_name, mae) in enumerate(ranking, 1):
            model = next(
                (m for m in result.models if m.model_name == model_name),
                None
            )
            if model:
                rmse = model.metrics.get("rmse", float("nan"))
                mape = model.metrics.get("mape", float("nan"))
                dir_acc = model.metrics.get("direction_accuracy", float("nan"))
                runtime = model.runtime_seconds

                lines.append(
                    f"| {rank} | {model_name} | {mae:.4f} | {rmse:.4f} | "
                    f"{mape:.1f}% | {dir_acc:.2f} | {runtime:.1f}s |"
                )

        lines.append("")  # Blank line between datasets

    return "\n".join(lines)


def _generate_significance_section(report: ComparisonReport) -> str:
    """Generate statistical significance notes."""
    lines = [
        "## Statistical Significance\n",
        "Diebold-Mariano test comparing best model to alternatives "
        "(p < 0.05 indicates significant difference).\n",
    ]

    for result in report.results:
        if not result.statistical_tests:
            continue

        lines.append(f"### {result.dataset_name}\n")
        lines.append(f"Best model: {result.best_model}\n")

        lines.extend([
            "| Comparison | DM Statistic | p-value | Significant |",
            "|------------|--------------|---------|-------------|",
        ])

        for model_name, test_result in result.statistical_tests.items():
            if isinstance(test_result, dict) and "statistic" in test_result:
                stat = test_result["statistic"]
                pval = test_result["p_value"]
                sig = "Yes" if test_result.get("significant", False) else "No"
                lines.append(
                    f"| vs {model_name} | {stat:.3f} | {pval:.4f} | {sig} |"
                )
            elif isinstance(test_result, dict) and "error" in test_result:
                lines.append(f"| vs {model_name} | - | - | Error: {test_result['error']} |")

        lines.append("")

    return "\n".join(lines)


def _generate_methodology_section() -> str:
    """Generate methodology documentation."""
    return """## Methodology

### Datasets

**M4 Competition** (Makridakis et al., 2020):
- 100,000 time series across 6 frequencies
- Used subset for benchmarking (configurable sample size)
- Official train/test splits from competition

**M5 Competition** (Makridakis et al., 2022):
- 30,490 hierarchical time series from Walmart
- 28-day forecast horizon
- Requires manual download due to Kaggle TOS

### Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error - primary ranking metric |
| **RMSE** | Root Mean Squared Error - penalizes large errors |
| **MAPE** | Mean Absolute Percentage Error - scale-independent |
| **Direction Accuracy** | Proportion of correct direction predictions |

### Statistical Tests

**Diebold-Mariano Test** (Diebold & Mariano, 1995):
- Tests whether forecast accuracy difference is significant
- Uses HAC variance estimator for autocorrelation
- p < 0.05 indicates statistically significant difference

### Models

| Model | Package | Description |
|-------|---------|-------------|
| Naive | temporalcv | Last value persistence |
| SeasonalNaive | temporalcv | Seasonal lag persistence |
| AutoARIMA | statsforecast | Automatic ARIMA selection |
| AutoETS | statsforecast | Automatic exponential smoothing |
| AutoTheta | statsforecast | Theta method with automatic tuning |
| CrostonClassic | statsforecast | Intermittent demand model |
| ADIDA | statsforecast | Aggregate-Disaggregate Intermittent Demand |
| IMAPA | statsforecast | Intermittent Multiple Aggregation Prediction |

### Reproducibility

Results can be reproduced with:
```bash
# Quick validation
python scripts/run_benchmark.py --quick

# Full benchmark
python scripts/run_benchmark.py --full
```

See `benchmarks/results/` for raw JSON data.

### References

- Makridakis, S., et al. (2020). "The M4 Competition." IJF.
- Makridakis, S., et al. (2022). "M5 accuracy competition." IJF.
- Diebold, F. & Mariano, R. (1995). "Comparing Predictive Accuracy." JBES.
"""


# =============================================================================
# Utility Functions
# =============================================================================


def generate_summary_table(report: ComparisonReport) -> str:
    """
    Generate standalone summary table.

    Parameters
    ----------
    report : ComparisonReport
        Benchmark results

    Returns
    -------
    str
        Markdown table
    """
    lines = [
        "| Dataset | Best Model | MAE | vs Naive |",
        "|---------|------------|-----|----------|",
    ]

    for result in report.results:
        best = result.best_model
        best_result = next(
            (m for m in result.models if m.model_name == best),
            None
        )
        naive_result = next(
            (m for m in result.models if m.model_name == "Naive"),
            None
        )

        if best_result:
            mae = best_result.metrics.get("mae", float("nan"))

            vs_naive = ""
            if naive_result and best != "Naive":
                naive_mae = naive_result.metrics.get("mae", float("nan"))
                if naive_mae > 0:
                    improvement = (naive_mae - mae) / naive_mae * 100
                    vs_naive = f"{improvement:+.1f}%"

            lines.append(
                f"| {result.dataset_name} | **{best}** | {mae:.4f} | {vs_naive} |"
            )

    return "\n".join(lines)


def generate_ranking_table(result: ComparisonResult) -> str:
    """
    Generate ranking table for single dataset.

    Parameters
    ----------
    result : ComparisonResult
        Single dataset result

    Returns
    -------
    str
        Markdown table
    """
    lines = [
        f"| Rank | Model | {result.primary_metric.upper()} |",
        "|------|-------|------|",
    ]

    for rank, (model_name, value) in enumerate(result.get_ranking(), 1):
        lines.append(f"| {rank} | {model_name} | {value:.4f} |")

    return "\n".join(lines)


__all__ = [
    "generate_benchmark_docs",
    "generate_summary_table",
    "generate_ranking_table",
]
