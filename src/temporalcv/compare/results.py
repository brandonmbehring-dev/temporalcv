"""
Result Serialization for Benchmark Comparison.

Provides JSON serialization and metadata collection for benchmark results.

Example
-------
>>> from temporalcv.compare.results import save_benchmark_results, load_benchmark_results
>>> from temporalcv.compare import run_benchmark_suite
>>>
>>> report = run_benchmark_suite(datasets, adapters)
>>> save_benchmark_results(report, Path("results.json"))
>>> loaded = load_benchmark_results(Path("results.json"))
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from temporalcv.compare.base import (
    ComparisonReport,
    ComparisonResult,
    ModelResult,
)


def create_run_metadata(
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create metadata for a benchmark run.

    Captures environment information for reproducibility.

    Parameters
    ----------
    models : list[str], optional
        List of model names being compared
    datasets : list[str], optional
        List of dataset names being evaluated

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with:
        - run_id: Unique identifier
        - timestamp: ISO 8601 timestamp
        - temporalcv_version: Package version
        - python_version: Python version
        - platform: OS platform
        - cpu_count: Number of CPUs
        - models: List of model names (if provided)
        - datasets: List of dataset names (if provided)
    """
    import temporalcv

    metadata: Dict[str, Any] = {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "temporalcv_version": getattr(temporalcv, "__version__", "unknown"),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "cpu_count": os.cpu_count() or 1,
    }

    if models:
        metadata["models"] = models
    if datasets:
        metadata["datasets"] = datasets

    return metadata


def _serialize_report(report: ComparisonReport) -> Dict[str, Any]:
    """
    Serialize ComparisonReport to JSON-compatible dictionary.

    Parameters
    ----------
    report : ComparisonReport
        Report to serialize

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary
    """
    serialized_results = []
    for result in report.results:
        serialized_models = []
        for model in result.models:
            model_dict = model.to_dict()
            # Note: predictions are excluded from serialization (too large)
            # They can be reconstructed by re-running
            serialized_models.append(model_dict)

        result_dict = {
            "dataset_name": result.dataset_name,
            "models": serialized_models,
            "primary_metric": result.primary_metric,
            "best_model": result.best_model,
            "statistical_tests": result.statistical_tests,
        }
        serialized_results.append(result_dict)

    return {
        "results": serialized_results,
        "summary": report.summary,
    }


def _deserialize_report(data: Dict[str, Any]) -> ComparisonReport:
    """
    Deserialize dictionary to ComparisonReport.

    Parameters
    ----------
    data : dict[str, Any]
        Serialized report data

    Returns
    -------
    ComparisonReport
        Reconstructed report (without predictions array)
    """
    results = []
    for result_data in data["results"]:
        models = []
        for model_data in result_data["models"]:
            # Create ModelResult without predictions (set to empty array)
            model = ModelResult(
                model_name=model_data["model_name"],
                package=model_data["package"],
                metrics=model_data["metrics"],
                predictions=np.array([]),  # Placeholder
                runtime_seconds=model_data["runtime_seconds"],
                model_params=model_data.get("model_params"),
            )
            models.append(model)

        # Create ComparisonResult
        # Note: We need to set best_model after creation since __post_init__ computes it
        result = ComparisonResult(
            dataset_name=result_data["dataset_name"],
            models=models,
            primary_metric=result_data["primary_metric"],
            statistical_tests=result_data.get("statistical_tests"),
        )
        # Override best_model to match saved value (in case of tie-breaking differences)
        object.__setattr__(result, "best_model", result_data["best_model"])
        results.append(result)

    return ComparisonReport(results=results, summary=data.get("summary", {}))


def save_benchmark_results(
    report: ComparisonReport,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save benchmark results to JSON file.

    Parameters
    ----------
    report : ComparisonReport
        Results to save
    path : Path
        Output file path (should end in .json)
    metadata : dict, optional
        Additional metadata (from create_run_metadata)

    Raises
    ------
    ValueError
        If path doesn't end with .json
    """
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError(f"Output path must end with .json, got: {path}")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": metadata or create_run_metadata(),
        "report": _serialize_report(report),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=_json_default)


def load_benchmark_results(path: Path) -> tuple[ComparisonReport, Dict[str, Any]]:
    """
    Load benchmark results from JSON file.

    Parameters
    ----------
    path : Path
        Input file path

    Returns
    -------
    tuple[ComparisonReport, dict]
        Loaded report and metadata

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    ValueError
        If file format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "report" not in data:
        raise ValueError(f"Invalid results file format: missing 'report' key in {path}")

    report = _deserialize_report(data["report"])
    metadata = data.get("metadata", {})

    return report, metadata


def _json_default(obj: Any) -> Any:
    """
    JSON encoder for numpy types and other special objects.

    Parameters
    ----------
    obj : Any
        Object to encode

    Returns
    -------
    Any
        JSON-serializable representation

    Raises
    ------
    TypeError
        If object cannot be serialized
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(
    result: ComparisonResult,
    checkpoint_dir: Path,
    dataset_name: str,
) -> Path:
    """
    Save a single dataset result as a checkpoint.

    Parameters
    ----------
    result : ComparisonResult
        Result to checkpoint
    checkpoint_dir : Path
        Directory for checkpoints
    dataset_name : str
        Name of the dataset (used for filename)

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize dataset name for filename
    safe_name = dataset_name.replace("/", "_").replace(" ", "_").lower()
    checkpoint_path = checkpoint_dir / f"{safe_name}.json"

    # Create mini-report for this single result
    mini_report = ComparisonReport(results=[result])
    save_benchmark_results(mini_report, checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> Optional[ComparisonResult]:
    """
    Load a single dataset result from checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file

    Returns
    -------
    ComparisonResult or None
        Loaded result, or None if checkpoint doesn't exist
    """
    if not checkpoint_path.exists():
        return None

    report, _ = load_benchmark_results(checkpoint_path)
    if report.results:
        return report.results[0]
    return None


def list_checkpoints(checkpoint_dir: Path) -> List[str]:
    """
    List available checkpoints in a directory.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoints

    Returns
    -------
    list[str]
        List of dataset names with available checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = []
    for path in checkpoint_dir.glob("*.json"):
        # Extract dataset name from filename
        name = path.stem.replace("_", " ").title()
        checkpoints.append(name)

    return checkpoints


__all__ = [
    "create_run_metadata",
    "save_benchmark_results",
    "load_benchmark_results",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
]
