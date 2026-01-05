# Reproducing Benchmark Results

Step-by-step guide to reproducing temporalcv benchmark results.

## Prerequisites

### Required

```bash
# Install temporalcv with benchmark dependencies
pip install temporalcv[benchmarks]

# Or from source
pip install -e ".[dev]"
```

### Optional

```bash
# For statsforecast models
pip install statsforecast

# For M5 dataset
pip install kaggle
```

## Quick Start

### Validation Run (~1 hour)

```bash
# 100 series per frequency, baseline + statsforecast models
python scripts/run_benchmark.py --quick
```

### Full Benchmark (~8-10 hours)

```bash
# 1000 series per frequency
python scripts/run_benchmark.py --full
```

## Dataset-Specific Runs

### M4 Competition

```bash
# All M4 frequencies
python scripts/run_benchmark.py --dataset m4_yearly --sample 1000
python scripts/run_benchmark.py --dataset m4_quarterly --sample 1000
python scripts/run_benchmark.py --dataset m4_monthly --sample 1000
python scripts/run_benchmark.py --dataset m4_weekly --sample 1000
python scripts/run_benchmark.py --dataset m4_daily --sample 1000
python scripts/run_benchmark.py --dataset m4_hourly --sample 1000
```

### M5 Competition

M5 requires manual download due to Kaggle TOS:

```bash
# 1. Download from Kaggle
kaggle competitions download -c m5-forecasting-accuracy

# 2. Extract to a directory
unzip m5-forecasting-accuracy.zip -d ~/data/m5/

# 3. Run benchmark
python scripts/run_benchmark.py --dataset m5 --m5-path ~/data/m5/ --sample 1000
```

## Model Selection

### Baseline Only (Fast)

```bash
python scripts/run_benchmark.py --quick --models baseline
```

### Statsforecast Only

```bash
python scripts/run_benchmark.py --quick --models statsforecast
```

### All Models (Default)

```bash
python scripts/run_benchmark.py --quick --models all
```

## Resuming Interrupted Runs

Benchmarks save checkpoints after each dataset. To resume:

```bash
# Find your run directory
ls benchmarks/results/

# Resume from checkpoint
python scripts/run_benchmark.py --resume benchmarks/results/run_abc123
```

## Output Interpretation

### results.json

```json
{
  "metadata": {
    "run_id": "abc12345",
    "timestamp": "2025-01-15T10:30:00Z",
    "temporalcv_version": "1.0.0",
    "total_runtime_seconds": 3600.0
  },
  "report": {
    "results": [
      {
        "dataset_name": "M4_monthly",
        "best_model": "AutoETS",
        "models": [...]
      }
    ],
    "summary": {
      "wins_by_model": {"AutoETS": 3, "AutoARIMA": 2, ...}
    }
  }
}
```

### results.md

Markdown tables ready for documentation:

```markdown
## Summary

| Dataset | Best Model | MAE | vs Naive |
|---------|------------|-----|----------|
| M4_yearly | AutoETS | 0.1234 | -15.2% |
...
```

## Generating Documentation

After running benchmarks:

```python
from temporalcv.compare import (
    load_benchmark_results,
    generate_benchmark_docs,
)
from pathlib import Path

# Load results
report, metadata = load_benchmark_results(
    Path("benchmarks/results/run_abc123/results.json")
)

# Generate comprehensive documentation
docs = generate_benchmark_docs(report, metadata)

# Save to docs
Path("docs/benchmarks.md").write_text(docs)
```

## Troubleshooting

### ImportError: statsforecast not found

```bash
pip install statsforecast
```

### M5 DatasetNotFoundError

Download M5 data from Kaggle and provide path:
```bash
python scripts/run_benchmark.py --m5-path /path/to/m5/
```

### Out of Memory

Reduce sample size:
```bash
python scripts/run_benchmark.py --sample 100
```

Or run single frequency:
```bash
python scripts/run_benchmark.py --dataset m4_monthly --sample 500
```

### Slow Performance

Enable parallel execution (requires joblib):
```python
from temporalcv.compare.adapters.multi_series import MultiSeriesAdapter

# Wrap adapter for parallel execution
adapter = MultiSeriesAdapter(base_adapter, n_jobs=8)
```

## Hardware Requirements

| Mode | RAM | Time (8-core) | Time (1-core) |
|------|-----|---------------|---------------|
| Quick | 4GB | ~1 hour | ~6-8 hours |
| Full | 8GB | ~8-10 hours | ~60+ hours |

## Verifying Results

Compare your results to published benchmarks:

```python
from temporalcv.compare import load_benchmark_results

# Load your results
my_report, _ = load_benchmark_results("benchmarks/results/run_xyz/results.json")

# Load reference results (if available)
ref_report, _ = load_benchmark_results("benchmarks/reference/results.json")

# Compare
for my_result, ref_result in zip(my_report.results, ref_report.results):
    print(f"{my_result.dataset_name}:")
    print(f"  Your best: {my_result.best_model}")
    print(f"  Reference: {ref_result.best_model}")
```
