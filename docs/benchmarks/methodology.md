# Benchmark Methodology

This document describes how temporalcv benchmark results are generated.

## Datasets

### M4 Competition

The M4 Competition (Makridakis et al., 2020) provides 100,000 time series across 6 frequencies:

| Frequency | Series | Horizon | Min Length |
|-----------|--------|---------|------------|
| Yearly | 23,000 | 6 | 13 |
| Quarterly | 24,000 | 8 | 16 |
| Monthly | 48,000 | 18 | 42 |
| Weekly | 359 | 13 | 80 |
| Daily | 4,227 | 14 | 93 |
| Hourly | 414 | 48 | 700 |

**Benchmark sampling**: To reduce compute time, we sample a subset of series per frequency:
- Quick mode: 100 series/frequency
- Full mode: 1000 series/frequency

Official train/test splits from the competition are preserved.

### M5 Competition

The M5 Competition (Makridakis et al., 2022) provides 30,490 hierarchical time series from Walmart:

- **Horizon**: 28 days
- **Features**: Rich exogenous variables (price, promotions, calendar)
- **Characteristics**: Intermittent demand, hierarchical structure

**Data access**: Due to Kaggle Terms of Service, M5 data cannot be bundled. Users must download manually:
```bash
# From Kaggle
kaggle competitions download -c m5-forecasting-accuracy
```

## Models

### Baseline Models

| Model | Description | Parameters |
|-------|-------------|------------|
| **Naive** | Repeats last observed value | None |
| **SeasonalNaive** | Repeats value from same seasonal period | `season_length` |

### Statsforecast Models

Requires: `pip install statsforecast`

| Model | Description | Use Case |
|-------|-------------|----------|
| **AutoARIMA** | Automatic ARIMA selection | General purpose |
| **AutoETS** | Automatic exponential smoothing | Trended/seasonal data |
| **AutoTheta** | Theta method with automatic tuning | Competition-winning |
| **CrostonClassic** | Intermittent demand model | Sparse demand |
| **ADIDA** | Aggregate-Disaggregate Intermittent | Intermittent demand |
| **IMAPA** | Multiple Aggregation Prediction | Intermittent demand |
| **HistoricAverage** | Mean of all historical values | Stable series |

## Metrics

### Error Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|y - ŷ|)` | Average absolute error |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalizes large errors |
| **MAPE** | `mean(|y - ŷ| / |y|) × 100` | Percentage error |

### Direction Metrics

| Metric | Description |
|--------|-------------|
| **Direction Accuracy** | Proportion of correct direction predictions |

## Statistical Tests

### Diebold-Mariano Test

Tests whether forecast accuracy difference between two models is statistically significant.

**Null hypothesis**: No difference in predictive accuracy

**Implementation**:
- Uses HAC (Heteroskedasticity and Autocorrelation Consistent) variance estimator
- Accounts for forecast horizon via Newey-West bandwidth
- p < 0.05 indicates significant difference

**Reference**: Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." JBES 13(3): 253-263.

## Reproducibility

### Running Benchmarks

```bash
# Quick validation (~1 hour)
python scripts/run_benchmark.py --quick

# Full benchmark (~8-10 hours with 8 cores)
python scripts/run_benchmark.py --full

# Single frequency
python scripts/run_benchmark.py --dataset m4_monthly --sample 1000

# Resume interrupted run
python scripts/run_benchmark.py --resume benchmarks/results/run_abc123
```

### Output Files

Results are saved to `benchmarks/results/run_<uuid>/`:

| File | Content |
|------|---------|
| `results.json` | Structured benchmark results |
| `results.md` | Markdown-formatted tables |
| `benchmark.log` | Execution log |
| `checkpoints/` | Per-dataset checkpoints |

### Environment

Benchmark metadata includes:
- temporalcv version
- Python version
- Platform
- CPU count
- Runtime

## Limitations

1. **Sampling bias**: Subsampling may not represent full dataset characteristics
2. **Model configurations**: Default parameters used; tuning may improve results
3. **Single train/test split**: No cross-validation uncertainty estimates
4. **Compute constraints**: Full M4 (100k series) requires significant resources

## References

- Makridakis, S., et al. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." International Journal of Forecasting 36(1): 54-74.
- Makridakis, S., et al. (2022). "M5 accuracy competition: Results, findings, and conclusions." International Journal of Forecasting 38(4): 1346-1364.
- Hyndman, R.J. & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice." 3rd ed. OTexts.
