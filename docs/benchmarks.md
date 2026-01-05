# Benchmark Results

**Last updated**: 2025-12-31
**temporalcv version**: 1.0.0-rc1
**Run ID**: run_c700bfd9 (full benchmark)

---

## Overview

This document presents benchmark results comparing forecasting models on the M4 Competition dataset using temporalcv's model comparison framework. The benchmarks evaluate baseline models against statsforecast's automatic model selection algorithms.

## Summary

| Metric | Value |
|--------|-------|
| Datasets | 6 (M4 Competition, all frequencies) |
| Total series | 4,773 (1000 yearly + 1000 quarterly + 1000 monthly + 359 weekly + 1000 daily + 414 hourly) |
| Models compared | 9 |
| Total runtime | 14.3 minutes |

### Model Wins by Frequency

| Model | Wins | Winning Frequencies |
|-------|------|---------------------|
| **AutoETS** | 3 | quarterly, monthly, weekly |
| AutoTheta | 1 | yearly |
| Naive | 1 | daily |
| AutoARIMA | 1 | hourly |

### Mean MAE Across All Frequencies

| Rank | Model | Mean MAE | vs Naive |
|------|-------|----------|----------|
| 1 | **AutoARIMA** | 475.9 | -12.9% |
| 2 | AutoETS | 518.7 | -5.0% |
| 3 | AutoTheta | 521.8 | -4.5% |
| 4 | Naive | 546.2 | — |
| 5 | ADIDA | 609.5 | +11.6% |
| 6 | IMAPA | 609.5 | +11.6% |
| 7 | CrostonClassic | 698.8 | +27.9% |
| 8 | HistoricAverage | 784.4 | +43.6% |
| 9 | SeasonalNaive_12 | 800.4 | +46.5% |

> **Note**: SeasonalNaive_12 uses season_length=12 for all frequencies, which is only appropriate for monthly data. This explains its poor performance on yearly, quarterly, weekly, daily, and hourly data.

> **Key insight**: AutoARIMA has the best mean MAE overall (-12.9% vs Naive), but AutoETS wins the most individual frequencies (3/6). This suggests AutoETS is more robust across frequency types, while AutoARIMA excels at high-frequency data (hourly).

---

## Per-Frequency Results

### M4 Yearly (1000 series, horizon=6)

| Rank | Model | MAE |
|------|-------|-----|
| 1 | **AutoTheta** | 625.6 |
| 2 | AutoETS | 628.8 |
| 3 | AutoARIMA | 703.3 |
| 4 | Naive | 704.5 |
| 5 | ADIDA | 896.3 |
| 6 | IMAPA | 896.3 |
| 7 | HistoricAverage | 1256.1 |
| 8 | CrostonClassic | 1277.5 |
| 9 | SeasonalNaive_12 | 1548.4 |

### M4 Quarterly (1000 series, horizon=8)

| Rank | Model | MAE |
|------|-------|-----|
| 1 | **AutoETS** | 426.2 |
| 2 | AutoTheta | 435.0 |
| 3 | AutoARIMA | 444.3 |
| 4 | Naive | 455.6 |
| 5 | ADIDA | 482.6 |
| 6 | IMAPA | 482.6 |
| 7 | CrostonClassic | 629.8 |
| 8 | SeasonalNaive_12 | 692.4 |
| 9 | HistoricAverage | 700.2 |

### M4 Monthly (1000 series, horizon=18)

| Rank | Model | MAE |
|------|-------|-----|
| 1 | **AutoETS** | 479.7 |
| 2 | AutoARIMA | 486.8 |
| 3 | AutoTheta | 492.7 |
| 4 | ADIDA | 516.1 |
| 5 | IMAPA | 516.1 |
| 6 | Naive | 537.5 |
| 7 | CrostonClassic | 573.7 |
| 8 | SeasonalNaive_12 | 587.9 |
| 9 | HistoricAverage | 868.6 |

### M4 Weekly (359 series, horizon=13)

| Rank | Model | MAE |
|------|-------|-----|
| 1 | **AutoETS** | 247.2 |
| 2 | Naive | 249.1 |
| 3 | AutoTheta | 251.8 |
| 4 | AutoARIMA | 266.6 |
| 5 | ADIDA | 276.3 |
| 6 | IMAPA | 276.3 |
| 7 | CrostonClassic | 318.1 |
| 8 | SeasonalNaive_12 | 366.1 |
| 9 | HistoricAverage | 408.2 |

### M4 Daily (1000 series, horizon=14)

| Rank | Model | MAE |
|------|-------|-----|
| 1 | **Naive** | 109.6 |
| 2 | AutoETS | 110.8 |
| 3 | AutoTheta | 111.5 |
| 4 | AutoARIMA | 114.2 |
| 5 | ADIDA | 120.4 |
| 6 | IMAPA | 120.4 |
| 7 | CrostonClassic | 148.0 |
| 8 | SeasonalNaive_12 | 156.3 |
| 9 | HistoricAverage | 287.2 |

> **Note**: On daily data, the simple Naive baseline wins. This is notable — for short-horizon daily forecasting, complex models may overfit.

### M4 Hourly (414 series, horizon=48)

| Rank | Model | MAE |
|------|-------|-----|
| 1 | **AutoARIMA** | 840.2 |
| 2 | HistoricAverage | 1186.2 |
| 3 | AutoTheta | 1214.3 |
| 4 | AutoETS | 1219.8 |
| 5 | Naive | 1220.7 |
| 6 | CrostonClassic | 1245.5 |
| 7 | ADIDA | 1365.2 |
| 8 | IMAPA | 1365.2 |
| 9 | SeasonalNaive_12 | 1451.5 |

> **Note**: AutoARIMA significantly outperforms all other models on hourly data (-31% vs Naive), suggesting ARIMA captures high-frequency patterns better than exponential smoothing methods.

---

## Key Findings

### 1. AutoETS Most Robust Across Frequencies

AutoETS wins 3/6 frequencies (quarterly, monthly, weekly), making it the most robust choice for general-purpose forecasting. However, AutoARIMA has the best mean MAE overall due to its exceptional performance on hourly data.

### 2. Frequency-Specific Winners

| Frequency | Best Model | Key Insight |
|-----------|------------|-------------|
| Yearly | AutoTheta | Captures long-term trends with damping |
| Quarterly | AutoETS | Smooth exponential patterns |
| Monthly | AutoETS | Handles seasonal variation well |
| Weekly | AutoETS | Short-term smoothing effective |
| Daily | **Naive** | Simple baseline wins — complex models overfit |
| Hourly | AutoARIMA | ARIMA excels at high-frequency patterns |

### 3. Naive Baseline Surprisingly Strong on Daily Data

The Naive baseline won on M4 daily data. This is notable — for short-horizon daily forecasting, complex models may introduce unnecessary variance.

### 4. AutoARIMA Dominates High-Frequency Data

AutoARIMA achieved -31% improvement vs Naive on hourly data, by far the largest improvement. ARIMA's autoregressive structure captures high-frequency patterns that exponential smoothing misses.

### 5. Intermittent Demand Models Underperform

CrostonClassic, ADIDA, and IMAPA are designed for intermittent demand (many zeros). They underperform on M4 data which contains continuous demand patterns.

### 6. Seasonality Mismatch Hurts Performance

SeasonalNaive_12 performs poorly because it uses a fixed 12-period seasonal lag regardless of the actual data frequency. Proper seasonal period selection is critical.

---

## Methodology

### Data

- **Source**: M4 Competition (Makridakis et al., 2018)
- **Frequencies**: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly
- **Series**: 1000/1000/1000/359/1000/414 per frequency (4,773 total)
- **Split**: Official M4 train/test splits
- **Sampling**: Random seed 42 (M4 weekly/hourly have fewer than 1000 series total)

### Models

| Model | Package | Type |
|-------|---------|------|
| Naive | temporalcv | Baseline |
| SeasonalNaive | temporalcv | Baseline |
| AutoARIMA | statsforecast | Automatic |
| AutoETS | statsforecast | Automatic |
| AutoTheta | statsforecast | Automatic |
| CrostonClassic | statsforecast | Intermittent |
| ADIDA | statsforecast | Intermittent |
| IMAPA | statsforecast | Intermittent |
| HistoricAverage | statsforecast | Simple |

### Metrics

- **MAE**: Mean Absolute Error (primary metric)
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Direction Accuracy**: Fraction of correct direction predictions

### Statistical Testing

Diebold-Mariano (DM) test with HAC variance estimation compares each model against the best model per frequency. Significance level: p < 0.05 (marked with *).

---

## Reproduction

```bash
# Install dependencies
pip install temporalcv[compare] datasetsforecast statsforecast

# Run quick benchmark (100 series/freq, ~4 minutes)
python scripts/run_benchmark.py --quick --models all

# Run full benchmark (1000 series/freq, ~15 minutes)
python scripts/run_benchmark.py --full --models all

# Results saved to benchmarks/results/run_<id>/
```

**Hardware**: 128-core AMD EPYC (8 jobs parallel via joblib)
**Runtime**: 14.3 minutes for full benchmark

See [docs/benchmarks/reproduce.md](benchmarks/reproduce.md) for detailed instructions.

---

## References

1. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion and way forward. International Journal of Forecasting, 34(4), 802-808.

2. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.

3. Nixtla. (2023). statsforecast: Lightning fast forecasting with statistical and econometric models. https://github.com/Nixtla/statsforecast
