# Benchmark Strategy

**Purpose**: Datasets and metrics for validating temporalcv.

---

## Benchmark Matrix by Feature

| Focus | Datasets | Purpose |
|-------|----------|---------|
| Univariate baseline | M3, M4 (subset), NN5 | Classical comparison |
| Exogenous + hierarchy | M5, Rossmann, Favorita | Covariate alignment |
| Probabilistic | M5, GluonTS Electricity | Coverage, CRPS |
| Persistence-heavy | FRED rates/yields | Move metrics |
| High-frequency | PEMS traffic | Spatial, gap-aware |

---

## Event-Aware Metrics (Novel)

| Metric | Purpose | Existing? |
|--------|---------|-----------|
| **MC-SS** | Skill on moves only | ❌ None |
| **Move-only MAE** | Error when target moved | ❌ None |
| **Brier score** | Direction probability | Partial |
| **PR-AUC** | Imbalanced direction | Partial |
| **Move threshold** | Fair persistence baseline | ❌ None |

---

## Dataset Licensing

### Kaggle Datasets (M5, Rossmann, Favorita)

⚠️ **TOS**: Cannot redistribute; users must download themselves

**Implementation**:
```python
def load_m5(path: Optional[Path] = None) -> Dataset:
    """Load M5 dataset.

    NOTE: Due to Kaggle TOS, data cannot be bundled.
    Download from: https://www.kaggle.com/competitions/m5-forecasting-accuracy

    Raises
    ------
    DatasetNotFoundError
        If data not found with download instructions.
    """
```

### FRED/Macro Data

✅ **Public domain** (US government data)

**Implementation**: Use `fredapi` with documented series IDs

### GluonTS Bundle

✅ **Open access** with documented splits

**Implementation**: Direct loaders with split verification

### Monash Repository

✅ **Open access** with train/test splits

---

## Validation Protocol

```python
from temporalcv.benchmarks import load_benchmark, run_validation_suite

# Load persistence-heavy benchmark
data = load_benchmark("fred_rates")

# Run full validation
results = run_validation_suite(
    model=my_model,
    data=data,
    gates=[
        gate_shuffled_target(),
        gate_horizon_gap(h=2),
        gate_exogenous_alignment(),
    ],
    metrics=["mae", "mc_ss", "move_only_mae", "direction_brier"],
    compare_to=["persistence", "arima", "statsforecast"]
)
```

---

## Comparison Targets

**Python**: statsforecast, neuralforecast, gluonts, sktime, darts, prophet

**R**: forecast, fable/feasts, modeltime

---

## Data Sources

| Source | Series | Key Feature |
|--------|--------|-------------|
| M5 (Walmart) | Hierarchical retail | Rich exogenous |
| FRED panels | Rates, spreads | High persistence |
| Rossmann | Daily retail | Intermittent demand |
| GluonTS | Electricity, Traffic | Probabilistic standard |
| PEMS | Sensor volumes | Spatial graphs |
