# temporalcv Tutorial Notebooks

Interactive Jupyter notebooks teaching time-series validation to ML practitioners.

## Target Audience

ML practitioners who understand standard ML (KFold, train_test_split, MAE/RMSE) but are new to time-series validation challenges.

---

## Learning Paths

### Path A: Complete Newcomer (Start Here!) — ~6 hours
**For sklearn practitioners who are new to time-series ML:**

| Step | Notebook | Time | What You'll Learn |
|------|----------|------|-------------------|
| 1 | `00_time_series_fundamentals.ipynb` | 30 min | Why time-series differs from regular ML, ACF intuition |
| 2 | `01_why_temporal_cv.ipynb` | 45 min | Why KFold fails, WalkForwardCV, three types of leakage |
| 3 | `02_gap_enforcement.ipynb` | 30 min | Multi-step forecasting, gap >= horizon rule |
| 4 | `05_shuffled_target_gate.ipynb` | 45 min | Definitive leakage detection |
| 5 | `08_validation_workflow.ipynb` | 45 min | Complete HALT/WARN/PASS pipeline |
| 6 | `10_high_persistence_metrics.ipynb` | 45 min | MASE, MC-SS, move-conditional metrics |

**Also read**: [Feature Engineering Safety Guide](../docs/tutorials/feature_engineering_safety.md) (15 min)

### Path B: Know Time-Series, Learn temporalcv — ~3 hours
**For practitioners who already understand ACF, stationarity, and temporal dependence:**

| Step | Notebook | Time | What You'll Learn |
|------|----------|------|-------------------|
| 1 | `01_why_temporal_cv.ipynb` | 45 min | WalkForwardCV, three leakage types |
| 2 | `05_shuffled_target_gate.ipynb` | 45 min | Definitive leakage detection |
| 3 | `08_validation_workflow.ipynb` | 45 min | Complete HALT/WARN/PASS pipeline |
| 4 | `10_high_persistence_metrics.ipynb` | 45 min | Move-conditional metrics |

### Complete Curriculum — ~14 hours
**For comprehensive understanding:**
- **Week 1**: Tier 0 + Tier 1 (00, 01-04) — Why time-series is different
- **Week 2**: Tier 2 (05-08) — Detecting and avoiding leakage
- **Week 3**: Tier 3 (09-12) — Advanced metrics and statistical tests

---

## Curriculum Structure

### Tier 0: Fundamentals — Bridge from sklearn to Time-Series

**For sklearn practitioners who don't understand why time-series is different.**

| Notebook | Key Concept | Prerequisites |
|----------|-------------|---------------|
| `00_time_series_fundamentals.ipynb` | ACF intuition, why shuffling destroys dependencies, three types of leakage | Basic ML knowledge |

**Supporting Materials**:
- [Feature Engineering Safety Guide](../docs/tutorials/feature_engineering_safety.md) — Safe vs dangerous features
- [Metric Selection Guide](../docs/tutorials/metric_selection.md) — Which metric for which problem
- [Diagnostic Flowchart](../docs/tutorials/diagnostic_flowchart.md) — Troubleshooting validation failures

---

### Tier 1: Foundation — Why Time-Series is Different

| Notebook | Key Concept | Prerequisites |
|----------|-------------|---------------|
| `01_why_temporal_cv.ipynb` | KFold leakage, WalkForwardCV, persistence baseline | Tier 0 or time-series experience |
| `02_gap_enforcement.ipynb` | h-step forecasting, gap >= horizon | 01 |
| `03_persistence_baseline.ipynb` | MASE, why persistence is hard to beat | 01-02 |
| `04_autocorrelation_matters.ipynb` | HAC variance, MA(h-1) error structure | 01-03 |

### Tier 2: Prevention — Detecting and Avoiding Leakage

| Notebook | Key Concept | Prerequisites |
|----------|-------------|---------------|
| `05_shuffled_target_gate.ipynb` | Definitive leakage detection | 01, 04 |
| `06_feature_engineering_pitfalls.ipynb` | Safe lag features, rolling stats | 01, 05 |
| `07_threshold_leakage.ipynb` | Regime/percentile computation | 01, 05, 06 |
| `08_validation_workflow.ipynb` | HALT/WARN/PASS pipeline, run_gates | 01-07 |

### Tier 3: Evaluation — High-Persistence and Advanced Metrics

| Notebook | Key Concept | Prerequisites |
|----------|-------------|---------------|
| `09_statistical_tests_dm_pt.ipynb` | Diebold-Mariano, Pesaran-Timmermann | 01-04 |
| `10_high_persistence_metrics.ipynb` | MC-SS, move-conditional, Theil's U | 01-04 |
| `11_conformal_prediction.ipynb` | Split/Adaptive conformal, intervals | 01-04 |
| `12_regime_stratified_evaluation.ipynb` | Volatility regimes, stratified gates | 01-11 |

---

## What You'll Learn

After completing these notebooks, you'll be able to:

1. **Validate correctly**: Use walk-forward CV instead of KFold for time series
2. **Detect leakage**: Run shuffled target tests to catch subtle bugs
3. **Enforce gaps**: Configure proper gaps for multi-step forecasting
4. **Evaluate properly**: Use MASE and move-conditional metrics on persistent data
5. **Quantify uncertainty**: Build prediction intervals with coverage guarantees

---

## Pedagogical Approach

Each notebook follows the **PROBLEM → FAILURE → SOLUTION** pattern:

1. **The Problem**: Demonstrate a common mistake (e.g., using KFold)
2. **The Failure**: Show how it produces misleading results
3. **The Solution**: Introduce the temporalcv approach

### Pitfall Sections
Every notebook includes explicit WRONG vs RIGHT code examples:

```python
# WRONG: Centered rolling mean includes future!
smoothed[t] = np.mean(series[t-3:t+4])  # BUG!

# RIGHT: Backward-looking only
smoothed[t] = np.mean(series[t-6:t+1])  # Safe
```

### Knowledge Tiers
Concepts are tagged by confidence level:
- **[T1]**: Academically validated (DM test, conformal theory)
- **[T2]**: Empirical findings (70th percentile threshold)
- **[T3]**: Assumptions (20% suspicious threshold)

---

## Requirements

```bash
pip install temporalcv
```

All notebooks use synthetic data by default (zero external dependencies).

---

## Related Resources

- **[examples/](../examples/)**: Runnable Python scripts (same content, different format)
- **[docs/tutorials/](../docs/tutorials/)**: Markdown tutorials (quick reference)
- **[SPECIFICATION.md](../SPECIFICATION.md)**: Authoritative thresholds and parameters
