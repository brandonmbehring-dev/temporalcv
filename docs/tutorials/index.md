# Tutorials

Step-by-step guides for common temporalcv workflows.

---

## New to Time-Series ML?

**If you know sklearn but don't understand why time-series is different, start here:**

### Foundational Reading (~1 hour)

| Resource | Time | What You'll Learn |
|----------|------|-------------------|
| [00_time_series_fundamentals.ipynb](../../notebooks/00_time_series_fundamentals.ipynb) | 30 min | Why autocorrelation matters, ACF intuition, three types of leakage |
| [Feature Engineering Safety Guide](feature_engineering_safety.md) | 15 min | Safe vs dangerous features, decision tree |
| [Metric Selection Guide](metric_selection.md) | 15 min | Which metric for which problem |

### Quick Start Path (~4 hours)

After the foundational reading, complete these notebooks:

| Step | Notebook | Concept |
|------|----------|---------|
| 1 | [01_why_temporal_cv](../../notebooks/01_why_temporal_cv.ipynb) | KFold leakage, WalkForwardCV, diagnose your data |
| 2 | [05_shuffled_target_gate](../../notebooks/05_shuffled_target_gate.ipynb) | Definitive leakage detection |
| 3 | [08_validation_workflow](../../notebooks/08_validation_workflow.ipynb) | Complete HALT/WARN/PASS pipeline |
| 4 | [10_high_persistence_metrics](../../notebooks/10_high_persistence_metrics.ipynb) | MASE, MC-SS, move-conditional metrics |

**Troubleshooting**: [Diagnostic Flowchart](diagnostic_flowchart.md) — What to do when validation fails

---

## Already Know Time-Series?

If you understand ACF, stationarity, and temporal dependence, skip Tier 0:

| Step | Notebook | Concept |
|------|----------|---------|
| 1 | [01_why_temporal_cv](../../notebooks/01_why_temporal_cv.ipynb) | WalkForwardCV, temporalcv gates |
| 2 | [05_shuffled_target_gate](../../notebooks/05_shuffled_target_gate.ipynb) | Definitive leakage detection |
| 3 | [08_validation_workflow](../../notebooks/08_validation_workflow.ipynb) | Complete HALT/WARN/PASS pipeline |

See [notebooks/README.md](../../notebooks/README.md) for the complete 14-hour curriculum.

---

## Tutorial Documents

Quick-reference markdown tutorials for specific topics:

```{toctree}
:maxdepth: 1

feature_engineering_safety
metric_selection
diagnostic_flowchart
leakage_detection
walk_forward_cv
high_persistence
uncertainty
```

### By Topic

**New Guides for Newcomers:**
- **[Feature Engineering Safety](feature_engineering_safety.md)** — Safe vs dangerous features, decision tree
- **[Metric Selection](metric_selection.md)** — Which metric for which problem
- **[Diagnostic Flowchart](diagnostic_flowchart.md)** — Troubleshooting validation failures

**Core Topics:**
- **[Leakage Detection](leakage_detection.md)** — Detect and prevent data leakage with validation gates
- **[Walk-Forward CV](walk_forward_cv.md)** — Time-series cross-validation with gap enforcement
- **[High Persistence](high_persistence.md)** — Handle high-persistence series with MC-SS metrics
- **[Uncertainty Quantification](uncertainty.md)** — Conformal prediction and bootstrap intervals

---

## Interactive Notebooks vs. Markdown Tutorials

| Format | Best For | Location |
|--------|----------|----------|
| **Jupyter Notebooks** | Learning, experimentation, running examples | `notebooks/` |
| **Markdown Tutorials** | Quick reference, searching, documentation | `docs/tutorials/` |

Both formats cover similar content. Start with notebooks for learning, use tutorials for quick lookups.

---

## Tier 1: Foundation Notebooks (Available)

These notebooks teach *why* time-series validation is fundamentally different from standard ML:

| Notebook | Key Concept | Interactive Link |
|----------|-------------|------------------|
| 01_why_temporal_cv | KFold leakage, WalkForwardCV, persistence baseline | [Open](../../notebooks/01_why_temporal_cv.ipynb) |
| 02_gap_enforcement | h-step forecasting, gap >= horizon rule | [Open](../../notebooks/02_gap_enforcement.ipynb) |
| 03_persistence_baseline | MASE, why persistence is hard to beat | [Open](../../notebooks/03_persistence_baseline.ipynb) |
| 04_autocorrelation_matters | HAC variance, MA(h-1) error structure | [Open](../../notebooks/04_autocorrelation_matters.ipynb) |

## Tier 2: Prevention Notebooks (Available)

These notebooks teach *how* to detect and prevent common leakage patterns:

| Notebook | Key Concept | Interactive Link |
|----------|-------------|------------------|
| 05_shuffled_target_gate | Permutation testing for leakage detection | [Open](../../notebooks/05_shuffled_target_gate.ipynb) |
| 06_feature_engineering_pitfalls | Safe rolling stats, feature selection | [Open](../../notebooks/06_feature_engineering_pitfalls.ipynb) |
| 07_threshold_leakage | Regime/percentile computation without lookahead | [Open](../../notebooks/07_threshold_leakage.ipynb) |
| 08_validation_workflow | Complete HALT/WARN/PASS pipeline | [Open](../../notebooks/08_validation_workflow.ipynb) |

## Tier 3: Evaluation Notebooks (Available)

These notebooks teach *how* to properly evaluate high-persistence data and compare models:

| Notebook | Key Concept | Interactive Link |
|----------|-------------|------------------|
| 09_statistical_tests_dm_pt | DM test, PT test, HAC variance | [Open](../../notebooks/09_statistical_tests_dm_pt.ipynb) |
| 10_high_persistence_metrics | MC-SS, move-conditional, Theil's U | [Open](../../notebooks/10_high_persistence_metrics.ipynb) |
| 11_conformal_prediction | Split/Adaptive conformal, intervals | [Open](../../notebooks/11_conformal_prediction.ipynb) |
| 12_regime_stratified_evaluation | Volatility regimes, stratified gates (capstone) | [Open](../../notebooks/12_regime_stratified_evaluation.ipynb) |
