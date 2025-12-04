# Ecosystem Gap Analysis

**Purpose**: Document what temporalcv fills that existing packages don't.

---

## Critical Gaps (No Solution Anywhere)

| Gap | Python | R | Julia | temporalcv |
|-----|--------|---|-------|------------|
| **Leakage detection framework** | ❌ | ❌ | ❌ | `gates.py` + anti-pattern tests |
| **Shuffled target test** | ❌ | ❌ | ❌ | `gate_shuffled_target()` |
| **Suspicious improvement detection** | ❌ | ❌ | ❌ | `gate_suspicious_improvement()` |
| **3-stage validation gates** | ❌ | ❌ | ❌ | External → Internal → Statistical |

## Partial Coverage Gaps

| Gap | Existing | temporalcv Addition |
|-----|----------|---------------------|
| Walk-forward with gap | sktime (good), Darts (`output_chunk_shift`) | Cross-package validation + postconditions |
| DM test | `dieboldmariano` PyPI, `feval` | Integrated with gates + auto-threshold |
| PT test (3-class) | ❌ | `pt_test()` with move_threshold |
| Conformal | MAPIE `TimeSeriesRegressor` | Regime-aware calibration windows |

## Package Comparison

### vs sktime

| Feature | sktime | temporalcv |
|---------|--------|------------|
| Walk-forward CV | ✅ Good | ✅ + strict temporal safety |
| Leakage detection | ❌ | ✅ Novel (gates framework) |
| DM test | ❌ | ✅ With HAC variance |
| High-persistence | ❌ | ✅ Specialized tools |

### vs Darts / Nixtla / GluonTS

| Feature | Others | temporalcv |
|---------|--------|------------|
| Focus | Model architectures | Validation rigor |
| Leakage prevention | ❌ | ✅ Framework-level |
| Statistical testing | ❌ | ✅ Comprehensive |

## Known Issues in Other Packages

| Package | Issue | GitHub Ref |
|---------|-------|------------|
| sktime | FourierFeatures pipeline produces incorrect results | #5975 |
| sktime | SARIMAX with exogenous fails in update_predict | #8096 |
| Nixtla | Individual vs global forecast mismatch | #455 |
| Nixtla | Lag leakage in distributed mode | #438 |
| Darts | Scaler fit on validation set leaks | PR #2529 |
| GluonTS | iid assumption violation not addressed | #1887 |

## Unique Value Propositions

**Truly unique**:
1. Shuffled target test — Definitive leakage detection gate
2. HALT/PASS/WARN/SKIP gates — Composable validation framework
3. >20% improvement trigger — Automated suspicion protocol
4. MC-SS + move thresholds — Event-aware metrics for high-persistence

**Integration differentiators**:
5. DM test + gates — `dieboldmariano` exists, not integrated
6. Conformal + regime — MAPIE exists, not regime-aware
7. Gap enforcement + validation — Darts has shift, no cross-package audit

## Target Audience

Python ML practitioners who need statistical rigor without leaving sklearn ecosystem.

## Academic Foundation

| Paper | Key Insight |
|-------|-------------|
| Tashman (2000) | Walk-forward yields robust error measures |
| Hewamalage et al. (2023) | 6 Evaluation Pitfalls bridges statistician/ML |
| Diebold & Mariano (1995) | Non-quadratic loss, serially correlated errors |
| Harvey et al. (1997) | Small-sample correction critical for h=2 |
| Stock & Watson (2001) | Structural instability pervasive |
