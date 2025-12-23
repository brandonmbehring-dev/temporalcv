# Changelog

All notable changes to temporalcv will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Phase E: Testing Infrastructure
- 6-tier test architecture: known-answer, Monte Carlo, golden reference, adversarial
- 72 new tests (total: 628)
- Nightly CI workflow for Monte Carlo validation (`.github/workflows/nightly-tests.yml`)
- Golden reference JSON with R-validated DM test cases (`tests/fixtures/golden_reference.json`)
- Monte Carlo calibration tests for DM, wild bootstrap, conformal coverage
- Adversarial edge case tests for numerical stability

#### Phase D: Documentation
- Sphinx infrastructure with napoleon, autodoc, intersphinx (`docs/conf.py`)
- 12 API reference stubs in `docs/api_reference/`
- Knowledge Tier glossary (T1/T2/T3) in `docs/glossary.rst`
- ReadTheDocs deployment configuration (`.readthedocs.yaml`)

#### Phase C: Larger Features
- Wild cluster bootstrap for few-fold inference (`inference/wild_bootstrap.py`)
  - Auto weight selection: Webb <13 folds, Rademacher ≥13
  - `WildBootstrapResult` dataclass with p-value and CI
- `CrossFitCV` for debiased out-of-sample predictions
  - Forward-only temporal semantics
  - `fit_predict()` and `fit_predict_residuals()` methods
- `run_gates_stratified()` for regime-conditional validation
  - `StratifiedValidationReport` with per-regime gate results

#### Phase B: Diagnostics
- `gate_theoretical_bounds()` - AR(1) theoretical minimum validation
  - Estimates phi from ACF(1), computes innovation variance
  - AR(1) assumption check via Ljung-Box on residuals
- `compute_dm_influence()` - Observation and block influence diagnostics
  - HAC-adjusted observation-level influence
  - Block jackknife for theoretically-justified decisions
- `gap_sensitivity_analysis()` - Gap parameter sensitivity
  - Configurable degradation threshold (default 10%)
  - Returns break-even gap and sensitivity score

#### Phase A: Quick Wins
- Frozen `SplitInfo` dataclass (immutable)
- `gate_residual_diagnostics()` - residual quality checks
  - Custom Ljung-Box test for autocorrelation
  - Jarque-Bera test for normality
  - Mean-zero t-test

#### Other Additions
- SPECIFICATION.md with frozen parameters and mathematical definitions
- Knowledge tier system documentation ([T1]/[T2]/[T3])
- Leakage audit trail documenting 10 bug categories
- Episode documentation for common leakage patterns
- Expanded CLAUDE.md with decision flowchart and gate references

### Changed
- Updated phase status to reflect Phase 5 completion
- Enhanced Suspicious Results Protocol with concrete thresholds
- Updated Reference Patterns to include myga-forecasting-v4

### Fixed

#### Phase 0: Correctness Fixes
- DM test uses t-distribution when `harvey_correction=True` (was normal)
- Gates use WalkForwardCV internally for out-of-sample evaluation (was in-sample)
- Conformal quantile uses `method="higher"` for conservative coverage
- PT test minimum samples increased to 30 (was 20)
- phi validation (-1 < phi < 1) added to `gate_synthetic_ar1`
- SPECIFICATION.md synced with code (n_shuffles=5, block_len=n^(1/3), DM min=30)

#### Other Fixes
- GitHub URL consistency (unified to brandonmbehring-dev)

## [0.1.0-alpha] - 2025-12-23

### Added

#### Phase 5: Benchmarks and Event Metrics
- Benchmark dataset loaders: FRED, M5, Monash (M3/M4), GluonTS
- Model comparison framework with adapter pattern
- Event metrics: Brier score, PR-AUC, direction probability
- `create_synthetic_ar1()` for controlled testing
- 115 new tests (366 total)

#### Phase 4: Uncertainty and Ensemble Methods
- Split conformal prediction with coverage guarantee
- Adaptive conformal inference for distribution shift (Gibbs & Candès 2021)
- Bootstrap uncertainty estimation
- Time-series bagging: block bootstrap, stationary bootstrap, feature bagging
- `walk_forward_conformal()` for online prediction intervals
- `evaluate_interval_quality()` metrics
- 89 new tests (251 total)

#### Phase 3: High-Persistence Handling
- Move-Conditional Skill Score (MC-SS) for sticky series
- `compute_move_threshold()` with training-only enforcement
- `classify_moves()` for UP/DOWN/FLAT classification
- `compute_move_only_mae()` excluding flat periods
- `compute_direction_accuracy()` with 2-class and 3-class modes
- Regime classification: volatility and direction regimes
- `mask_low_n_regimes()` for small-sample protection
- 58 new tests (162 total)

#### Phase 2: Walk-Forward CV Infrastructure
- `WalkForwardCV` with sliding and expanding windows
- Gap parameter enforcement (`gap >= horizon`)
- `SplitInfo` dataclass for split metadata
- sklearn-compatible splitter API
- 40 new tests (104 total)

#### Phase 1: Validation Gates and Statistical Tests
- `gate_shuffled_target()` - definitive leakage detection
- `gate_synthetic_ar1()` - theoretical bounds validation
- `gate_suspicious_improvement()` - >20% improvement trigger
- `gate_temporal_boundary()` - gap enforcement validation
- `run_gates()` - aggregation with HALT > WARN > PASS priority
- `dm_test()` - Diebold-Mariano with HAC variance
- `pt_test()` - Pesaran-Timmermann direction test
- `compute_hac_variance()` - Bartlett kernel estimator
- `GateStatus`, `GateResult`, `ValidationReport` types
- 64 tests with 97% coverage

#### Phase 0: Repository Setup
- Package structure with `src/` layout
- PEP 561 type markers (`py.typed`)
- pyproject.toml with hatchling build
- Pre-commit hooks (ruff, mypy)
- GitHub Actions CI (Python 3.9-3.12)
- CLAUDE.md with project principles
- Knowledge tier system documentation

### Technical Notes

**Dependencies**:
- Core: numpy>=1.21, scipy>=1.7, scikit-learn>=1.0
- Optional: pandas, fredapi, gluonts, statsforecast

**Coverage**: 86% overall, 95%+ on core modules

**Test Count**: 628 tests (627 passing, 1 skipped)

---

## Version History

| Version | Date | Phase | Highlights |
|---------|------|-------|------------|
| 0.1.0-alpha | 2025-12-23 | 5 | Complete feature set |
| 0.0.5-dev | 2025-12-XX | 4 | Uncertainty + ensemble |
| 0.0.4-dev | 2025-12-XX | 3 | High-persistence |
| 0.0.3-dev | 2025-12-XX | 2 | Walk-forward CV |
| 0.0.2-dev | 2025-12-XX | 1 | Gates + tests |
| 0.0.1-dev | 2025-12-XX | 0 | Repository setup |

---

## Migration Notes

### From Unreleased to 0.1.0

No breaking changes. New features are additive.

### Future Breaking Changes (v1.0)

The following may change before v1.0:
- Default threshold values (currently [T3])
- CLI exit codes
- Minimum n requirements for tests

See SPECIFICATION.md for current frozen values.
