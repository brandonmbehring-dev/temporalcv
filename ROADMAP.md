# temporalcv Roadmap

**Last Updated**: 2025-12-31

---

## Current Version: 1.0.0-rc1

### Status: Release Candidate
- Code complete and stabilized
- Tests passing (1,741)
- Documentation updated
- Ready for community feedback

---

## v1.0.0 (Final Release)

**Target**: Q1 2025

### Remaining Items
- [ ] Community feedback integration

### Already Complete
- [x] Full M4/M5 benchmark comparison table (run_c700bfd9, 4,773 series, 14.3 min)
- [x] End-to-end benchmark test with bundled dataset (21 tests, M4 + statsforecast)
- [x] Two-mode shuffled target gate (permutation + effect_size)
- [x] Strict mode for get_n_splits()
- [x] Gap/horizon validation in WalkForwardCV
- [x] Comprehensive DM test limitations documentation
- [x] Model cloning in gates
- [x] Hub infrastructure (.tracking/, requirements.lock)

---

## v1.1.0 (Planned)

**Target**: Q2 2025

### Statistical Extensions
- [x] Self-normalized DM test (robust to bandwidth) — `variance_method="self_normalized"`
- [x] Giacomini-White test (conditional predictive ability) — `gw_test()`
- [x] Clark-West test (nested model comparison) — `cw_test()`

### Enhanced Features
- [x] Nested CV for hyperparameter tuning — `NestedWalkForwardCV`
- [x] Multi-horizon comparison utilities — `compare_horizons()`, `compare_models_horizons()`
- [x] Blocked bootstrap CI for gates — `bootstrap_ci=True` in gates

### Infrastructure
- [x] Model cards for WalkForwardCV, gate_shuffled_target (docs/model_cards/)
- [x] Enhanced coverage diagnostics for conformal — `CoverageDiagnostics`, `compute_coverage_diagnostics()`

---

## v1.2.0 (Future)

**Target**: Q3 2025

### Advanced Methods
- [x] Bellman Conformal Inference (optimal adaptation) — `BellmanConformalPredictor`
- [x] Reality Check / SPA test (multiple comparison) — `reality_check_test()`, `spa_test()`
- [x] Forecast encompassing test — `forecast_encompassing_test()`, `forecast_encompassing_bidirectional()`

### Integration
- [x] research-kb integration hooks — MCP-based, documented in CLAUDE.md
- [ ] Julia package registered in General

---

## Julia Package

### Current: v0.1.0
- Feature parity ~95% with Python
- 63 modules implemented
- 37 test files

### v1.0.0 (Target: Q1 2025)
- [ ] Benchmark dataset loaders (idiomatic Julia)
- [ ] Register in Julia General registry
- [ ] Documentation site (Documenter.jl)

---

## Contribution Areas

### High Impact
1. **Benchmark comparisons**: M4/M5 results vs competitors
2. **Real-world examples**: Production pipeline integration
3. **Documentation**: Additional tutorials

### Medium Impact
4. **Statistical extensions**: New tests and methods
5. **Performance**: Optimization for large datasets
6. **Julia ecosystem**: Integration with MLJ, Flux

---

## Versioning Policy

- **Patch** (x.x.N): Bug fixes, documentation
- **Minor** (x.N.0): New features, backward compatible
- **Major** (N.0.0): Breaking changes (rare)

See CHANGELOG.md for detailed history.
