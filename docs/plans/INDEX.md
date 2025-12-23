# temporalcv Planning Index

**Package**: temporalcv - Temporal CV with leakage protection
**Status**: Phase 5 Complete, v1.0 Preparation

---

## Current Work

→ Phase 1: Complete ✓ (97% coverage, 64 tests)
→ Phase 2: Complete ✓ (96% coverage, 104 tests)
→ Phase 3: Complete ✓ (95% coverage, 162 tests)
→ Phase 4: Complete ✓ (96% coverage, 251 tests)
→ Phase 5: Complete ✓ (85% coverage, 366 tests) - 2025-12-23
→ Next: v1.0 Preparation (documentation, patterns, release)

## Quick Navigation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| `reference/ecosystem_gaps.md` | What gap we fill | Understanding positioning |
| `reference/api_design.md` | Ideal API examples | Designing interfaces |
| `reference/implementation_roadmap.md` | Phase timeline | Planning sprints |
| `reference/benchmark_strategy.md` | Test datasets | Validation design |
| `reference/julia_strategy.md` | Julia port plan | Future porting |
| `archive/original_plan.md` | Full research | Historical reference |

## Phase Checklist

- [x] **Phase 0**: Repository setup (pyproject.toml, CI, CLAUDE.md)
- [x] **Phase 1**: Core gates + statistical tests (64 tests, 97% coverage)
- [x] **Phase 2**: Walk-forward CV infrastructure (104 tests, 96% coverage)
- [x] **Phase 3**: High-persistence handling (162 tests, 95% coverage)
- [x] **Phase 4**: Uncertainty + ensemble (251 tests, 96% coverage)
- [x] **Phase 5**: Benchmark suite + event metrics (366 tests, 85% coverage) ✓ 2025-12-23
- [ ] **Phase 6**: Julia port (Deferred until v1.0 stable)

## Key Decisions

1. **Package name**: `temporalcv` (technical precision, CV-focused)
2. **Target audience**: ML practitioners needing statistical rigor
3. **Python first**: Julia native port after v1.0 stable
4. **Scope**: Broad (validation + testing + CV + regime + conformal)

## What Makes temporalcv Unique

**Truly novel** (no existing package):
- Shuffled target test — definitive leakage detection
- HALT/PASS/WARN/SKIP gates — composable validation framework
- >20% improvement trigger — automated suspicion protocol
- MC-SS + move thresholds — event-aware metrics for high-persistence

**Integration differentiators** (components exist, orchestration doesn't):
- DM test + gates — existing test, but integrated with validation
- Conformal + regime — MAPIE exists, not with regime-aware calibration
- Gap enforcement + audit — Darts has shift, no cross-package validation
