# temporalcv Planning Index

**Package**: temporalcv - Temporal CV with leakage protection
**Status**: Phase 0 Complete, Phase 1 Ready

---

## Current Work

→ Phase 1: Complete ✓
→ Phase 2: Complete ✓
→ Next: Phase 3 (High-persistence handling)

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
- [ ] **Phase 3**: High-persistence handling (Weeks 9-12)
- [ ] **Phase 4**: Uncertainty + ensemble (Weeks 13-16)
- [ ] **Phase 5**: Benchmark suite + event metrics (Weeks 17-20)
- [ ] **Phase 6**: Julia port (Weeks 21-28)

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
