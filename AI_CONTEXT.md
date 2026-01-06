# AI_CONTEXT.md - temporalcv

**Version**: 1.0.0-rc1 | **Status**: v1.0 Release Candidate

---

## Project Overview

**temporalcv**: Temporal cross-validation with leakage protection for time-series ML.

**Purpose**: Fill ecosystem gap for rigorous time-series validation — shuffled target tests, gap enforcement, statistical testing (DM, PT), and high-persistence handling.

**Audience**: ML practitioners needing statistical rigor without leaving sklearn ecosystem.

---

## Quick Start

```bash
# Development setup
pip install -e ".[dev]"
pytest tests/ -v
mypy src/temporalcv
```

---

## Core Principles (Priority Order)

1. **NEVER FAIL SILENTLY** - Every error must be explicitly reported
2. **External-First Validation** - Shuffled target → Synthetic AR(1) → Internal tests
3. **Temporal Safety** - Gap enforcement: `gap >= horizon` always
4. **Skepticism of Success** - >20% improvement = HALT and investigate

---

## Seed Policy

All stochastic functions accept `random_state: int | None` parameter:
- **Default**: `random_state=None` (non-deterministic for exploration)
- **Tests**: Always use fixed seed for reproducibility
- **Gates**: Shuffled target uses `random_state` with documented default

---

## Package Structure

```
src/temporalcv/
├── __init__.py
├── py.typed              # PEP 561 type marker
├── gates/                # Validation gates (HALT/PASS/WARN/SKIP)
├── tests/                # Statistical tests (DM, PT)
├── cv/                   # Walk-forward cross-validation
├── metrics/              # MC-SS, move-only MAE
└── benchmarks/           # Dataset loaders
```

---

## Gate Status Codes

The validation gates return status codes for programmatic use:

| Status | Meaning |
|--------|---------|
| PASS | All checks passed |
| HALT | Critical failure - stop and investigate |
| WARN | Caution - proceed with verification |
| SKIP | Insufficient data for test |

---

## Development Standards

### Code Style
- **Formatter**: ruff (black-compatible, 100-char lines)
- **Type hints**: Required on all public functions
- **Docstrings**: NumPy style

### Testing
- Real tests only — no stubs, TODOs, or placeholders
- Coverage target: 80%+ for modules
- Anti-pattern tests for leakage detection

### Git
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`
- Pre-commit hooks must pass

---

## Knowledge Tier System

All claims in documentation and docstrings are tagged by confidence level:

| Tier | Meaning | Example | Action |
|------|---------|---------|--------|
| **[T1]** | Academically validated with full citation | DM test (Diebold & Mariano 1995) | Trust and apply |
| **[T2]** | Empirical finding from prior work | 70th percentile threshold from v2 | Apply, verify in new contexts |
| **[T3]** | Assumption needing justification | 13-week volatility window | Consider sensitivity analysis |

**When working with tiered knowledge**:
- **T1**: Apply directly; these are established statistical methods
- **T2**: Apply but monitor; may need adjustment for different domains
- **T3**: Question first; document why assumption holds for your data

---

## Documentation Index

| Document | Purpose | Location |
|----------|---------|----------|
| **SPECIFICATION.md** | Frozen parameters, authoritative thresholds | `SPECIFICATION.md` |
| Mathematical foundations | DM, HAC, PT, Conformal derivations [T1] | `docs/knowledge/mathematical_foundations.md` |
| Assumptions | Explicit [T3] assumptions by module | `docs/knowledge/assumptions.md` |
| Notation | Variable definitions | `docs/knowledge/notation.md` |
| Leakage audit trail | Bug postmortems, gate mappings | `docs/knowledge/leakage_audit_trail.md` |
| Planning index | Phase tracking | `docs/plans/INDEX.md` |
| API design | Code examples | `docs/plans/reference/api_design.md` |
| Ecosystem gaps | What we fill | `docs/plans/reference/ecosystem_gaps.md` |

---

## Hub Integration

**This project is a spoke in the lever_of_archimedes hub-and-spoke architecture.**

| Attribute | Value |
|-----------|-------|
| **Hub** | `~/Claude/lever_of_archimedes/` |
| **Status** | Integrated (2025-12-23) |
| **Relationship Doc** | `docs/HUB_RELATIONSHIP.md` |

**Patterns Used**:
- `patterns/testing.md` - 6-layer validation architecture
- `patterns/data_leakage_prevention.md` - 10 bug categories
- `patterns/ds_ml_lifecycle.md` - Phase progression standards

**What This Spoke Contributes**:
- Temporal cross-validation primitives (`WalkForwardCV`)
- Leakage detection gates (shuffled target, synthetic AR(1))
- Statistical testing (DM, PT with HAC variance)
- High-persistence metrics (MC-SS, move-conditional)
- Conformal prediction for time series

---

## Research Context (MCP Integration)

**Optional**: Claude Code can query research-kb for methodological background on statistical tests.

**Available Queries**:
```python
# Query time_series domain for method documentation
research_kb_search("Diebold Mariano test", domain="time_series")
research_kb_search("HAC variance estimation", domain="time_series")
research_kb_search("conformal prediction coverage", domain="time_series")

# Cross-domain for causal inference connections
research_kb_cross_domain_concepts(concept_name="instrumental variables",
                                   source_domain="causal_inference",
                                   target_domain="time_series")
```

**Key Sources** (with research-kb source IDs when available):
| Method | Paper | Year | Status |
|--------|-------|------|--------|
| DM Test | Diebold & Mariano | 1995 | [T1] Core |
| PT Test | Pesaran & Timmermann | 1992 | [T1] Core |
| GW Test | Giacomini & White | 2006 | [T1] Core |
| CW Test | Clark & West | 2007 | [T1] Core |
| Reality Check | White | 2000 | [T1] Core |
| SPA Test | Hansen | 2005 | [T1] Core |
| Forecast Encompassing | Harvey et al. | 1998 | [T1] Core |
| HAC Variance | Newey & West | 1987 | [T1] Core |
| Conformal Prediction | Vovk et al. | 2005 | [T1] Core |
| Adaptive Conformal | Gibbs & Candès | 2021 | [T1] Core |

**Philosophy**: temporalcv = pure validation library; research-kb = developer tooling for methodological context.

---

## Reference Patterns

This project follows patterns from:
- `~/Claude/lever_of_archimedes/patterns/testing.md` - 6-layer validation architecture
- `~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md` - 10 bug categories, ML guardrails
- `~/Claude/lever_of_archimedes/patterns/ds_ml_lifecycle.md` - Phase progression standards
- `~/Claude/myga-forecasting-v4/CLAUDE.md` - Domain-specific validation, episode documentation

---

## Suspicious Results Protocol

**Automatic HALT triggers** — if any condition is met, validation halts for investigation.

| Condition | Threshold | Gate to Run | What It Catches |
|-----------|-----------|-------------|-----------------|
| Improvement over persistence | >20% | `gate_suspicious_improvement()` | Leakage, overfitting |
| MAE below theoretical minimum | <0.67σ | `gate_synthetic_ar1()` | Lookahead bias |
| Model beats shuffled target | p < 0.05 | `gate_signal_verification()` | Feature-target alignment |
| h=1 >> h=2,3,4 performance | >2x better | `gate_temporal_boundary()` | Gap enforcement failure |

**Decision flowchart**:
```
1. Run gate_signal_verification() FIRST
   └─ HALT? → Feature encodes target position → FIX
   └─ PASS? → Continue

2. Run gate_synthetic_ar1() on AR(1) data
   └─ HALT? → Beating theoretical bounds → INVESTIGATE
   └─ PASS? → Continue

3. Run gate_suspicious_improvement()
   └─ HALT? → >20% improvement → Verify with external data
   └─ WARN? → 10-20% improvement → Proceed with caution
   └─ PASS? → Safe to deploy
```

**Bug categories prevented** (see `docs/knowledge/leakage_audit_trail.md`):
1. Lag leakage (full series computation)
2. Temporal boundary violations (no gap)
3. Threshold leakage (full series percentiles)
4. Feature selection on target
5. Regime computation lookahead

---

## Amendment Process

Changes to default thresholds, gate logic, or statistical methodology require:

1. **Documented justification** - Why the change is needed
2. **Backward compatibility note** - Impact on existing users
3. **CHANGELOG entry** - Version and description
4. **Test coverage** - Tests validating new behavior
5. **Tier tag update** - If changing from [T3] to [T2], document the validation
