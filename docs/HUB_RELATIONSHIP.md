# Hub Relationship: temporalcv ↔ lever_of_archimedes

**Status**: Integrated | **Date**: 2025-12-23

---

## Overview

temporalcv is a **spoke** project in the lever_of_archimedes hub-and-spoke architecture.

| Attribute | Value |
|-----------|-------|
| **Hub Location** | `~/Claude/lever_of_archimedes/` |
| **Spoke Location** | `~/Claude/temporalcv/` |
| **Integration Date** | 2025-12-23 |
| **Primary Domain** | Time-series ML validation |

---

## Pattern Compliance

| Pattern | Status | Implementation |
|---------|--------|----------------|
| 6-layer validation | ✅ Complete | `tests/property/` (Layer 6), unit tests (Layer 3-5) |
| Data leakage prevention | ✅ Complete | All 10 bug categories addressed in gates |
| Knowledge tiers | ✅ Complete | [T1]/[T2]/[T3] in `docs/knowledge/` |
| Session workflow | ✅ Complete | CURRENT_WORK.md at project root |
| Precision feedback | ✅ Complete | AI_CONTEXT.md output style integration |
| Git workflow | ✅ Complete | Conventional commits in practice |

---

## What This Spoke Contributes

temporalcv provides **temporal cross-validation primitives** for the ML ecosystem:

### Core Capabilities

1. **Validation Gates** (HALT/WARN/PASS/SKIP framework)
   - `gate_signal_verification()` - Definitive leakage detection
   - `gate_synthetic_ar1()` - Theoretical bounds validation
   - `gate_suspicious_improvement()` - >20% improvement trigger
   - `gate_temporal_boundary()` - Gap enforcement

2. **Statistical Testing**
   - Diebold-Mariano test with HAC variance [T1]
   - Pesaran-Timmermann direction test [T1]
   - Proper small-sample corrections (Harvey adjustment)

3. **Walk-Forward CV**
   - `WalkForwardCV` with sliding/expanding windows
   - Gap enforcement (`gap >= horizon`)
   - sklearn-compatible splitter API

4. **High-Persistence Metrics**
   - MC-SS (Move-Conditional Skill Score) [T2]
   - Move-only MAE excluding flat periods
   - Direction accuracy (2-class and 3-class)

5. **Conformal Prediction**
   - Split conformal with coverage guarantee [T1]
   - Adaptive conformal for distribution shift [T1]
   - Walk-forward conformal intervals

---

## Hub Resources Used

### Patterns Applied

| Pattern File | How Used |
|--------------|----------|
| `patterns/testing.md` | 6-layer structure, property-based tests |
| `patterns/data_leakage_prevention.md` | 10 bug categories → gate design |
| `patterns/ds_ml_lifecycle.md` | Phase 0-5 progression, coverage targets |
| `patterns/git.md` | Commit format, attribution |

### Templates Followed

- AI_CONTEXT.md structure from hub template
- SPECIFICATION.md pattern (frozen parameters)
- Episode documentation from myga-forecasting-v4

---

## Cross-Project Dependencies

### Inbound (Used by temporalcv)

| Project | What We Use |
|---------|-------------|
| lever_of_archimedes | Core patterns, templates |
| myga-forecasting-v4 | Episode documentation pattern, [T2] thresholds |

### Outbound (Could Use temporalcv)

| Project | Potential Use |
|---------|--------------|
| myga-forecasting-v4 | `WalkForwardCV`, validation gates |
| annuity_forecasting | Leakage detection for forecasting |
| double_ml_time_series | Temporal CV for causal inference |

---

## Implementation Notes

### Pattern Adaptations

1. **6-Layer Validation**:
   - Layer 6 (property-based) uses Hypothesis for gate invariants
   - Layer 3-5 traditional pytest structure
   - Layer 1-2 via mypy strict + input validation in gates

2. **Data Leakage Prevention**:
   - All 10 bug categories mapped to specific gates
   - Episode documentation explains each mapping
   - `docs/knowledge/leakage_audit_trail.md` tracks postmortems

3. **Knowledge Tiers**:
   - [T1] for academically validated methods (DM, PT, HAC, Conformal)
   - [T2] for empirical findings (MC-SS, 70th percentile threshold)
   - [T3] for assumptions (20% improvement trigger)

---

## Verification Checklist

- [x] AI_CONTEXT.md references hub patterns
- [x] Hub Integration section in AI_CONTEXT.md
- [x] HUB_RELATIONSHIP.md exists (this file)
- [x] Knowledge tiers applied to documentation
- [x] Pattern compliance documented above
- [x] Listed in hub README spoke registry
- [x] Session workflow files (CURRENT_WORK.md)

---

## Amendment History

| Date | Change | Reason |
|------|--------|--------|
| 2025-12-23 | Initial integration | v1.0 preparation audit |
