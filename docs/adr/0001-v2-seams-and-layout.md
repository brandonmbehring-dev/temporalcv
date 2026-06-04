# ADR 0001 — v2.0 seam strategy & module layout

**Status:** Accepted (2026-06-04)
**Context:** v2.0 is a breaking modernization to the hub `library-design-playbook`. `dml_ts` is the
**sole consumer**, migrated in lockstep, so breaking changes are acceptable now (vs additive-only).

## Decision
1. **Seam strategy:** static `@runtime_checkable` Protocols (typed seam) + sklearn base inheritance
   (interop) + mixins/ABCs we own (shared impl) + **capabilities-as-tags** + an executable
   conformance suite. This replaces the current inconsistent vocabulary (`BootstrapStrategy` and
   `ForecastAdapter` are ABCs; `SupportsPredict` is a Protocol).
2. **Result objects:** `frozen=True, slots=True` + `__post_init__` + `to_dict()` + versioned JSON
   schema; point estimate = degenerate distributional case.
3. **Contract future-proofing:** `ArrayLike` signatures, lazy splitter seam (`get_n_splits()` may be
   `None`), reserved internal `xp`/narwhals seams. Implementations stay numpy/batch/point.
4. **Layout / public contract:** the stable v2.0 surface is the top-level `temporalcv` namespace via
   `__all__`. Existing subpackages (`compare/`, `bagging/`, `metrics/`) are implementation grouping,
   **not** stable import paths — consumers import from `temporalcv`, not `temporalcv.bagging.base`.
5. **Stability tiers:** a named frozen Tier-2 Protocol set gated by a public-API test; new capability
   arrives as additive sub-Protocols, never by widening the core.

## Consequences
- (+) One consistent extension story; typed consumers; sklearn interop; future
  backends/streaming/distributional become additive, not breaking.
- (−) One-time breaking churn → **v2.0.0** (acceptable: single consumer migrated together).
- Enforcement: conformance suite + public-API test in CI; `dml_ts` golden-parity gates validate
  behavioral equivalence during its migration.
