# temporalcv v2.0 roadmap

Tracks the v2.0 **universal time-series toolkit** modernization. Breaking changes are acceptable
because `dml_ts` is the sole consumer, migrated in lockstep — see
`docs/adr/0001-v2-seams-and-layout.md`, `STYLE.md`, and hub
`patterns/universal-vs-unique.md` / `patterns/library-design-playbook.md`. Each item is a filed
GitHub issue.

## Capability
- [ ] #7  `BlockedTimeSeriesCV` (whole-block-preserving CV) — port from dml_ts.
- [ ] #8  dual-model `cross_fit_residualize(model_a, model_b, X, A, B, cv) -> (A_resid, B_resid)` with fold-0 NaN-mask.
- [ ] #9  standalone HAC residual covariance (Bartlett/Parzen/QS + optimal bandwidth + Newey-West), **matrix-accepting** (panel-ready).
- [ ] #10 numeric/stat output validators (`finite_se`, `psd`, `ci_ordered`, `coverage_in_unit`).
- [ ] #11 generic AR/ARMA time-series simulators.

## Design (breaking → v2.0)
- [ ] #12 unify seam vocabulary → static `@runtime_checkable` Protocols + sklearn base + mixins.
- [ ] #13 frozen result objects (`frozen`+`slots`+`__post_init__`+`to_dict` + versioned JSON schema).
- [ ] #14 capabilities-as-tags + executable conformance suite (`check_temporal_splitter`/`check_temporal_estimator`).
- [ ] #15 backend-agnostic contract (`ArrayLike` sigs + lazy splitter seam; reserve `xp`/narwhals).
- [ ] #16 governance: layout/public-contract ADR + public-API stability test; fix `pyproject` URLs (→ canonical brandon-behring).

## Follow-up
- [ ] #17 eval-toolkit ↔ temporalcv purged-splitter overlap (keep separate per hub `universal-vs-unique.md`).

## Pilot
The **CrossFitter seam** (the #8 / #12 slice) is the vertical pilot validating the approach
end-to-end before the rest of v2.0. Status: `src/temporalcv/protocols.py` (`Splitter`/`CrossFitter`
static Protocols) added; next = `cross_fit_residualize` + conformance suite + dml_ts golden-parity.
