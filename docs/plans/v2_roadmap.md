# temporalcv v2.0 roadmap

Tracks the v2.0 **universal time-series toolkit** modernization. Breaking changes are acceptable
because `dml_ts` is the sole consumer, migrated in lockstep — see
`docs/adr/0001-v2-seams-and-layout.md`, `STYLE.md`, and hub
`patterns/universal-vs-unique.md` / `patterns/library-design-playbook.md`. Each item is a filed
GitHub issue.

## Capability
- [ ] #7  `BlockedTimeSeriesCV` (whole-block-preserving CV) — port from dml_ts.
- [x] #8  dual-model `cross_fit_residualize(model_a, model_b, X, A, B, cv) -> (A_resid, B_resid)` with fold-0 NaN-mask. **Done (pilot)** — `src/temporalcv/cv.py`, typed `cv: Splitter`, identical shared NaN mask; exported top-level.
- [ ] #9  standalone HAC residual covariance (Bartlett/Parzen/QS + optimal bandwidth + Newey-West), **matrix-accepting** (panel-ready).
- [ ] #10 numeric/stat output validators (`finite_se`, `psd`, `ci_ordered`, `coverage_in_unit`).
- [ ] #11 generic AR/ARMA time-series simulators.

## Design (breaking → v2.0)
- [ ] #12 unify seam vocabulary → static `@runtime_checkable` Protocols + sklearn base + mixins.
- [ ] #13 frozen result objects (`frozen`+`slots`+`__post_init__`+`to_dict` + versioned JSON schema).
- [ ] #14 capabilities-as-tags + executable conformance suite (`check_temporal_splitter`/`check_temporal_estimator`). **Conformance suite done (pilot)** — `src/temporalcv/conformance.py` (positive + negative tests); capabilities-as-tags still pending.
- [ ] #15 backend-agnostic contract (`ArrayLike` sigs + lazy splitter seam; reserve `xp`/narwhals).
- [ ] #16 governance: layout/public-contract ADR + public-API stability test; fix `pyproject` URLs (→ canonical brandon-behring).

## Follow-up
- [ ] #17 eval-toolkit ↔ temporalcv purged-splitter overlap (keep separate per hub `universal-vs-unique.md`).
- [ ] (to file, surfaced by pilot) port `TimeSeriesCrossValidator` (expanding, test-from-end, gap/purge/
  sliding) into temporalcv as a universal splitter — the primary dml_ts DML cross-fitter.
- [ ] (to file, surfaced by pilot) port `PurgedGroupTimeSeriesCV` (embargo/purge) into temporalcv, or
  reconcile with existing `cv_financial.PurgedKFold`/`PurgedWalkForward` (check for genuine overlap).

## Pilot — ✅ complete
The **CrossFitter seam** (the #8 / #12 slice) was the vertical pilot validating the approach
end-to-end before the rest of v2.0. Delivered on `feat/v2-seams-pilot`:
- `src/temporalcv/protocols.py` — `Splitter`/`CrossFitter` static Protocols (wired into `__init__`).
- `src/temporalcv/cv.py::cross_fit_residualize` — dual-variable OOF residualization, typed `cv: Splitter`.
- `src/temporalcv/conformance.py` — `check_temporal_splitter`/`check_temporal_estimator` (the contract).
- `tests/test_pilot_crossfitter_seam.py` (+ `tests/property/…`) — 21 tests: unit, property (hypothesis),
  conformance positive **and** negative, and a dependency-free **dml_ts golden-parity** test.

**Validated finding (drives A1/A2 scope):** a throwaway spike against the *live* dml_ts consumer
(`_cross_fit_nuisance_time_series`, `dml_ts/dml/temporal_plr_dml.py`) showed (a) dml_ts's existing
splitters (`TimeSeriesCrossValidator`, `BlockedTimeSeriesCV`) **already satisfy** the `Splitter`
Protocol, and (b) `cross_fit_residualize` reproduces the consumer's OOF residuals **to 1e-10** on all
of them. The cross-fit **mechanism** is therefore fully upstreamed; the only remaining universal work
is porting the **splitters** themselves — `TimeSeriesCrossValidator` (expanding, test-from-end,
gap/purge/sliding) and `PurgedGroupTimeSeriesCV` need their own issues alongside #7's
`BlockedTimeSeriesCV`. In Track B, dml_ts's `_cross_fit_nuisance_time_series` collapses to a one-line
call on this seam.
