# temporalcv v2.0 roadmap

Tracks the v2.0 **universal time-series toolkit** modernization. Breaking changes are acceptable
because `dml_ts` is the sole consumer, migrated in lockstep — see
`docs/adr/0001-v2-seams-and-layout.md`, `STYLE.md`, and hub
`patterns/universal-vs-unique.md` / `patterns/library-design-playbook.md`. Each item is a filed
GitHub issue.

## Capability
- [x] #7  `BlockedTimeSeriesCV` (whole-block-preserving CV) — **ported (A2)** into
  `cv.py`, now **fail-loud** (raises on an under-provisioned config instead of silently
  dropping a fold); golden-parity vs dml_ts on skip-free configs.
- [x] #8  dual-model `cross_fit_residualize(model_a, model_b, X, A, B, cv) -> (A_resid, B_resid)` with fold-0 NaN-mask. **Done (pilot)** — `src/temporalcv/cv.py`, typed `cv: Splitter`, identical shared NaN mask; exported top-level.
- [ ] #9  standalone HAC residual covariance (Bartlett/Parzen/QS + optimal bandwidth + Newey-West), **matrix-accepting** (panel-ready).
- [ ] #10 numeric/stat output validators (`finite_se`, `psd`, `ci_ordered`, `coverage_in_unit`).
- [ ] #11 generic AR/ARMA time-series simulators.

## Design (breaking → v2.0)
- [x] #12 unify seam vocabulary → static `@runtime_checkable` Protocols + sklearn base + mixins.
  **Done (A1, `feat/v2-seam-vocab`):** added `SupportsBootstrap` / `SupportsForecast` accept-seam
  Protocols (peers of `SupportsFitPredict`); `BootstrapStrategy` / `ForecastAdapter` **kept** as
  owned shared-impl ABC bases (the ADR's "mixins/ABCs we own"); consumers retyped to the Protocols;
  `check_bootstrap_strategy` / `check_forecast_adapter` conformance added. The `cv_financial`
  `BaseCrossValidator` realign was reconciled out of scope → deferred to **#25**.
- [x] #13 frozen result objects (`frozen`+`slots`+`__post_init__`+`to_dict` + versioned JSON schema).
  **Done (PR #20).**
- [x] #14 capabilities-as-tags + executable conformance suite (`check_temporal_splitter`/`check_temporal_estimator`).
  Conformance suite done (pilot); **tags done (A1):** a frozen `TemporalTags` descriptor
  (`forward_only`/`deterministic`/`produces_oof`/`requires_groups`; **no** `SCHEMA_VERSION`) exposed
  via `temporal_tags()` on the 4 forward-only `cv.py` splitters; `check_temporal_splitter`
  cross-validates declared tags against observed behavior. (`NestedWalkForwardCV` excluded — a
  tuning meta-estimator, not a splitter.)
- [ ] #15 backend-agnostic contract (`ArrayLike` sigs + lazy splitter seam; reserve `xp`/narwhals).
- [ ] #16 governance: layout/public-contract ADR + public-API stability test; fix `pyproject` URLs (→ canonical brandon-behring).

## Follow-up
- [ ] #17 eval-toolkit ↔ temporalcv purged-splitter overlap (keep separate per hub `universal-vs-unique.md`).
- [x] #22 port `TimeSeriesCrossValidator` (expanding, test-from-end, gap/purge/sliding) into
  temporalcv — the primary dml_ts DML cross-fitter. **Ported (A2)** into `cv.py`;
  `get_fold_info`/`CVFold` reconciled to `get_split_info() -> list[SplitInfo]`; golden-parity
  vs dml_ts across an (n_splits × gap × purge × expanding × test_size) grid.
- [x] #23 reconcile `PurgedGroupTimeSeriesCV` — **resolved: do not port.** It is a
  *bidirectional* purged K-fold (trains both sides of the test block → violates the forward-only
  `check_temporal_splitter` no-lookahead invariant) and redundant in *mechanism* with the richer
  `cv_financial.PurgedKFold` (+ `CombinatorialPurgedCV`); the forward-only purged analog is
  `PurgedWalkForward`. **#23 stays OPEN** as the Track-B tracker: migrate dml_ts
  `cv_strategy="purged_cv"` → `PurgedWalkForward` with an explicit before/after estimate
  re-validation (consumer-owned, not a silent swap).

## A2 — ✅ complete (`feat/v2-port-splitters`)
Ported the two **forward-only** splitters into `temporalcv/cv.py` and reconciled the
bidirectional one:
- `TimeSeriesCrossValidator` (#22) + `BlockedTimeSeriesCV` (#7), top-level exported, each
  passing `check_temporal_splitter` and a **dependency-free dml_ts golden-parity** suite
  (`tests/test_cv_splitters_ported.py`) — verbatim-reference parity confirmed bit-exact against
  the *live* dml_ts package (48 TSCV + 18 Blocked configs, 0 mismatches) at dev time.
- Reused the existing `SplitInfo` (no new `SCHEMA_VERSION` result object; `CVFold` dropped).
- `BlockedTimeSeriesCV` hardened to fail loud on degenerate configs.
- `PurgedGroupTimeSeriesCV` reconciled, not ported (#23 → Track B). `create_time_series_cv`
  not ported (string-factory; temporalcv favors explicit classes).

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
