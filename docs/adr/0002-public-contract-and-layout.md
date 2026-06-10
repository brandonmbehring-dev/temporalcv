# ADR 0002 — public contract & module layout

**Status:** Accepted (2026-06-05)
**Context:** v2.0 froze its seams (#12), result objects (#13), capability tags (#14), and the
`ArrayLike` input contract (#15). The remaining governance gap is to *document and enforce* the
stable public surface so that `dml_ts` (the sole consumer) and the Track-B migration can build
against a fixed contract, and so that accidental API breakage fails CI rather than reaching a
release. This ADR records the public-contract and module-layout decisions; ADR 0001 records the
seam strategy.

**Implementation status (2026-06-05):** in place — `tests/test_public_api.py` snapshots the
top-level `__all__` (175 names at adoption; the test snapshot is the canonical count) and guards
no-dangling-export / no-private-leak; the result-object
registry test guards versioned result objects; the conformance suite guards the seams. `pyproject`
URLs already point at canonical `brandon-behring`.

## Decision
1. **The stable v2.0 surface is the top-level `temporalcv` namespace via `__all__`** (175 names
   at adoption; additions land via deliberate snapshot updates in `tests/test_public_api.py`,
   which is the canonical count).
   Subpackages (`compare/`, `bagging/`, `metrics/`, `viz/`, `diagnostics/`, `inference/`,
   `validators/`, `benchmarks/`) are **implementation grouping**: subpackage import paths
   (`from temporalcv.bagging.base import X`) are **unstable and unsupported** — they may move
   between versions, and only the top-level surface carries the v2.0 stability guarantee (restates
   ADR 0001 §4). *Scope note:* the repo's own tests and examples currently import some subpaths for
   convenience, and `tests/test_public_api.py` enforces the **contents** of the top-level surface
   (drift / dangling export / private leak), **not** a subpath-import ban; migrating examples to
   top-level imports and adding an import-surface guard is future work, not an enforced invariant
   today.
2. **Enforcement = `tests/test_public_api.py`:** a frozen snapshot of `sorted(__all__)` (any
   addition/removal/rename fails loud and must update the snapshot deliberately), plus a
   no-dangling-export guard (every name resolves on the package) and a no-private-leak guard (no
   `_`-prefixed name except `__version__`). A public-contract change is therefore a reviewable diff.
3. **Module layout (canonical):**
   - `cv.py` — the forward-only splitter family (`WalkForwardCV`, `TimeSeriesCrossValidator`,
     `BlockedTimeSeriesCV`, `CrossFitCV`, `NestedWalkForwardCV`) + `cross_fit_residualize`.
   - `cv_financial.py` — the de Prado **purged** family (`PurgedKFold`, `CombinatorialPurgedCV`,
     `PurgedWalkForward`). Separate from `cv.py` because purged K-folds are bidirectional.
   - `protocols.py` — typed `@runtime_checkable` accept-seams (`Splitter`, `CrossFitter`,
     `SupportsFitPredict`, `SupportsBootstrap`, `SupportsForecast`); `conformance.py` — the
     executable `check_*` contract; `tags.py` — the `TemporalTags` capability descriptor.
   - `_typing.py` — the reserved `ArrayLike` / future-backend (`xp`/narwhals) seam.
   - Result objects live beside their producers, all `frozen=True, slots=True` +
     `SCHEMA_VERSION` + `to_dict()`, registry-enforced (`tests/test_result_objects.py`).
4. **Type contract:** public **input** parameters are `ArrayLike`; **return** types and stored
   result-object fields are concrete `np.ndarray`. The internal strategy/adapter seams
   (`generate_samples`/`transform_for_predict`, adapter `fit_predict`) stay `np.ndarray` by design
   (the bagger/runner normalize before delegating).
5. **Packaging metadata:** `[project.urls]` point at canonical
   `https://github.com/brandon-behring/temporalcv` (Homepage/Repository/Documentation/Issues).

## Consequences
- (+) Consumers have one import surface; the layout and contract are documented and machine-checked;
  accidental public-API breakage fails CI before release.
- (+) Track-B (`dml_ts`) migrates against a frozen, enforced contract, behind the golden-parity gate.
- (−) Every *intentional* public-API change requires a deliberate snapshot update in
  `test_public_api.py` — small friction, by design (the point is that it cannot happen silently).
- Enforcement: `test_public_api.py` (public surface) + `test_result_objects.py` (versioned result
  objects) + the conformance suite (seams) all run in CI.
