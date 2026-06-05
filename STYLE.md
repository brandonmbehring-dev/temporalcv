# temporalcv — Style & Design

Operational **design** contract. Code style (ruff, mypy strict, NumPy docstrings, conventional
commits) lives in [CONTRIBUTING.md](CONTRIBUTING.md). This file references the canonical hub patterns
rather than restating them.

## Hub patterns (source of truth — link, don't copy)
- `~/Claude/lever_of_archimedes/patterns/library-design-playbook.md` — seams, result objects, typing,
  evolution (the durability playbook).
- `~/Claude/lever_of_archimedes/patterns/universal-vs-unique.md` — temporalcv is the **universal**
  side: domain-general time-series machinery only, never a consumer's estimand/interpretation.

## temporalcv design contract (v2.0 target)

This is the **target** contract for the v2.0 modernization (the package is `1.0.0` until the A5
release). Status markers: **✅ in place** on the v2 seam/pilot surface · **◷ planned** (tracked in
[`docs/plans/v2_roadmap.md`](docs/plans/v2_roadmap.md), issue noted). The roadmap is the source of truth
for what is built; items below say so explicitly so this contract never overclaims.

1. **Seams = static Protocols + ecosystem base + tags.** Public extension points (`Splitter`,
   `CrossFitter`, `SupportsFitPredict` ✅; `ForecastAdapter`, `BootstrapStrategy`, gate `GateFunction`)
   are `@runtime_checkable` Protocols for typed consumers — a *static aid only* (presence-check; never a
   validator or hot path). Splitters also subclass sklearn `BaseCrossValidator` for
   `cross_val_score`/`Pipeline` interop. Shared implementation lives in mixins/ABCs we own.
   Capabilities will be exposed as **tags (data)**, not subclass levels — **◷ planned (#14)**, not yet
   implemented. (Unifying the legacy `BootstrapStrategy`/`ForecastAdapter` ABCs to this pattern is
   **◷ planned (#12)**.)
2. **Results = frozen value objects.** `@dataclass(frozen=True, slots=True)` + `__post_init__` +
   explicit `to_dict()` + a version-stamped, `json.dumps`-able dict: each result declares
   `SCHEMA_VERSION: ClassVar[int]`, surfaced in `to_dict()` as `schema_version` (arrays → lists, dates →
   ISO strings). **✅ on the cv.py result objects** (`SplitInfo`/`SplitResult`/`WalkForwardResults`/
   `NestedCVResult`); other modules' results (`GateResult`, conformal, stats) migrate under **◷ #13**.
   "Point estimate = degenerate case of a distributional/interval result" is design intent, not yet a
   property of these objects. (Caveat: on CPython `frozen+slots` raises `TypeError`, not
   `FrozenInstanceError`, when assigning an *undeclared* attribute — declared-field immutability is
   unaffected and is the guarantee we rely on.)
3. **Contract is backend-agnostic.** Public params typed `ArrayLike` (not `np.ndarray`) **✅ on the v2
   seam surface** (splitters, `cross_fit_residualize`, conformance); the exported bagging/compare APIs
   still take `np.ndarray` and migrate under the same rule (**◷ #15**). An internal `xp` array-namespace
   seam and a narwhals dataframe boundary are **reserved** (not built). The splitter seam permits a
   **lazy** `get_n_splits()` that returns `None` — the `Splitter` Protocol types it `int | None` and the
   conformance suite accepts it ✅; current concrete splitters are eager and return an `int`.
4. **Conformance suite is the contract.** `check_temporal_splitter` / `check_temporal_estimator`
   enforce seams behaviorally; exported for consumers and run in CI. **✅**
5. **Evolution.** Small frozen core Protocols + additive sub-Protocols; a named frozen Tier-2 set
   gated by a public-API stability test (**◷ planned #16**); ~2-cycle user / ~1-cycle dev deprecation.
   Stdlib exceptions only; `__all__` per module; domain suffixes; **language-neutral** seam names
   (planned TemporalCV.jl).

## Naming canon
`X, y, time_index, train_idx, test_idx, n_splits, gap, horizon, rng` (extend as needed; deviations
need PR justification — e.g. `cross_fit_residualize(…, A, B, …)` uses `A`/`B` for its two symmetric
targets, neither being "y"). Suffixes: `*CV`/`*Splitter`, `*Result`, `*Gate`/`GateFunction`, `*Adapter`,
`*Strategy`.

## Consumers
`dml_ts` (causal inference) is the live second-consumer pressure test — if a real consumer can't sit
on a seam, the seam is wrong. `eval-toolkit` is a sibling in a *different* domain
(classification-evaluation); shared-named concepts (e.g. a purged splitter) stay separate unless
genuinely identical (see hub `universal-vs-unique.md`).
