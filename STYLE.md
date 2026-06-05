# temporalcv — Style & Design

Operational **design** contract. Code style (ruff, mypy strict, NumPy docstrings, conventional
commits) lives in [CONTRIBUTING.md](CONTRIBUTING.md). This file references the canonical hub patterns
rather than restating them.

## Hub patterns (source of truth — link, don't copy)
- `~/Claude/lever_of_archimedes/patterns/library-design-playbook.md` — seams, result objects, typing,
  evolution (the durability playbook).
- `~/Claude/lever_of_archimedes/patterns/universal-vs-unique.md` — temporalcv is the **universal**
  side: domain-general time-series machinery only, never a consumer's estimand/interpretation.

## temporalcv design contract (v2.0)
1. **Seams = static Protocols + ecosystem base + tags.** Public extension points (`Splitter`,
   `CrossFitter`, `ForecastAdapter`, `BootstrapStrategy`, gate `GateFn`) are `@runtime_checkable`
   Protocols for typed consumers — a *static aid only* (presence-check; never a validator or hot
   path). Splitters also subclass sklearn `BaseCrossValidator` for `cross_val_score`/`Pipeline`
   interop. Shared implementation lives in mixins/ABCs we own; capabilities are **tags (data)**,
   not subclass levels.
2. **Results = frozen value objects.** `@dataclass(frozen=True, slots=True)` + `__post_init__` +
   explicit `to_dict()` + a versioned JSON schema: each result declares
   `SCHEMA_VERSION: ClassVar[int]`, surfaced in `to_dict()` as `schema_version` (arrays → lists, dates →
   ISO strings, so `to_dict()` is `json.dumps`-able). A point estimate is the degenerate case of a
   distributional/interval result. (Caveat: on CPython `frozen+slots` raises `TypeError`, not
   `FrozenInstanceError`, when assigning an *undeclared* attribute — declared-field immutability is
   unaffected and is the guarantee we rely on.)
3. **Contract is backend-agnostic.** Public params typed `ArrayLike` (not `np.ndarray`); an internal
   `xp` array-namespace seam and a narwhals dataframe boundary are reserved. numpy/batch impls run
   under the hood. The splitter seam is a **lazy iterator**; `get_n_splits()` may return `None`.
4. **Conformance suite is the contract.** `check_temporal_splitter` / `check_temporal_estimator`
   enforce seams behaviorally; exported for consumers and run in CI.
5. **Evolution.** Small frozen core Protocols + additive sub-Protocols; a named frozen Tier-2 set
   gated by a public-API test; ~2-cycle user / ~1-cycle dev deprecation. Stdlib exceptions only;
   `__all__` per module; domain suffixes; **language-neutral** seam names (planned TemporalCV.jl).

## Naming canon
`X, y, time_index, train_idx, test_idx, n_splits, gap, horizon, rng` (extend as needed; deviations
need PR justification). Suffixes: `*CV`/`*Splitter`, `*Result`, `*Gate`/`GateFn`, `*Adapter`,
`*Strategy`.

## Consumers
`dml_ts` (causal inference) is the live second-consumer pressure test — if a real consumer can't sit
on a seam, the seam is wrong. `eval-toolkit` is a sibling in a *different* domain
(classification-evaluation); shared-named concepts (e.g. a purged splitter) stay separate unless
genuinely identical (see hub `universal-vs-unique.md`).
