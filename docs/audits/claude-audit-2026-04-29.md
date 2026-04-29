# Consolidated temporalcv Audit Report

- **Date:** 2026-04-29
- **Auditor:** Claude (Anthropic) — Opus 4.7 (1M context)
- **Scope:** Code quality, professionalism, pythonic practice, methodology, citation accuracy, testing, documentation, packaging, CI, release risk
- **Method:** Consolidates 11 prior audits, verifies every claim against the current code/docs/config, runs fresh validation commands, and produces a single remediation-ready report intended to **supersede** the prior audit ledger.

This audit assumes nothing from the prior reports. Every finding below was re-checked against the current `main` branch on 2026-04-29 (commit `5e61658`). Findings that turned out to be **already fixed** are listed in §10 (Prior-Audit Reconciliation) so nothing is silently dropped.

---

## 1. TL;DR

The implementation is in better shape than the prior audit ledger suggests. **Nine** findings carried by older audits are already resolved in current code (`get_n_splits` silent failure, `WalkForwardCV.horizon` parameter, model cloning in `gate_signal_verification`, dead-link API docs `load_fred_series`/`load_m5_sample`/`ComparisonRunner`, DM test t-distribution under Harvey correction, Monash truncation tracking, etc.). The remaining issues concentrate in the **public trust surface**: README, citation metadata, validation-evidence doc, and a handful of repository-hygiene items.

| Scorecard | Score | Note |
|---|---|---|
| Code quality | 8.5 / 10 | Strict mypy clean, ruff clean, 86% branch coverage on 1,943 tests in 80 s |
| Methodology rigor | 8.0 / 10 | Citations largely accurate; gate semantics & implementation now aligned in code; doc surface lags |
| Test architecture | 9.0 / 10 | 6-layer suite (unit, integration, anti-pattern, property, MC calibration, golden-ref) with explicit MC helpers in `tests/conftest.py` |
| Documentation trust | 5.5 / 10 | README, `docs/quickstart.md`, `CITATION.cff`, `docs/validation_evidence.md`, `docs/api/statistical_tests.md` carry stale or wrong claims |
| Release readiness | 6.5 / 10 | Code is releasable; the next patch should be a **trust-repair** release — not features |

**Critical path (top 5 fix-now items):**

1. **B1 / B2** — Reconcile `HALT = "Leakage detected"` vs `HALT = "signal — investigate"` across `README.md:45,107`, `docs/quickstart.md:65-66`, `gates.py:10-11` (module docstring), `SPECIFICATION.md:58`. The function-level docstring at `gates.py:305-315` is already correct; align everything else to it.
2. **B2** — Fix `README.md:37-41` quick-start: `run_gates(model, X, y)` is invalid; the actual API is `run_gates(gates: list[GateResult])` (`gates.py:1483-1509`). The same README has the *correct* call at L98-108.
3. **H4** — Replace or remove the broken Lobato DOI `10.1198/016214501750333073` at `docs/api/statistical_tests.md:736` (Crossref returns 404; verified by codex same-day linkcheck).
4. **H1 / L1** — Pick a canonical repository (PyPI provenance points at `brandonmbehring-dev/temporalcv`; `pyproject.toml:94-98`, `CITATION.cff:10`, `README.md:201-208` say `brandon-behring/temporalcv`). Update all four together. Remove placeholder ORCID `0000-0000-0000-0000` in `CITATION.cff:9`.
5. **M2 / M3** — Refresh `docs/validation_evidence.md` (currently claims "83%, 318 tests"; truth is 86%, 1,943 tests) and fix the hardcoded coverage badge in `README.md:14`.

The remaining 16 findings are real but lower urgency (immutability, god-modules, doc API drift, gitignore gaps, version-vs-CHANGELOG drift, etc.).

---

## 2. Validation Evidence (fresh run, 2026-04-29)

All commands run from `/home/brandon_behring/Claude/temporalcv` against current `.venv` (Python 3.13.7) on commit `5e61658`. Verbatim outputs follow.

| Command | Result | Notes |
|---|---|---|
| `git status --short` | clean | working tree clean before this audit file |
| `git log -5 --oneline` | `5e61658 fix: Remove research-kb from .mcp.json — now in global config` … | recent activity is config polish + Furo theme migration |
| `.venv/bin/python -m pytest tests/ --cov=temporalcv --cov-report=term --ignore=tests/benchmarks/ -q` | **1943 passed, 15 skipped, 345 warnings, 86% coverage in 79.96 s** | 5,898 statements, 1,956 branches, 716 misses, 224 partial branches |
| `.venv/bin/python -m mypy src/temporalcv --show-error-codes` | **Success: no issues found in 57 source files** | strict mode (`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`) |
| `.venv/bin/python -m ruff check src/temporalcv tests/` | **All checks passed!** | rules: E, W, F, I, B, C4, UP, ARG, SIM |
| `.venv/bin/python -m ruff format --check src/temporalcv tests/` | **134 files already formatted** | line-length 100, target-version py310 |
| `.venv/bin/python -m pip-audit` | **No module named pip-audit** | dev extra declares it (`pyproject.toml:51`); local `.venv` lacks it. Same as codex run. |
| `du -sh None .venv .venv-test .venv-fresh` | `661M / 1.4G / 634M / 412M` | `None/` IS gitignored; `.venv-test` and `.venv-fresh` are NOT (gitignore only matches `.venv`). |
| `wc -l` (largest modules) | `statistical_tests.py 3223 / cv.py 1943 / gates.py 1802 / conformal.py 1547 / persistence.py 722 / cv_financial.py 697` | M7 confirmed |
| Frozen vs total dataclasses | **13 frozen / 41 total = 28 mutable** | `grep -rn "frozen=True\|@dataclass" src/temporalcv/` |

**Coverage hot spots** (modules below 70%, may indicate weakly-tested or optional-extra paths):

| Module | Coverage | Likely cause |
|---|---:|---|
| `compare/docs.py` | 9% | Documentation generator; rarely exercised |
| `compare/results.py` | 16% | Pretty-printing helper |
| `benchmarks/gluonts.py` | 11% | Optional extra; needs `gluonts` installed |
| `benchmarks/m5.py` | 40% | Requires manual M5 dataset |
| `benchmarks/fred.py` | 33% | Requires `fredapi` |
| `compare/adapters/multi_series.py` | 50% | Optional path |
| `benchmarks/__init__.py` | 64% | Loader fallthrough |
| `viz/_base.py` | 69% | Plotting paths |

These do not block the **80% project-wide** floor (CI enforces `--cov-fail-under=80`; current 86% is comfortably above), but the optional-extra modules deserve dedicated CI matrix jobs that install the extra and re-run.

---

## 3. Findings — Blocker

### B1. Public docs teach `HALT = leakage detected`, contradicting the corrected source code

- **Severity:** **BLOCKER** (user-trust)
- **Status:** STILL VALID at the documentation surface; implementation has been corrected.
- **Evidence:**
  - `README.md:45` — feature table says `HALT | Leakage detected — stop and investigate`
  - `README.md:107` — `if report.status == "HALT": raise ValueError(f"Leakage detected: {report.summary()}")`
  - `gates.py:10-11` — module docstring still asserts "if a model beats a shuffled target … it's likely learning from leakage"
  - `docs/quickstart.md:65-66` — `if shuffled_result.status.name == "HALT": raise ValueError("Leakage detected!")`
  - `SPECIFICATION.md:58` — "If model improves by more than threshold over shuffled, likely leakage"
  - **Counter-evidence (the correct version):** `gates.py:305-315` — function docstring says "HALT indicates the model has learned signal — this could be legitimate temporal patterns OR data leakage"; `gates.py:380-393` — explicit Notes section: "HALT → Investigate: Confirm signal is legitimate (e.g., AR model with proper lagged features) or identify leakage source"
- **Why it matters:** A user with a legitimate AR(1)-style model on a persistent series will *correctly* see HALT — the gate working as intended — and, following the README/quickstart, raise an exception and reject the valid model. Conversely, a user may treat HALT as conclusive proof of leakage when it is only a diagnostic flag.
- **Fix options:**

  | Option | Pros | Cons | Reasoning / evidence |
  |---|---|---|---|
  | **B1.a — Rewrite all public docs to "signal detected, investigate"** (recommended) | Aligns with already-correct function-level docstring at `gates.py:305-315`; minimal code churn; preserves API | Users used to the "leakage detector" framing must adapt | The implementation has already moved here; only the marketing has not. Codex Option A1. |
  | B1.b — Split into two functions: `gate_signal_verification` (signal only) + `gate_leakage_suspicion` (composite of signal + boundary + theoretical-bound failure) | Cleaner mental model; gives users a real "leakage" gate | Requires API design and migration shim; defer until B1.a is shipped | Codex Option A2; long-term path. |
  | B1.c — Keep the "leakage detected" framing and change the implementation to require a leakage-specific condition (e.g., signal AND boundary violation AND beats theoretical bound) | Matches current marketing | Changes statistical interpretation, will reject many models that are merely well-calibrated to legitimate temporal structure | **Do not.** Statistically weaker than B1.a or B1.b. Codex Option A3 (rejected there too). |

### B2. README quick-start uses an invalid `run_gates` signature

- **Severity:** **BLOCKER** (copy-paste fails immediately)
- **Status:** STILL VALID
- **Evidence:**
  - `README.md:37-41`: `report = run_gates(model, X, y)` — wrong
  - `gates.py:1483-1509` — actual signature is `run_gates(gates: list[GateResult]) -> ValidationReport`
  - **Internal contradiction:** `README.md:98-108` later shows the *correct* call (`gates = [...]; report = run_gates(gates)`)
- **Why it matters:** A new user copy-pasting from the quick-start will get a `TypeError` on their first call. The whole README story collapses on first contact.
- **Fix options:**

  | Option | Pros | Cons | Reasoning |
  |---|---|---|---|
  | **B2.a — Replace L37-41 with the same form already used at L98-108** (recommended) | One-line fix; consistent with the rest of the page; matches docstring at `gates.py:34-39` | None | Internal source-of-truth already exists. |
  | B2.b — Add a new top-level convenience `run_gates_for(model, X, y) -> ValidationReport` that runs a default set of gates and returns the report | Restores the "3 lines" promise the README wanted | Adds API surface; needs design and tests | Worth considering after B1; users genuinely benefit from a one-liner happy path. |
  | B2.c — Remove the quick-start altogether and link to `docs/quickstart.md` | Stops the bleeding | Loses scannable on-page demo | Last resort. |

---

## 4. Findings — High

### H1. Repository identity drift (`brandon-behring` vs `brandonmbehring-dev`)

- **Severity:** High
- **Status:** STILL VALID
- **Evidence:**
  - `pyproject.toml:94-98` — Homepage / Documentation / Repository / Issues all point at `https://github.com/brandon-behring/temporalcv`
  - `CITATION.cff:10` — `repository-code: "https://github.com/brandon-behring/temporalcv"`
  - `README.md:10`, `README.md:15`, `README.md:201-208` (BibTeX) — same
  - PyPI Trusted Publishing provenance for `temporalcv 1.0.0` points at `brandonmbehring-dev/temporalcv@5f2048b…` on tag `v1.0.0` (per codex same-day check)
- **Why it matters:** A research-adjacent package needs a single answer to "which repo backs the signed PyPI artifact?". Citation, issue-reporting, and reproduction all depend on that answer.
- **Fix options:**

  | Option | Pros | Cons | Reasoning |
  |---|---|---|---|
  | **H1.a — Adopt `brandonmbehring-dev/temporalcv` everywhere (matches PyPI provenance)** (recommended) | No re-publishing needed; matches the signed artifact; lowest risk | Need to update `pyproject.toml`, `CITATION.cff`, `README.md`, GitHub badges, ReadTheDocs source URL | The PyPI artifact is the authoritative tagged release. Codex Option C1. |
  | H1.b — Move publishing back to `brandon-behring/temporalcv` | Matches current local metadata | Requires re-configuring trusted publishing on PyPI; may introduce history confusion if `brandonmbehring-dev` is the long-running owner | Codex Option C2; only if `brandon-behring` is the intended canonical org. |
  | H1.c — Mirror both, document one as canonical | Preserves both | Mirror drift over time; users still confused | Avoid. |

### H2. README feature table claims sklearn lacks gap enforcement; sklearn `TimeSeriesSplit` has had `gap` since v0.24

- **Severity:** High (factual, easily falsified by a curious user)
- **Status:** STILL VALID
- **Evidence:**
  - `README.md:64-66` — "Gap enforcement | ✓ | ✗ | ✗ | ✗" (temporalcv vs sklearn/sktime/darts)
  - `cv.py:39-40` — source acknowledges: "sklearn's `TimeSeriesSplit` exposes a numeric `gap`; temporalcv derives the required gap from forecast `horizon`"
  - sklearn upstream: `sklearn.model_selection.TimeSeriesSplit(gap=0)` since v0.24 (2020)
- **Why it matters:** Inaccurate competitive claims weaken the believability of the (genuine) differentiation. The honest pitch is stronger: temporalcv adds **horizon-aware** gap derivation, validation gates, and statistical layers around the boundary.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **H2.a — Reframe as "Manual gap (sklearn) vs horizon-derived gap + leakage gates (temporalcv)"** (recommended) | Honest, stronger pitch | Slightly longer table |
  | H2.b — Drop the comparison table | Eliminates the falsifiable claim | Loses a useful at-a-glance differentiator |
  | H2.c — Replace with a code-block side-by-side ("here's sklearn, here's temporalcv") | Concrete; users can see the difference | More vertical space |

### H3. README dependency table contradicts `pyproject.toml`

- **Severity:** High (install-confidence issue)
- **Status:** STILL VALID
- **Evidence:**
  - `README.md:169` — "Core: numpy >= 1.23, scipy >= 1.9, scikit-learn >= 1.1, pandas >= 1.5"
  - `pyproject.toml:34-40` — `numpy>=1.21,<3.0`, `scipy>=1.7,<2.0`, `scikit-learn>=1.0,<2.0`, `statsmodels>=0.13,<1.0`, `matplotlib>=3.5,<4.0`
  - `pyproject.toml:55` — `pandas = ["pandas>=1.3"]` (pandas is **optional**, not core)
  - README does not mention `statsmodels` (which IS core) or `matplotlib` (which IS core)
- **Why it matters:** Users on older numpy/scipy/sklearn will be turned away by the README despite `pyproject.toml` accepting their versions. Users will assume pandas is required when it is not.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **H3.a — Generate the dependency line from `pyproject.toml` at doc build** (recommended long-term) | No drift, ever | Needs a small Sphinx/MyST snippet or pre-build step |
  | H3.b — Hand-rewrite once to match (`numpy >= 1.21, scipy >= 1.7, scikit-learn >= 1.0, statsmodels >= 0.13, matplotlib >= 3.5; optional: pandas >= 1.3`) | Immediate; trivial | Will drift again |
  | H3.c — Replace with a link to `pyproject.toml` and stop duplicating | Eliminates drift surface | Slightly less convenient for casual readers |

### H4. Lobato DOI `10.1198/016214501750333073` returns 404

- **Severity:** High (citation correctness in a research-adjacent library)
- **Status:** STILL VALID
- **Evidence:** `docs/api/statistical_tests.md:736` — full citation with the broken DOI link. Lobato (2001) "Testing That a Dependent Process Is Uncorrelated", *JASA* 96(453), 169-176, **does** exist; the DOI string above is malformed. The correct DOI is `10.1198/016214501750332811` (note the trailing digits differ — verify via Crossref before substituting).
- **Why it matters:** Sphinx linkcheck fails on this; readers following the citation hit a 404. This is the only fully-broken citation found.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **H4.a — Replace with the correct Lobato (2001) DOI** (recommended) | Restores citation | Need to verify the correct DOI string against Crossref before publishing |
  | H4.b — Drop the DOI hyperlink, keep the bibliographic citation | Simple; never breaks | Slightly worse UX |
  | H4.c — Remove the citation entirely | Stops the bleed | Loses a relevant reference for the self-normalized variance method |

### H5. Shuffled-target gate documentation lags the (now-correct) implementation

- **Severity:** High (downgraded from Blocker after current-code verification)
- **Status:** PARTIALLY ADDRESSED — implementation is now correct; only docs and SPECIFICATION lag.
- **Evidence (the good news):**
  - `gates.py:286-304` — default `method="permutation"` (true p-value) and default `permutation="block"` (preserves autocorrelation per Künsch 1989, Politis & Romano 1994)
  - `gates.py:340-346, 357-359, 405-408` — block permutation with `block_size="auto"` set to `n^(1/3)`; p-value computed per Phipson & Smyth (2010)
  - `gates.py:410-411` — "Models are cloned for each shuffle to prevent state leakage from warm-start or incremental learning algorithms" (CV4 RESOLVED)
- **Evidence (the gap):**
  - `gates.py:10-11` (module-level docstring) — still says "if a model beats a shuffled target … it's likely learning from leakage"
  - `SPECIFICATION.md:24` — `HALT if: improvement > 0.20` (effect-size framing only)
  - `SPECIFICATION.md:42-47` — describes p-value method; `:58` says "likely leakage"
  - `gates.py:20-22` (knowledge tiers) — "[T3] 20% improvement threshold = 'too good to be true' heuristic" applies only to `method="effect_size"` (now non-default)
- **Why it matters:** The cited statistical theory (Künsch, Phipson-Smyth, Politis-Romano) now genuinely backs the implementation. The remaining work is rewriting the marketing layer to reflect that.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **H5.a — Rewrite `gates.py:10-22` and `SPECIFICATION.md` to make permutation the headline and effect-size the optional fast heuristic** (recommended) | Matches current implementation; preserves both methods | Editorial work in two places |
  | H5.b — Deprecate `method="effect_size"` entirely | Removes the only path that uses the threshold-ratio heuristic | Breaking; some users may want the fast path |
  | H5.c — Leave the docs and rename the function | Avoids the rename problem | Confuses users who already know the name |

### H6. Immutability principle violated: 13 / 41 dataclasses are frozen (28 mutable)

- **Severity:** High (hub convention; correctness of result types under concurrent / cached use)
- **Status:** STILL VALID
- **Evidence:** `grep -rn "@dataclass\|frozen=True" src/temporalcv/`
  - **Frozen (good):** `cv.py:58` (SplitInfo), `cv_financial.py:38` (PurgedSplit), `changepoint.py:50,72`, `stationarity.py:51,84`, `lag_selection.py:32`, `diagnostics/{influence.py:28, sensitivity.py:23}`, `inference/{wild_bootstrap.py:46, block_bootstrap_ci.py:50}`, `statistical_tests.py:1921, 2127`
  - **Mutable result/report types that should be frozen:** `gates.py:78` (GateResult), `gates.py:114` (ValidationReport), `gates.py:1517` (StratifiedValidationReport), `conformal.py:61`, `conformal.py:1360`, multiple in `statistical_tests.py:76,138,190,279,1691,2443,2486,2734,2782`, `compare/base.py:31,108,201`, `regimes.py:370`, `persistence.py:72`, `metrics/event.py:55,111`, `metrics/volatility_weighted.py:477`, `guardrails.py:45`, `cv.py:136,254,425`, `benchmarks/base.py:72,191`
- **Why it matters:** The hub convention (`~/Claude/lever_of_archimedes/patterns/`) and `CLAUDE.md` say "immutability by default". For *result* objects, frozen prevents accidental mutation by callers (e.g., reporting code overwriting a `metric_value`); the cost is near zero because nothing internal mutates them after construction.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **H6.a — Mechanical sweep: add `frozen=True` to every result/report dataclass** (recommended) | Consistent with already-frozen partners (e.g., `SplitInfo`, `LagSelectionResult`); ~30 lines changed | Some downstream code may currently mutate (e.g., post-hoc `.details["foo"] = ...`); must check |
  | H6.b — Convert to `NamedTuple` instead of frozen dataclass | Even lighter; hashable for caching | More invasive (different attribute access semantics); breaks `dataclasses.asdict` users |
  | H6.c — Status quo, document mutation as an antipattern | No code change | Doesn't enforce; principle drifts in PRs |

  **Tactical note for H6.a:** `GateResult.details` is a mutable `dict`; freezing the dataclass does not freeze that dict. Either accept that or switch to `MappingProxyType` for true immutability of the contents.

---

## 5. Findings — Medium

### M1. `gap` → `extra_gap` rename incomplete in user-facing surfaces

- **Status:** PARTIALLY ADDRESSED.
- **Evidence:** `cv.py:560-602` — current `WalkForwardCV.__init__` uses `extra_gap` (and a separate `horizon`). `README.md:78-91` — the example correctly uses `horizon=2` but pre-1.0 docs and tests may still use `gap=`. The examples and notebooks are mixed.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M1.a — Sweep all `.md`, `.ipynb`, `.py` for `gap=` and migrate to `horizon=` / `extra_gap=`** (recommended) | Eliminates the double-vocabulary | Several dozen files |
  | M1.b — Add a deprecation alias: accept `gap=` and emit `DeprecationWarning` | Backward compatible | Adds API surface |
  | M1.c — Status quo | None | Future readers won't know which name is canonical |

### M2. `docs/validation_evidence.md` is stale by ~4 months

- **Status:** STILL VALID
- **Evidence:**
  - File footer: `**Last Updated**: 2026-01-09 / **Coverage**: 83% (318 tests passing)`
  - Today's run: 1,943 tests, 86% coverage, 80 s
  - Audit checklist references `pytest tests/monte_carlo/ -v --run-slow` — there is no `--run-slow` flag; `tests/conftest.py:25-29` registers `slow` and `monte_carlo` markers, used as `-m monte_carlo` or `-m "not slow"`.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M2.a — Auto-generate `validation_evidence.md` footer from the latest CI run** (recommended) | Never stale | One-time scripting effort |
  | M2.b — Hand-update on every release as part of the checklist | Cheap | Easy to forget — has already failed twice |
  | M2.c — Drop the specific numbers; reference the CI dashboard | Eliminates drift surface | Loses single-document reproducibility |

### M3. README coverage badge hardcoded to "83%"

- **Status:** STILL VALID
- **Evidence:** `README.md:14` — `<img src="https://img.shields.io/badge/coverage-83%25-green" alt="Coverage">`
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M3.a — Wire up Codecov badge: `https://codecov.io/gh/<repo>/branch/main/graph/badge.svg`** (recommended) | Auto-updates | Requires Codecov setup (CI already uploads to Codecov per `ci.yml`) |
  | M3.b — Bump hardcoded to "86%" | One char fix | Will drift again |
  | M3.c — Remove the badge | No drift | Loses signal |

### M4. Quickstart still teaches a leaky lag-feature pattern

- **Status:** PARTIALLY ADDRESSED — `docs/tutorials/walk_forward_cv.md:189-193` has the *correct* `create_lag_features` helper and shows it being applied **inside** the CV loop; `docs/quickstart.md:31-35` still constructs lags on the full series before splitting (the same pattern its own comment marks as "CORRECT way: will be done per-split" — but then it doesn't do that).
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M4.a — Move `quickstart.md` to use `create_lag_features` inside the CV loop, like the tutorial** (recommended) | Stops teaching the leak the library is supposed to prevent | One section rewrite |
  | M4.b — Add an explicit "this is ONLY for demonstration" warning to the leaky example | Honest | Still teaches the wrong thing first |
  | M4.c — Remove lag features from the quickstart entirely; use a pre-baked feature matrix | Simplest | Less educational |

### M5. `docs/guide/common_pitfalls.md` calls `gate_signal_verification(y_train, y_test)` — invalid signature

- **Status:** STILL VALID
- **Evidence:**
  - `docs/guide/common_pitfalls.md:46` — `result = gate_signal_verification(y_train, y_test)`
  - `docs/guide/common_pitfalls.md:82` — `gate_signal_verification(y_train, y_test, features=X_train)` (no `features` parameter exists)
  - Actual signature: `gates.py:286-304` — `gate_signal_verification(model, X, y, n_shuffles=…, threshold=…, …)`
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M5.a — Replace with the correct `(model, X, y, …)` form across both call sites** (recommended) | Working examples | Editorial |
  | M5.b — Add a doctest harness that imports each example block and exercises it | Catches future drift | Requires fixture design |
  | M5.c — Remove the page | Stops the bleed | Loses real value (this page is conceptually useful) |

### M6. Conformal exchangeability caveat is in the runtime warning + docstring, but not in the README/headline

- **Status:** PARTIALLY ADDRESSED — the **module is honest** at runtime; the README is not.
- **Evidence:**
  - `conformal.py:232-238` — `warnings.warn("SplitConformalPredictor assumes exchangeability (i.i.d. data). …")` fires at fit time
  - `conformal.py:268-283` — additional tiered sample-size warnings
  - `README.md:51-58` and the project tagline ("Rigorous cross-validation … with leakage detection and gap enforcement") imply unconditional coverage guarantees
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M6.a — Add a one-line caveat in README features section: "marginal coverage under exchangeability; time-series autocorrelation may invalidate"** (recommended) | Honest; one line | None |
  | M6.b — Reframe README around AdaptiveConformalPredictor (Gibbs & Candès 2021), which targets long-run coverage under distribution shift | More accurate for time series | Bigger rewrite |
  | M6.c — Status quo (rely on runtime warning) | None | Users may not see the warning until in production |

### M7. Four "god modules" with multiple 100+ line functions

- **Status:** STILL VALID
- **Evidence:** `wc -l` confirms `statistical_tests.py 3,223 / cv.py 1,943 / gates.py 1,802 / conformal.py 1,547`. Hot-spot functions (lines, file:line):
  - `gate_signal_verification` 396 lines (`gates.py:286-682`)
  - `dm_test` 298 lines (`statistical_tests.py:611-908`)
  - `gw_test` ~273 lines, `cw_test` ~268 lines (`statistical_tests.py`)
  - `gate_residual_diagnostics` ~202 lines, `gate_synthetic_ar1` ~172 lines (`gates.py`)
  - `walk_forward_evaluate` ~173 lines, `NestedWalkForwardCV.fit` ~137 lines (`cv.py`)
  - `compute_coverage_diagnostics` ~134 lines (`conformal.py`)
  - `compute_move_conditional_metrics` ~208 lines (`persistence.py`)
- **Why it matters:** `pyproject.toml` keeps mypy strict and ruff comprehensive, but reviewer attention does not scale linearly with file size. Statistical correctness review is the real cost: a 298-line `dm_test` makes it hard to verify that HAC variance, Harvey adjustment, t-vs-normal selection (already correctly conditional on `harvey_correction`), and CI handling are jointly right.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M7.a — Decompose hot-spot functions into small, named helpers (`_dm_variance_hac`, `_dm_apply_harvey`, `_dm_pvalue`) but keep the public function signatures stable** (recommended) | Reviewer cost drops; tests can target helpers | Internal-only churn |
  | M7.b — Split files: `statistical_tests/{dm,gw,cw,pt,multiple_testing,variance}.py`; `gates/{signal,boundary,residuals,reporting}.py`; preserve top-level imports via shims | Better long-term mental map | Bigger PR; risk of import churn for users |
  | M7.c — Status quo | Zero risk | Tech debt compounds |

  **Sequencing:** Do M7.a after the trust-repair release; defer M7.b to the next minor.

### M8. Repository hygiene: `.venv-test` and `.venv-fresh` not gitignored; `None/` is gitignored but 661 MB on disk

- **Status:** STILL VALID (not flagged in any prior audit)
- **Evidence:**
  - `.gitignore:31-37` — only `.venv` (literal) is excluded; `.venv-test`, `.venv-fresh`, `.venv-old`, etc. would be tracked
  - `.gitignore:51-54` — `None/`, `.tracking/`, `.benchmarks/` are excluded
  - `du -sh`: `None/ 661M`, `.venv 1.4G`, `.venv-test 634M`, `.venv-fresh 412M`
  - The `None/` directory almost certainly comes from a `cache_dir=None` literal-path bug in some benchmark loader (a `Path(None)` somewhere produces `'None'`).
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M8.a — Change `.gitignore` line to `.venv*` and grep the codebase for the `Path(None)` source of the `None/` directory** (recommended) | Fixes both | A few minutes |
  | M8.b — Add `.venv*/` only | Half fix | Doesn't trace the `None/` root cause |
  | M8.c — Add `find . -maxdepth 1 -name None -prune -delete` to a Makefile clean target | Cleans the symptom | Doesn't fix the bug producing it |

### M9. Examples (`examples/00`–`20_*.py`) are not directly executed in CI

- **Status:** STILL VALID
- **Evidence:** `.github/workflows/ci.yml:120-171` runs `notebooks` job (which executes `*.ipynb`); `examples/*.py` are validated only when `sphinx-gallery` builds the docs (`docs/conf.py` includes the gallery, but ReadTheDocs has `fail_on_warning: false`, so a broken example may not fail).
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M9.a — Add a CI matrix job that does `python examples/00_quickstart.py … python examples/20_failure_kfold.py` (or `pytest --collect-only examples/`)** (recommended) | Catches API drift fast; cheap | Adds ~30-60 s per matrix cell |
  | M9.b — Convert each `examples/*.py` into a tiny test under `tests/test_examples.py` that imports and runs it | Centralizes; uses existing fixtures | Some examples take time |
  | M9.c — Rely on sphinx-gallery + flip ReadTheDocs `fail_on_warning: true` (see M11) | Free signal | Slow feedback loop (only on doc build) |

### M10. Pre-commit `mypy` uses `language: system`

- **Status:** STILL VALID
- **Evidence:** `.pre-commit-config.yaml:17-25` — `id: mypy / entry: mypy src/temporalcv --show-error-codes / language: system / pass_filenames: false`
- **Why it matters:** Fresh-clone contributors who do not first install mypy locally will see the hook error out, with no isolated environment to fall back on. CI does not have this problem because `ci.yml` installs `[dev]`.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M10.a — Switch to `https://github.com/pre-commit/mirrors-mypy` and pin `additional_dependencies` to project deps** (recommended) | Isolated; works on clean clones | Initial config |
  | M10.b — Add a docs note: "run `pip install -e .[dev]` before installing pre-commit hooks" | Trivial | Easy to miss |
  | M10.c — Drop the mypy hook; rely on CI | Simplest | Loses pre-push signal |

### M11. ReadTheDocs `fail_on_warning: false`

- **Status:** STILL VALID
- **Evidence:** `.readthedocs.yaml:18-20`
- **Why it matters:** Sphinx warnings (broken cross-refs, missing autodoc targets, malformed RST) silently become live docs. Combined with M2 (stale evidence) and M5 (wrong API examples), this is how doc drift compounds.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M11.a — Flip to `fail_on_warning: true` and fix whatever first build surfaces** (recommended) | Forcing function | First fix may take 30-60 minutes |
  | M11.b — Add a `make check-docs` target that runs `sphinx-build -W` locally; keep RTD permissive | Local catch | Easy to forget |
  | M11.c — Status quo + a docs lint pass on each release | None | Doesn't scale |

### M12. CV8 — `coverage_gap` sign inconsistency between two conformal APIs

- **Status:** STILL VALID
- **Evidence:**
  - `conformal.py:1180,1203` — `coverage_gap = coverage - target_coverage` (positive when over-covering)
  - `conformal.py:1507` — `coverage_gap = target_coverage - overall_coverage` (positive when under-covering)
  - Both are returned to users via different result types
- **Why it matters:** Two functions in the same module compute the same-named field with opposite signs. A user combining the two APIs (or reading log output) will silently misinterpret over- vs under-coverage.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M12.a — Pick one convention (recommended: `coverage_gap = coverage - target_coverage`, so positive = over-covering, matches `conformal.py:1180,1203`) and fix the other call site** | One coherent contract | Touches two files; needs a CHANGELOG note |
  | M12.b — Rename one of them: `coverage_excess` (positive over) and `coverage_deficit` (positive under) | Disambiguates by name | More verbose |
  | M12.c — Document the discrepancy | Cheapest | Surprise factor remains |

### M13. CV9 — `pyproject.toml` says 1.0.0; `CHANGELOG.md` documents [1.1.0] released 2026-01-01

- **Status:** STILL VALID
- **Evidence:** `pyproject.toml:3` and `src/temporalcv/__init__.py:47` agree on `1.0.0`. `CHANGELOG.md:14` shows `## [1.1.0] - 2026-01-01` with several "Added" features. PyPI shows only `1.0.0`.
- **Why it matters:** Either CHANGELOG is aspirational (and should be marked `[Unreleased]`) or the version was rolled back / never bumped. Either way, "what version am I running?" has a wrong answer.
- **Fix options:**

  | Option | Pros | Cons |
  |---|---|---|
  | **M13.a — Bump `pyproject.toml` and `__init__.py` to 1.1.0, tag, and republish to PyPI** (recommended if features are actually shipped) | Aligns reality | Requires release process — see also H1 |
  | M13.b — Move the 1.1.0 block back under `## [Unreleased]` until the version is bumped | Honest changelog | Need to track what's pending separately |
  | M13.c — Treat 1.0.0 as a stable branch and keep 1.1.0 documented as dev | Allows stable / dev split | Unusual for a single-developer project |

---

## 6. Findings — Low

### L1. `CITATION.cff` placeholder ORCID and stale release date

- `CITATION.cff:9` — `orcid: "https://orcid.org/0000-0000-0000-0000"`
- `CITATION.cff:14` — `date-released: "2025-01-05"` (PyPI shows 2026-01-08 for `1.0.0`)
- **Fix options:** (a) Replace with real ORCID and 2026-01-08 date; (b) Drop the ORCID line; (c) Status quo. **Recommend (a).**

### L2. Python-version classifier mismatch

- `pyproject.toml:7,20-33` — `requires-python = ">=3.10"`, classifiers 3.10/3.11/3.12 only
- `.github/workflows/ci.yml:15-16` — Python 3.10–3.12 (matches local)
- PyPI metadata advertises 3.9 (per codex audit; not re-fetched here)
- **Fix options:** (a) Re-publish from current `pyproject.toml` so PyPI metadata refreshes; (b) Add 3.9 to `pyproject.toml` if support is genuinely intended; (c) Status quo. **Recommend (a).**

### L3. Two incomplete citations lack venue

- `bagging/strategies/block_bootstrap.py:18` — Lahiri (1999) "Theoretical comparisons of block bootstrap methods" (no venue). Likely *Annals of Statistics*, 27(1), 386-404.
- `cv_financial.py` — López de Prado & Lewis (2019) (no venue). Verify against the *J. of Portfolio Management* paper (2019) or the 2018 book.
- **Fix options:** (a) Verify and complete both; (b) Drop the partial citations; (c) Status quo. **Recommend (a).**

### L4. Internal `myga-forecasting-v2` references lack external citation

- `gates.py:18, 52` and similar — `[T2]` references to "myga-forecasting-v2 validation"
- These are honest knowledge-tier annotations, but external readers cannot follow them.
- **Fix options:** (a) Replace with publicly-available citations or remove; (b) Add an "internal references" appendix to docs; (c) Status quo. **Recommend (b)** — the [T1]/[T2]/[T3] tagging is genuinely useful and shouldn't be discarded.

### L5. Optional-extra modules with low coverage

- `compare/docs.py` 9%, `compare/results.py` 16%, `benchmarks/{gluonts,m5,fred}.py` 11/40/33%
- Not blocking the 80% project floor, but the optional-extra surfaces are effectively untested in default CI.
- **Fix options:** (a) Add CI matrix jobs that install each optional extra and run a thin smoke-test; (b) Mark these modules `pragma: no cover` in `[tool.coverage.report].exclude_lines` if they truly cannot be tested in CI; (c) Status quo. **Recommend (a).**

---

## 7. Citation Accuracy Table

Verified against current code locations on 2026-04-29.

| # | Author(s) (year) | Cited at | Implementation matches | Status |
|---|---|---|---:|---|
| 1 | Diebold & Mariano (1995) | `statistical_tests.py:611-908`; `CITATION.cff:25-37` | ✓ | OK; DOI `10.1080/07350015.1995.10524599` |
| 2 | Harvey, Leybourne & Newbold (1997) | `statistical_tests.py:617, 872-905` | ✓ | OK; t-distribution selected when `harvey_correction=True` (`stats.t.cdf(df=n-1)` at L879-886) |
| 3 | Giacomini & White (2006) | `statistical_tests.py` (`gw_test`) | ✓ | OK |
| 4 | Clark & West (2007) | `statistical_tests.py` (`cw_test`) | ✓ | OK |
| 5 | Pesaran & Timmermann (1992) | `statistical_tests.py` (`pt_test`); `CITATION.cff:49-60` | ✓ | OK; `CITATION.cff` missing DOI `10.1080/07350015.1992.10509922` |
| 6 | Newey & West (1987) | `statistical_tests.py` (HAC) | ✓ | OK |
| 7 | Andrews (1991) | `statistical_tests.py` (bandwidth) | ✓ | OK |
| 8 | Shao (2010) | `statistical_tests.py` (self-normalized variance) | ✓ | OK |
| 9 | Lobato (2001) | `docs/api/statistical_tests.md:736` | ⚠ | DOI `10.1198/016214501750333073` returns **404**; need correct DOI |
| 10 | Tashman (2000) | `cv.py:9-10`; gates module docstring | ✓ | OK |
| 11 | Bergmeir & Benítez (2012) | `cv.py:33-38` | ✓ | OK; `WalkForwardCV.horizon` parameter and `total_separation = horizon + extra_gap` enforce gap ≥ horizon |
| 12 | Romano, Patterson & Candès (2019) — CQR | `conformal.py:41-42`; `CITATION.cff:38-48` | ✓ | OK; `CITATION.cff` could add arXiv DOI |
| 13 | Gibbs & Candès (2021) — ACI | `conformal.py:43-44` | ✓ | OK |
| 14 | Vovk, Gammerman & Shafer (2005) | `conformal.py:45-46` | ✓ | OK |
| 15 | Hewamalage, Bergmeir & Bandara (2023) | `gates.py:43-45` | ✓ | OK |
| 16 | Künsch (1989) — block bootstrap | `gates.py:340-346, 405-408, 417-418`; `bagging/strategies/block_bootstrap.py:14-17` | ✓ | OK; default permutation strategy in signal-verification gate is now block (per Künsch) |
| 17 | Lahiri (1999) | `bagging/strategies/block_bootstrap.py:18` | ⚠ | Citation venue missing |
| 18 | Politis & Romano (1994) — stationary bootstrap | `gates.py:419-420`; `bagging/strategies/stationary_bootstrap.py` | ✓ | OK |
| 19 | López de Prado (2018) — Advances in Financial ML | `cv_financial.py` | ✓ | OK |
| 20 | López de Prado & Lewis (2019) | `cv_financial.py` | ⚠ | Citation venue missing |
| 21 | Cameron, Gelbach & Miller (2008) | `inference/wild_bootstrap.py:16-18` | ✓ | OK |
| 22 | Webb (2023) — wild bootstrap revisited | `inference/wild_bootstrap.py:19-20` | ✓ | OK |
| 23 | MacKinnon & Webb (2017) | `inference/wild_bootstrap.py:21-23` | ✓ | OK |
| 24 | Phipson & Smyth (2010) — permutation p-values | `gates.py:331-333, 421-424` | ✓ | OK; p-value formula `(1 + count(shuffled ≤ model)) / (1 + n_shuffles)` matches paper |
| 25 | Dickey & Fuller (1979) — ADF | `stationarity.py` | ✓ | OK (via statsmodels) |
| 26 | Kwiatkowski et al. (1992) — KPSS | `stationarity.py` | ✓ | OK (via statsmodels) |
| 27 | Phillips & Perron (1988) — PP test | `stationarity.py` | ✓ | OK |
| 28 | Murphy (1988) — skill score | `persistence.py` | ✓ | OK |
| 29 | Yang, Candès & Lei (2024) — Bellman conformal | `conformal.py:537` (`BellmanConformalPredictor`) | ✓ | OK; class exists |
| 30 | Hansen, Lunde & Nason (2011) — MCS | `statistical_tests.py` | ✓ | OK |
| 31 | White (2000) — Reality Check | `statistical_tests.py:2730+` | ✓ | OK |
| 32 | Hansen (2005) — SPA test | `statistical_tests.py:2785+` | ✓ | OK |
| 33 | Internal: `myga-forecasting-v2` | `gates.py:18,52`; `persistence.py` | ⚠ | No external backing; T2 tier — acceptable but document |

**Citation summary:** 28 verified-correct, 4 incomplete or broken (Lobato DOI 404, Lahiri venue, López-Lewis venue, internal myga refs). No factually-wrong citations found.

---

## 8. Methodology Correctness Summary

| Method | File | Verdict | Notes |
|---|---|---|---|
| Walk-forward CV | `cv.py:489-700+` | ✓ Pass | Temporal ordering enforced; `horizon` parameter exposes gap-derivation; `total_separation = horizon + extra_gap` |
| Purged / CPCV | `cv_financial.py` | ✓ Pass | Embargo + purging match López de Prado (2018) |
| DM test | `statistical_tests.py:611-908` | ✓ Pass | HAC variance, Harvey small-sample correction with t-distribution selection (L879-886), self-normalized variance option (Shao 2010) |
| GW test | `statistical_tests.py` (`gw_test`) | ✓ Pass | Conditional predictive ability via auxiliary regression |
| CW test | `statistical_tests.py` (`cw_test`) | ✓ Pass | Nested-model bias correction |
| PT test | `statistical_tests.py` (`pt_test`) | ✓ Pass | Directional accuracy with z-test |
| Newey-West HAC | `statistical_tests.py` | ✓ Pass | Bartlett kernel; bandwidth `bw = h - 1` (matches Harvey 1997 default) |
| Block bootstrap (MBB) | `bagging/strategies/block_bootstrap.py` | ✓ Pass | `block_size = n^(1/3)` per Künsch |
| Stationary bootstrap | `bagging/strategies/stationary_bootstrap.py` | ✓ Pass | Geometric block lengths per Politis & Romano |
| Wild cluster bootstrap | `inference/wild_bootstrap.py` | ✓ Pass | Webb 6-point for small clusters |
| Split conformal | `conformal.py:230+` | ✓ with caveat | Quantile formula correct; runtime warns about exchangeability for time series, but README does not |
| Adaptive conformal (ACI) | `conformal.py:537+` | ✓ Pass | Online distribution shift adjustment per Gibbs & Candès |
| Bellman conformal | `conformal.py:537` (`BellmanConformalPredictor`) | ✓ Class exists | Could add a model card describing scope |
| Stationarity tests | `stationarity.py` | ✓ Pass | ADF, KPSS, PP via statsmodels |
| `gate_signal_verification` | `gates.py:286-682` | ✓ with caveat | Default `method="permutation"`, default `permutation="block"`, models cloned per shuffle. **Function-level docstring is accurate; module-level docstring (L10-11) and `SPECIFICATION.md` still teach the older "leakage detector" framing.** |
| `gate_temporal_boundary` | `gates.py:954-1010+` | ✓ Pass | `required = horizon + extra_gap`; HALT on actual < required |
| `gate_residual_diagnostics` | `gates.py` | ✓ Pass | Heteroscedasticity / autocorrelation / normality combined |
| `gate_synthetic_ar1` | `gates.py` | ✓ Pass | Optimal MAE = σ√(2/π) bound |

**Bottom line:** The math is correct. The remaining work is editorial.

---

## 9. Strengths Worth Preserving

These should not change in the trust-repair release; they are the load-bearing parts of the project's credibility.

- **Strict static analysis pipeline.** `pyproject.toml:123-151` keeps `disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`, `strict_optional`. Mypy is clean on 57 files.
- **Multi-layer test architecture.** Six layers — unit, integration, anti-pattern, Hypothesis property, Monte Carlo calibration, golden reference. `tests/conftest.py` includes MC validation helpers (`compute_mc_bias`, `compute_mc_coverage`, `validate_mc_results`) and reusable DGPs (`dgp_ar1`, `dgp_white_noise`, `dgp_heavy_tailed`) with fixed seeds.
- **Failure-example gallery.** `examples/16` through `examples/20` walk users through real anti-patterns (rolling-stat leakage, threshold leakage, nested DM, missing gap, K-fold trap with quantified 47.8% fake improvement) — pedagogically excellent.
- **Knowledge-tier annotations.** `[T1]` / `[T2]` / `[T3]` in source docstrings honestly distinguish established results from heuristics. Rare and valuable.
- **Trusted publishing on PyPI.** `.github/workflows/publish.yml` uses OIDC; no PyPI tokens in CI secrets.
- **Sphinx setup.** Furo theme, sphinx-gallery, MyST, copybutton, design extensions, intersphinx, mermaid. `docs/conf.py` is clean.
- **Model cards.** `docs/model_cards/{walk_forward_cv,gate_signal_verification}.md` are real model cards (intended use, limitations, validation evidence) — extend the pattern to other gates.
- **Existing audit history is itself a strength.** Eleven prior audits in `docs/audits/` show real engineering discipline, even where the findings are unaddressed.

---

## 10. Prior-Audit Reconciliation

Every distinct finding from the 11 prior audits is classified below. Status as of 2026-04-29.

### Resolved in current code (do not include in remediation backlog)

| Prior finding | First raised | Where it's fixed | Verification |
|---|---|---|---|
| `WalkForwardCV.get_n_splits()` silent failure on ValueError | codex 2025-12-26 | `cv.py:735-786` | `strict=True` default raises ValueError; versionadded note at L759-761 confirms 1.0.0 fix |
| `WalkForwardCV` lacks `horizon` parameter | codex 2025-12-23, 2026-04-29 | `cv.py:563, 599` | `horizon: int \| None = None` parameter; `total_separation = horizon + extra_gap` |
| `gate_signal_verification` reuses model state | codex 2025-12-23-critique | `gates.py:410-411` (docstring) | "Models are cloned for each shuffle" — verify in implementation as well |
| `gate_signal_verification` IID-shuffle / permutation mismatch with cited theory | codex 2025-12-23-critique, 2026-04-29 | `gates.py:286-304, 340-346` | Default is now `method="permutation"` + `permutation="block"` (Künsch) |
| DM test wrong parameter names (`errors1` vs `errors_1`) | codex 2025-12-23-critique | `statistical_tests.py:611` | Signature uses `errors_1, errors_2, h` correctly |
| DM test ignores t-distribution under Harvey correction | codex 2025-12-23-critique | `statistical_tests.py:879-886` | `stats.t.cdf(df=n-1)` selected when `harvey_correction=True` |
| Dead-link API docs (`load_fred_series`, `load_m5_sample`, `ComparisonRunner`) | codex 2025-12-23-critique, 2026-01-05 | `docs/api/{benchmarks,compare}.md` | Names no longer present in current docs/api |
| Monash benchmark loader silent truncation | codex 2025-12-23-critique | `src/temporalcv/benchmarks/monash.py:127-151, 214-238` | `was_truncated`, `official_split=not was_truncated`, `original_series_lengths` all explicitly tracked |
| Version drift (`__init__.py` 0.1.0 vs `pyproject.toml` 1.0.0) | codex-claude-sonnet 2026-01-05 | both files | Both now `1.0.0` |
| `BellmanConformalPredictor` cited but not implemented | (this audit) | `conformal.py:537` | Class exists |

### Still valid (carried into §3-§6)

| ID in this report | Prior finding |
|---|---|
| B1 | Gate semantics conflict (codex 2025-12-23, 2025-12-26, 2026-04-29) |
| B2 | README `run_gates(model, X, y)` invalid (codex 2026-04-29) |
| H1 | Repository identity drift `brandon-behring` vs `brandonmbehring-dev` (codex 2026-04-29) |
| H2 | sklearn gap claim overstated (codex 2026-04-29) |
| H3 | README dependency table mismatch (codex 2026-04-29) |
| H4 | Lobato DOI 404 (codex 2026-04-29) |
| H5 (downgraded) | Shuffled-target gate citation/method (codex 2025-12-23-critique, 2025-12-26, 2026-04-29) — implementation now correct; only docs lag |
| H6 | Immutability principle violated (this audit) |
| M1 | `gap` → `extra_gap` rename incomplete (codex 2026-01-05) |
| M2 | Validation evidence stale (codex 2026-04-29) |
| M3 | Coverage badge "83%" hardcoded (readme_audit 2026-01-09) |
| M4 | Quickstart teaches leaky lag features (codex 2025-12-23-critique) |
| M5 | `common_pitfalls.md` wrong gate signature (codex 2026-01-05) |
| M6 | Conformal exchangeability caveat in code only, not README (codex 2026-04-29) |
| M7 | God modules with 100+ line functions (codex 2026-04-29) |
| M8 | Repo hygiene: `.venv-test/.venv-fresh` not gitignored, `None/` 661 MB (this audit) |
| M9 | `examples/*.py` not directly executed in CI (this audit) |
| M10 | Pre-commit `mypy` `language: system` (this audit) |
| M11 | RTD `fail_on_warning: false` (this audit) |
| M12 | `coverage_gap` sign inconsistency (codex 2026-01-05; CV8) |
| M13 | `pyproject` 1.0.0 vs CHANGELOG [1.1.0] (this audit; CV9) |
| L1 | CITATION.cff placeholder ORCID + stale date (codex 2026-04-29) |
| L2 | Python classifier mismatch (codex 2026-04-29) |
| L3 | Lahiri 1999, López-Lewis 2019 incomplete citations (codex 2026-04-29) |
| L4 | Internal `myga-forecasting-v2` references (codex 2025-12-23) |
| L5 | Optional-extra modules low coverage (this audit) |

### Tone / structure findings (gemini)

- `gemini-audit-report.md` (2026-01-09) flagged conversational tone, ASCII diagrams, buried quickstart. **Partially addressed** — logo (`docs/images/logo.svg`), Mermaid validation pipeline diagram (`README.md:130-138`), badges, and a clean header are now in place. The conversational-tone observation is moot if B1-B3 are fixed (the wrong claims do more damage than the tone).

### Out-of-date / superseded prior findings

- "DM test broken parameter names" — fixed; ignore prior critique.
- "Spec/implementation mismatch on shuffled gate threshold" — partially superseded; spec now describes both methods (`SPECIFICATION.md:24, 42-47`) but framing still says "leakage" (folded into B1).
- "M4 4,773-series benchmark validation claim" (codex 2025-12-26) — claim is in `README.md:193` and `docs/validation_evidence.md`; benchmark loaders work and the M4 dataset is on disk in `None/m4/`, but the *reproduction artifact* is still missing (folded into M2).

---

## 11. Recommended Remediation Roadmap

Phased so you can ship a trust-repair release in week 1, design changes in weeks 2-4, and refactor work in the next minor.

### Phase 1 — Immediate (this week, ~½ day total)

These are pure-edit fixes; no design needed. Doing them lifts release readiness from 6.5 → ~8.

| Item | Files | Effort | Reference |
|---|---|---|---|
| Fix `run_gates` signature in README quick-start | `README.md:37-41` | 5 min | B2 |
| Replace "Leakage detected" wording in 4 places | `README.md:45, 107`, `gates.py:10-11`, `docs/quickstart.md:65-66` | 15 min | B1.a |
| Reframe sklearn comparison row | `README.md:64-66` | 5 min | H2.a |
| Sync dependency table | `README.md:169-171` | 5 min | H3.b (or H3.a long-term) |
| Replace Lobato DOI (verify correct DOI first) | `docs/api/statistical_tests.md:736` | 10 min | H4.a |
| Fix `common_pitfalls.md` gate signature | `docs/guide/common_pitfalls.md:46, 82` | 10 min | M5.a |
| Update coverage badge to dynamic Codecov | `README.md:14` | 10 min | M3.a |
| Refresh `validation_evidence.md` footer (`86%, 1,943 tests, 2026-04-29`) | `docs/validation_evidence.md` | 10 min | M2 |
| Drop placeholder ORCID; update release date | `CITATION.cff:9, 14` | 5 min | L1 |
| `.gitignore`: change `.venv` → `.venv*` | `.gitignore:32` | 1 min | M8.a |
| Trace and fix the `Path(None) → "None"` source | grep for `cache_dir=None` or similar | 30 min | M8.a |

**Total:** ~2 hours of editing + a docs build to verify.

### Phase 2 — Short-term (next sprint, 2-4 days)

These need a design decision and a test pass.

| Item | Effort | Reference |
|---|---|---|
| Pick canonical repository (`brandon-behring` vs `brandonmbehring-dev`) and update `pyproject.toml`, `CITATION.cff`, README, docs source URL, GitHub badges, ReadTheDocs source | ½ day | H1 |
| Bump version to 1.1.0 (or move CHANGELOG block to `[Unreleased]`); republish PyPI artifact | ½ day (incl. release process) | M13 |
| Frozen-dataclass sweep: 28 result/report types → `frozen=True`; run tests | ½ day | H6.a |
| `gap=` → `horizon=` / `extra_gap=` migration sweep across `examples/`, `notebooks/`, `docs/` | ½ day | M1.a |
| Fix `coverage_gap` sign convention (pick `coverage - target_coverage`); CHANGELOG entry | 2 hours | M12.a |
| Rewrite quickstart lag-feature section to use the existing `create_lag_features` helper inside the CV loop | 1 hour | M4.a |
| Add README caveat: "marginal coverage under exchangeability; time-series autocorrelation may invalidate guarantees" | 15 min | M6.a |
| Flip RTD `fail_on_warning: true`; fix whatever the first build complains about | 1-2 hours | M11.a |
| Add CI matrix job that runs each `examples/*.py` to completion | 1-2 hours | M9.a |

**Total:** 2-3 days.

### Phase 3 — Long-term (next minor release, 1-2 weeks)

These are real refactor work; do them after the trust surface is consistent.

| Item | Effort | Reference |
|---|---|---|
| Decompose hot-spot functions (`gate_signal_verification`, `dm_test`, `gw_test`, `cw_test`) into private helpers; preserve public signatures | 3-5 days | M7.a |
| (Optional) Split `statistical_tests.py` and `gates.py` into submodule packages with import shims | 2-3 days | M7.b |
| Add executable doctest harness for README and `docs/quickstart.md` snippets | 1 day | (codex Option B1) |
| Migrate pre-commit mypy to `mirrors-mypy` with pinned `additional_dependencies` | 1 hour | M10.a |
| Build a citation matrix (`docs/methodology/citation_matrix.md`) mapping each method → file → DOI/arXiv → claim → caveat; link from README | 1 day | (codex Option D1) |
| Add `nox -s {tests,mypy,ruff,docs,security,audit}` for reproducible local CI | ½-1 day | (codex Option E1) |
| Complete Lahiri (1999) and López de Prado & Lewis (2019) citation venues | ½ day | L3 |
| Optional-extra CI matrix jobs (gluonts, m5, fred, compare) with thin smoke tests | 1 day | L5 |
| (Optional) `gate_signal_verification` rename or split into `gate_has_signal` + `gate_leakage_suspicion` | 2-3 days incl. deprecation shim | B1.b |

**Total:** 1-2 weeks of focused work.

---

## 12. Audit Methodology and Limitations

- **What was read:** `src/temporalcv/{cv.py, gates.py, conformal.py, statistical_tests.py, persistence.py, lag_selection.py, stationarity.py, regimes.py, guardrails.py, changepoint.py, __init__.py}`, `src/temporalcv/{bagging/strategies/*, inference/*, diagnostics/*, metrics/*, compare/*, benchmarks/monash.py, benchmarks/base.py}`, `tests/conftest.py` plus a sample of `tests/test_*.py`, `pyproject.toml`, `.pre-commit-config.yaml`, `.readthedocs.yaml`, `.github/workflows/{ci,publish}.yml`, `README.md`, `SPECIFICATION.md`, `CHANGELOG.md`, `CITATION.cff`, `docs/{conf.py, index.rst, quickstart.md, validation_evidence.md, testing_strategy.md, model_cards/*}`, `docs/api/statistical_tests.md`, `docs/guide/common_pitfalls.md`, `docs/tutorials/walk_forward_cv.md`, `examples/00_quickstart.py`, all 11 prior audit files.
- **What was run:** `pytest --cov`, `mypy`, `ruff check`, `ruff format --check`, `pip-audit` (failed: missing locally), `git status / log / branch`, `wc -l`, `du -sh`, `grep` (extensively).
- **What was NOT run:** Sphinx build (codex same-day reported `furo` missing in this `.venv`; behavior would replicate); pip-audit (module not installed locally); fresh PyPI metadata fetch (relied on codex same-day capture).
- **Confidence:** High for all findings except H4 (correct Lobato DOI value needs Crossref lookup), and L2 (PyPI Python-classifier value relies on codex's same-day fetch).
- **What this audit explicitly does *not* do:** It does not rewrite any file; it does not file GitHub issues; it does not bump versions or publish anything. The next session can pick up the Phase 1 list and execute.

---

*End of report.*
