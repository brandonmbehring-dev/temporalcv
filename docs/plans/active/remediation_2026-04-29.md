# temporalcv Remediation Plan — 2026-04-29

- **Source audit:** [`docs/audits/claude-audit-2026-04-29.md`](../../audits/claude-audit-2026-04-29.md) (commit `5e61658`)
- **Generated:** 2026-04-29
- **Scope:** All 26 still-valid findings (B1–B2, H1–H6, M1–M13, L1–L5) plus 4 codex-derived Phase 3 infrastructure tasks
- **Lifecycle:** `active` → `implemented` → `archived` (per `docs/plans/INDEX.md`)
- **Total tasks:** 28 (Phase 1: 10, Phase 2: 9, Phase 3: 9)

## How to use this document

1. Tasks are checkboxes (`### [ ]`). Tick `[ ]` → `[x]` as you complete each one. `grep -c "^### \[x\]"` reports progress.
2. Within each phase, tasks are grouped by file touched, not by severity. Severity is preserved via the Critical-Path Top 5 (§2) and the dashboard (§1).
3. Each task is self-contained: file paths, exact action, verification command. You should not need to re-open the audit during execution.
4. Three tasks (H1, H4, H6) have inline `Fallback:` lines — read those before starting in case the recommended path hits an obstacle.
5. Each phase ends with an exit gate. All gate items must pass before tagging the corresponding release.
6. After Phase N ships: move on to Phase N+1.
7. After all phases ship: move this file to `docs/plans/implemented/`, add a row to `docs/plans/INDEX.md`, and update CHANGELOG.

---

## 1. Status dashboard

### Tasks by phase × severity (counts task blocks, not absorbed findings)

| | Blocker | High | Medium | Low | Extra | Total |
|---|---:|---:|---:|---:|---:|---:|
| Phase 1 (Immediate) | 2 | 3 | 4 | 1 | 0 | 10 |
| Phase 2 (Short-term) | 0 | 2 | 7 | 0 | 0 | 9 |
| Phase 3 (Long-term) | 0 | 0 | 3 | 2 | 4 | 9 |
| **Total** | **2** | **5** | **14** | **3** | **4** | **28** |

**Absorbed audit findings (covered inside another task block, no separate slot):**
- **H5** absorbed by **B1** — same editorial pass on `gates.py:10-22` and `SPECIFICATION.md`
- **L2** absorbed by **M13** — Python-version classifier refresh happens automatically on PyPI republish
- **L4** absorbed by **CITMTX** — internal `myga-forecasting-v2` refs go in the citation matrix doc

All 26 audit B/H/M/L findings remain individually traceable below via `Audit ref:` lines. **M7** is split into two blocks (`M7.a` decompose, `M7.b` optional split) since they have different effort estimates. The 4 "Extra" tasks come from audit §11 codex options (DOCTEST, CITMTX, NOX, B1.b).

### Release-readiness scoreboard (target trajectory)

| Phase | Release tag | Code quality | Methodology | Tests | Docs | **Overall** |
|---|---|---:|---:|---:|---:|---:|
| Now (`5e61658`) | — | 8.5 | 8.0 | 9.0 | 5.5 | **6.5** |
| After Phase 1 | `1.0.1` (trust-repair patch) | 8.5 | 8.5 | 9.0 | 8.5 | **8.5** |
| After Phase 2 | `1.1.0` (next minor) | 9.0 | 9.0 | 9.0 | 9.0 | **9.0** |
| After Phase 3 | `1.2.0` (refactor) | 9.5 | 9.5 | 9.5 | 9.5 | **9.5** |

---

## 2. Critical-path top 5

If you have only one hour, do these:

1. **B1** — Reconcile gate semantics across public docs (Phase 1, 15 min)
2. **B2** — Fix invalid `run_gates(model, X, y)` in README quickstart (Phase 1, 5 min)
3. **H4** — Replace broken Lobato DOI (Phase 1, 10 min — Crossref lookup required)
4. **L1** — Drop placeholder ORCID, fix release date in `CITATION.cff` (Phase 1, 5 min)
5. **M2 + M3** — Refresh `validation_evidence.md` footer and Codecov badge (Phase 1, 20 min)

---

## 3. Phase 1 — Immediate (~2 hours)

- **Target release:** `1.0.1` (trust-repair patch)
- **Risk profile:** Low. Pure docs/config edits; no API surface change; tests should be unchanged.
- **Task count:** 10

### Phase 1 ordering notes

- **B1 first.** Do B1 (cross-file leakage wording) before any other README/docs work — it changes the framing the rest of the page assumes.
- **H1 (Phase 2) is the only pre-req that could leak in.** If you choose to advance H1 into Phase 1, complete it *before* M3 (Codecov badge URL depends on canonical repo) and L1 (`CITATION.cff` `repository-code` URL).
- **M8 is two sub-tasks.** The `.gitignore` edit is 1 min; tracing the `Path(None) → "None"` source is ~30 min of grepping.
- **Bundling tip.** All `README.md` edits (B1-portion, B2, H2, H3, M3) fit one editor sitting; same for the `docs/` block.

### 3.1 Cross-file task (do first)

#### [ ] B1 — Reconcile gate semantics across public docs
- **Severity:** Blocker  •  **Effort:** 15 min  •  **Phase:** 1  •  **Audit ref:** §3 B1 (also resolves §4 H5 doc lag)
- **Depends on:** none  •  **Blocks:** B2, H2, H3, M3, M5  •  **File group:** cross-file

**Files & lines:**
- `README.md:45` — feature-table row `HALT | Leakage detected — stop and investigate`
- `README.md:107` — `if report.status == "HALT": raise ValueError(f"Leakage detected: ...")`
- `src/temporalcv/gates.py:10-11` — module docstring "if a model beats a shuffled target … it's likely learning from leakage"
- `src/temporalcv/gates.py:20-22` — `[T3]` knowledge-tier annotation framing effect-size threshold as the headline
- `docs/quickstart.md:65-66` — `raise ValueError("Leakage detected!")`
- `SPECIFICATION.md:24` — `HALT if: improvement > 0.20`
- `SPECIFICATION.md:42-47, 58` — leakage-framing language

**Source of truth (already correct):** `src/temporalcv/gates.py:305-315` — "HALT indicates the model has learned signal — this could be legitimate temporal patterns OR data leakage."

**Action:** Replace every user-facing "Leakage detected" string with "Signal detected — investigate" framing. In `gates.py:10-22` and `SPECIFICATION.md`, additionally make **permutation (block)** the headline method and effect-size the optional fast heuristic — matching the defaults at `gates.py:286-304`.

**Verification:**
```bash
rg -n "Leakage detected" README.md docs/ SPECIFICATION.md src/temporalcv/gates.py  # → no matches
.venv/bin/python -m pytest tests/test_gates.py -q                                   # → still passes
```

### 3.2 `README.md` block (4 tasks, one editor sitting)

#### [ ] B2 — Fix invalid `run_gates(model, X, y)` quickstart signature
- **Severity:** Blocker  •  **Effort:** 5 min  •  **Phase:** 1  •  **Audit ref:** §3 B2
- **Depends on:** B1  •  **Blocks:** none  •  **File group:** README.md

**Files & lines:**
- `README.md:37-41` — current invalid: `report = run_gates(model, X, y)`
- `README.md:98-108` — already-correct example to copy: `gates = [...]; report = run_gates(gates)`
- Real signature: `src/temporalcv/gates.py:1483-1509` — `run_gates(gates: list[GateResult]) -> ValidationReport`

**Action:** Replace the L37-41 quickstart snippet with the L98-108 idiom (build a `gates` list, then call `run_gates(gates)`).

**Verification:**
```bash
rg -n "run_gates\(model" README.md   # → no matches
# Optional: copy the new snippet into a scratch script and run it end-to-end
```

#### [ ] H2 — Reframe sklearn comparison row
- **Severity:** High  •  **Effort:** 5 min  •  **Phase:** 1  •  **Audit ref:** §4 H2
- **Depends on:** none  •  **Blocks:** none  •  **File group:** README.md

**Files & lines:**
- `README.md:64-66` — current: `Gap enforcement | ✓ | ✗ | ✗ | ✗` (claims sklearn lacks gap)
- Counter-evidence: `src/temporalcv/cv.py:39-40` already acknowledges sklearn `TimeSeriesSplit(gap=…)` since v0.24

**Action:** Reframe the row as **"Manual gap (sklearn) vs horizon-derived gap + leakage gates (temporalcv)"**. The honest pitch is stronger: temporalcv adds horizon-aware gap derivation, validation gates, and statistical layers around the boundary.

**Verification:** `rg -n "Gap enforcement" README.md` returns the rewritten row; manual read confirms claim is accurate.

#### [ ] H3 — Sync dependency table to `pyproject.toml`
- **Severity:** High  •  **Effort:** 5 min  •  **Phase:** 1  •  **Audit ref:** §4 H3
- **Depends on:** none  •  **Blocks:** none  •  **File group:** README.md

**Files & lines:**
- `README.md:169-171` — current: `numpy>=1.23, scipy>=1.9, scikit-learn>=1.1, pandas>=1.5` (wrong)
- Truth: `pyproject.toml:34-40` — `numpy>=1.21, scipy>=1.7, scikit-learn>=1.0, statsmodels>=0.13, matplotlib>=3.5`; `pyproject.toml:55` — `pandas>=1.3` is **optional**

**Action:** Replace the dependency line to read: `Core: numpy>=1.21, scipy>=1.7, scikit-learn>=1.0, statsmodels>=0.13, matplotlib>=3.5. Optional: pandas>=1.3`.

**Verification:**
```bash
diff <(rg -o 'numpy>=[0-9.]+' README.md | head -1) <(rg -o 'numpy>=[0-9.]+' pyproject.toml | head -1)  # → match
```

#### [ ] M3 — Replace static "83%" coverage badge with Codecov dynamic badge
- **Severity:** Medium  •  **Effort:** 10 min  •  **Phase:** 1  •  **Audit ref:** §5 M3
- **Depends on:** H1 (if H1 is moved into Phase 1; otherwise none)  •  **Blocks:** none  •  **File group:** README.md

**Files & lines:** `README.md:14` — current: `https://img.shields.io/badge/coverage-83%25-green`

**Action:** Replace with Codecov badge: `https://codecov.io/gh/<canonical-repo>/branch/main/graph/badge.svg`. CI already uploads to Codecov per `.github/workflows/ci.yml`.

**Verification:** `curl -sI <new-badge-url>` returns 200; `rg "coverage-83" README.md` returns no matches.

### 3.3 `docs/` block (3 tasks)

#### [ ] H4 — Replace broken Lobato DOI
- **Severity:** High  •  **Effort:** 10 min (incl. Crossref lookup)  •  **Phase:** 1  •  **Audit ref:** §4 H4
- **Depends on:** none  •  **Blocks:** none  •  **File group:** docs/

**Files & lines:** `docs/api/statistical_tests.md:736` — DOI `10.1198/016214501750333073` returns 404 (Crossref-verified).

**Action:** Look up Lobato (2001) "Testing That a Dependent Process Is Uncorrelated", *JASA* 96(453), 169-176, on Crossref. Codex same-day suggested `10.1198/016214501750332811`; **verify against Crossref before substituting**. Replace the DOI in the citation.

**Fallback:** If Crossref does not return a valid replacement DOI, drop the hyperlink and keep the bibliographic citation in plain text (audit §4 H4.b).

**Verification:**
```bash
curl -sI "https://doi.org/<new-doi>" | head -1   # → 200 or 302 (not 404)
rg -n "10.1198/016214501750333073" docs/         # → no matches
```

#### [ ] M2 — Refresh `validation_evidence.md` footer
- **Severity:** Medium  •  **Effort:** 10 min  •  **Phase:** 1  •  **Audit ref:** §5 M2
- **Depends on:** none  •  **Blocks:** none  •  **File group:** docs/

**Files & lines:** `docs/validation_evidence.md:254-255` — current "**Last Updated**: 2026-01-09 / **Coverage**: 83% (318 tests passing)"; truth (audit §2): 1,943 tests, 86% coverage, 80 s, 2026-04-29.

**Action:** Update footer to: `**Last Updated**: 2026-04-29  •  **Tests**: 1,943 passing, 15 skipped  •  **Coverage**: 86% (5,898 statements, 1,956 branches)  •  **Runtime**: ~80 s`. Also fix the bogus `--run-slow` flag reference; replace with `pytest -m monte_carlo` and `pytest -m "not slow"` per `tests/conftest.py:25-29`.

**Verification:**
```bash
rg -n "318 tests|--run-slow|83%" docs/validation_evidence.md  # → no matches
```

#### [ ] M5 — Fix `gate_signal_verification` signature in `common_pitfalls.md`
- **Severity:** Medium  •  **Effort:** 10 min  •  **Phase:** 1  •  **Audit ref:** §5 M5
- **Depends on:** B1 (terminology)  •  **Blocks:** none  •  **File group:** docs/

**Files & lines:**
- `docs/guide/common_pitfalls.md:46` — invalid: `result = gate_signal_verification(y_train, y_test)`
- `docs/guide/common_pitfalls.md:82` — invalid: `gate_signal_verification(y_train, y_test, features=X_train)` (no `features` kwarg exists)
- Real signature: `src/temporalcv/gates.py:286-304` — `gate_signal_verification(model, X, y, n_shuffles=…, threshold=…, …)`

**Action:** Rewrite both call sites to use `(model, X, y, …)` form. Use the same model/feature variables already established earlier in the page.

**Verification:**
```bash
rg -n "gate_signal_verification\(y_" docs/      # → no matches
.venv/bin/python -c "
from temporalcv.gates import gate_signal_verification
help(gate_signal_verification)
" | head -10                                     # confirms current signature
```

### 3.4 Config block (3 tasks)

#### [ ] L1 — Fix `CITATION.cff` placeholder ORCID and stale release date
- **Severity:** Low  •  **Effort:** 5 min  •  **Phase:** 1  •  **Audit ref:** §6 L1
- **Depends on:** H1 (if H1 is in Phase 1, for `repository-code` URL)  •  **Blocks:** none  •  **File group:** config

**Files & lines:**
- `CITATION.cff:9` — `orcid: "https://orcid.org/0000-0000-0000-0000"` (placeholder)
- `CITATION.cff:14` — `date-released: "2025-01-05"` (PyPI shows 2026-01-08 for `1.0.0`)

**Action:** Replace ORCID with the real value or remove the line. Update `date-released` to `2026-01-08`.

**Verification:**
```bash
rg -n "0000-0000-0000-0000|2025-01-05" CITATION.cff   # → no matches
.venv/bin/python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"  # → no parse errors
```

#### [ ] M8 — `.gitignore` `.venv` → `.venv*`; trace the `Path(None) → "None/"` source
- **Severity:** Medium  •  **Effort:** 1 min + ~30 min  •  **Phase:** 1  •  **Audit ref:** §5 M8
- **Depends on:** none  •  **Blocks:** none  •  **File group:** config + grep

**Files & lines:**
- `.gitignore:32` — currently `.venv` (literal); does not match `.venv-test`, `.venv-fresh`
- `du -sh None/` reports 661 MB on disk; `.gitignore:51-54` already excludes `None/` from VCS but the directory itself shouldn't exist
- Likely root cause: a `cache_dir=None` literal path producing `Path("None")` somewhere in the benchmark loaders

**Action (two parts):**
1. Edit `.gitignore:32` from `.venv` to `.venv*` (also matches `.venv-test/`, `.venv-fresh/`, `.venv-old/`).
2. `rg -n "cache_dir.*=.*None|Path\(None\)|str\(None\)" src/ benchmarks/` to locate the source. Likely candidates: `benchmarks/m4.py`, `benchmarks/monash.py`, `benchmarks/__init__.py`. Fix the offending call to either omit the kwarg (use the function's default `cache_dir=None` semantics correctly) or pass an explicit valid path.

**Verification:**
```bash
git check-ignore .venv-test/ .venv-fresh/        # → both reported as ignored
rg -n "Path\(None\)|cache_dir=['\"]None['\"]" src/ benchmarks/  # → no matches after fix
rm -rf None/  &&  .venv/bin/python -m pytest tests/benchmarks/ -q  # → None/ does not regenerate
```

### Phase 1 exit gate

All of the following must pass before tagging `1.0.1`:

- [ ] `.venv/bin/python -m pytest tests/ --cov=temporalcv --cov-fail-under=80 -q --ignore=tests/benchmarks/`
- [ ] `.venv/bin/python -m mypy src/temporalcv`
- [ ] `.venv/bin/python -m ruff check src/temporalcv tests/`
- [ ] `.venv/bin/python -m ruff format --check src/temporalcv tests/`
- [ ] `.venv/bin/python -m pip install -e .[docs]  &&  cd docs  &&  ../.venv/bin/sphinx-build -W --keep-going -b html . _build/html` (M11 may surface; defer fix to Phase 2)
- [ ] `rg -n "Leakage detected|run_gates\(model,|coverage-83%|318 tests|0000-0000-0000-0000" README.md docs/ SPECIFICATION.md CITATION.cff` → empty
- [ ] `git diff --stat origin/main` — size-of-change sanity (expect ~10 files, < 200 lines)
- [ ] CHANGELOG entry drafted under `## [1.0.1] - 2026-04-29` with the 10 task IDs as bullets
- [ ] Tag `v1.0.1` and let trusted publishing push to PyPI

---

## 4. Phase 2 — Short-term (~2–3 days)

- **Target release:** `1.1.0` (next minor)
- **Risk profile:** Medium. Includes API-adjacent changes (frozen-dataclass sweep), version bump, repository identity migration. Tests must remain green throughout.
- **Task count:** 9 (L2 absorbed in M13)

### Phase 2 ordering notes

- **H1 first.** Repository-identity unification is the keystone — it touches `pyproject.toml`, `CITATION.cff`, `README.md` badges, RTD source URL, GitHub Pages config. Doing H1 first avoids redoing M3 / L1 URL fields if those were touched in Phase 1.
- **H1 → M13.** Bump version *after* H1 so the republish targets the correct repository / Trusted Publishing config. M13 also absorbs L2 (Python-version classifier refresh).
- **H6 → M1 → M12.** Frozen-dataclass sweep first (touches result types broadly), then the `gap=` → `horizon=` migration (touches examples/notebooks/docs), then the `coverage_gap` sign fix (narrow, in `conformal.py`).
- **M11 last in this phase.** Flipping RTD `fail_on_warning: true` will surface latent doc errors — easiest after the doc rewrites in Phase 1 + this phase have settled.

### Task blocks

#### [ ] H1 — Unify canonical repository identity
- **Severity:** High  •  **Effort:** ½ day  •  **Phase:** 2  •  **Audit ref:** §4 H1
- **Depends on:** none  •  **Blocks:** M13, L1 (if not already done), M3 (if not already done)  •  **File group:** config + docs

**Files & lines:**
- `pyproject.toml:94-98` — Homepage / Documentation / Repository / Issues all `https://github.com/brandon-behring/temporalcv`
- `CITATION.cff:10` — `repository-code: "https://github.com/brandon-behring/temporalcv"`
- `README.md:10, 15, 201-208` (BibTeX) — same
- `.readthedocs.yaml` — source URL
- PyPI Trusted Publishing provenance for `temporalcv 1.0.0` currently points at `brandonmbehring-dev/temporalcv@5f2048b…` on tag `v1.0.0`

**Action:** Adopt `brandonmbehring-dev/temporalcv` (matches signed PyPI artifact) across `pyproject.toml`, `CITATION.cff`, `README.md` (text + badges + BibTeX), and `.readthedocs.yaml`.

**Fallback:** *User must choose between `brandon-behring` (current local metadata) and `brandonmbehring-dev` (current PyPI provenance).* Recommend `brandonmbehring-dev` to match the signed artifact. If the user prefers `brandon-behring`, additionally re-configure PyPI Trusted Publishing on that org and re-publish 1.0.0 from the canonical source — otherwise the next release will fail authentication.

**Verification:**
```bash
rg -n "brandon-behring/temporalcv" .                 # → no matches (or only matches in this remediation plan / audit)
rg -n "brandonmbehring-dev/temporalcv" pyproject.toml CITATION.cff README.md .readthedocs.yaml  # → all four present
```

#### [ ] M13 — Bump version `1.0.0` → `1.1.0`; absorb L2 classifier refresh
- **Severity:** Medium  •  **Effort:** ½ day  •  **Phase:** 2  •  **Audit ref:** §5 M13 (also resolves §6 L2)
- **Depends on:** H1  •  **Blocks:** none  •  **File group:** config + release

**Files & lines:**
- `pyproject.toml:3` — version
- `src/temporalcv/__init__.py:47` — `__version__`
- `CHANGELOG.md:14` — `## [1.1.0] - 2026-01-01` already exists with feature list (currently aspirational)
- PyPI metadata advertises Python 3.9 (per audit L2); local truth is `>=3.10`

**Action:** Bump both version strings to `1.1.0`. Verify the `[1.1.0]` CHANGELOG block reflects what actually shipped (move the doc-only items added in Phase 1 to a fresh `[1.0.1]` block if you released that intermediate). Tag `v1.1.0`. Trusted publishing republishes; PyPI metadata refresh fixes the Python-classifier mismatch automatically.

**Verification:**
```bash
diff <(rg -o '"[0-9]+\.[0-9]+\.[0-9]+"' pyproject.toml | head -1) \
     <(rg -o '"[0-9]+\.[0-9]+\.[0-9]+"' src/temporalcv/__init__.py | head -1)
# → match: "1.1.0"
.venv/bin/python -c "import temporalcv; assert temporalcv.__version__ == '1.1.0'"
# After PyPI publish:
.venv/bin/python -m pip index versions temporalcv | grep 1.1.0
```

#### [ ] H6 — Frozen-dataclass sweep on 28 mutable result/report types
- **Severity:** High  •  **Effort:** ½ day  •  **Phase:** 2  •  **Audit ref:** §4 H6
- **Depends on:** none  •  **Blocks:** none  •  **File group:** src/temporalcv/

**Files & lines (28 mutable dataclasses to freeze; full list in audit §4 H6):**
- `gates.py:78` (GateResult), `:114` (ValidationReport), `:1517` (StratifiedValidationReport)
- `conformal.py:61, :1360`
- `statistical_tests.py:76, 138, 190, 279, 1691, 2443, 2486, 2734, 2782`
- `compare/base.py:31, 108, 201`
- `regimes.py:370`, `persistence.py:72`, `metrics/event.py:55, 111`, `metrics/volatility_weighted.py:477`
- `guardrails.py:45`, `cv.py:136, 254, 425`, `benchmarks/base.py:72, 191`

**Action:** Add `frozen=True` to each `@dataclass` listed. Run the test suite after each module to localize any breakage. For `GateResult.details` (a mutable `dict`) — accept that freezing the dataclass does not freeze the dict, OR optionally switch to `MappingProxyType` for true content immutability.

**Fallback:** *If `frozen=True` breaks tests on a class whose `.details` dict is mutated downstream (e.g., reporting code overwriting a `metric_value`), leave that class mutable and add a docstring note explaining why.* Do not break the build to satisfy the convention — note the exception in the CHANGELOG and revisit later.

**Verification:**
```bash
.venv/bin/python -c "
import dataclasses, importlib
for mod in ['gates','conformal','statistical_tests','compare.base','cv','persistence']:
    m = importlib.import_module(f'temporalcv.{mod}')
    for name in dir(m):
        cls = getattr(m, name)
        if dataclasses.is_dataclass(cls) and 'Result' in name or 'Report' in name:
            assert getattr(cls, '__dataclass_params__').frozen, f'{mod}.{name} not frozen'
print('all result/report types frozen')
"
.venv/bin/python -m pytest tests/ -q --ignore=tests/benchmarks/  # → still passes
```

#### [ ] M1 — `gap=` → `horizon=` / `extra_gap=` migration sweep
- **Severity:** Medium  •  **Effort:** ½ day  •  **Phase:** 2  •  **Audit ref:** §5 M1
- **Depends on:** none  •  **Blocks:** none  •  **File group:** examples/, notebooks/, docs/

**Files & lines:**
- `cv.py:560-602` — current `WalkForwardCV.__init__` uses `extra_gap` and `horizon`
- Drift surface: `README.md:22`, `examples/*.py`, `notebooks/*.ipynb`, `docs/tutorials/*.md`, `docs/quickstart.md`, sphinx-gallery output
- Anything still using `gap=N` is pre-1.0 vocabulary

**Action:** `rg -n "gap=" examples/ notebooks/ docs/ README.md` to find call sites. For each, decide:
- If the original intent was "gap = horizon", rewrite to `horizon=N`.
- If the original intent was "additional gap beyond horizon", rewrite to `horizon=H, extra_gap=N`.
- If unclear, prefer `horizon=N` and leave a code comment.

**Verification:**
```bash
rg -n "WalkForwardCV\([^)]*gap=" examples/ notebooks/ docs/ README.md  # → no matches
.venv/bin/python -m pytest tests/ -q  # → still passes
```

#### [ ] M12 — Fix `coverage_gap` sign inconsistency
- **Severity:** Medium  •  **Effort:** 2 hours  •  **Phase:** 2  •  **Audit ref:** §5 M12
- **Depends on:** none  •  **Blocks:** none  •  **File group:** src/temporalcv/conformal.py

**Files & lines:**
- `conformal.py:1180, 1203` — `coverage_gap = coverage - target_coverage` (positive when over-covering) — **keep this convention**
- `conformal.py:1507` — `coverage_gap = target_coverage - overall_coverage` (positive when under-covering) — **fix**

**Action:** At `conformal.py:1507`, flip to `coverage_gap = overall_coverage - target_coverage`. Update any docstrings/tests that depend on the old sign. Add a CHANGELOG entry under `### Changed` warning users of the contract change.

**Verification:**
```bash
.venv/bin/python -m pytest tests/test_conformal.py -q
.venv/bin/python -c "
# Sanity: over-coverage produces positive gap in both APIs
from temporalcv.conformal import compute_coverage_diagnostics
# (compose minimal example matching both code paths; assert sign is consistent)
"
```

#### [ ] M4 — Quickstart leaky-lag rewrite
- **Severity:** Medium  •  **Effort:** 1 hour  •  **Phase:** 2  •  **Audit ref:** §5 M4
- **Depends on:** B1 (terminology), M1 (vocabulary)  •  **Blocks:** none  •  **File group:** docs/

**Files & lines:** `docs/quickstart.md:29-35` constructs lag features on the full series before splitting (the antipattern this library is supposed to prevent). The correct pattern already exists at `docs/tutorials/walk_forward_cv.md:189-193` (`create_lag_features` applied inside the CV loop).

**Action:** Replace the quickstart's lag-feature construction with a call to the existing `create_lag_features` helper, applied **inside** the per-split fit/predict block. Add an inline comment: `# Inside CV loop — features computed per-fold to prevent leakage`.

**Verification:**
```bash
rg -n "create_lag_features" docs/quickstart.md   # → present
# Manual: run the quickstart code end-to-end and verify it executes without leakage warnings
```

#### [ ] M6 — Add README exchangeability caveat for conformal coverage
- **Severity:** Medium  •  **Effort:** 15 min  •  **Phase:** 2  •  **Audit ref:** §5 M6
- **Depends on:** none  •  **Blocks:** none  •  **File group:** README.md

**Files & lines:**
- `README.md:51-58` — features list overstates "coverage guarantee" for time series
- `conformal.py:230-238` — runtime warning is honest; README is not

**Action:** Add a one-line caveat in the README features section: *"Conformal coverage: marginal under exchangeability; time-series autocorrelation may invalidate guarantees — see [`AdaptiveConformalPredictor`](docs/api/conformal.md#adaptive) for distribution-shift handling."*

**Verification:** `rg -n "exchangeability" README.md` returns the new caveat.

#### [ ] M11 — Flip ReadTheDocs `fail_on_warning: true`
- **Severity:** Medium  •  **Effort:** 1–2 hours  •  **Phase:** 2  •  **Audit ref:** §5 M11
- **Depends on:** B1, H4, M2, M5 (so the first build doesn't drown in warnings)  •  **Blocks:** none  •  **File group:** config

**Files & lines:** `.readthedocs.yaml:18-20` — `fail_on_warning: false`

**Action:** Change to `fail_on_warning: true`. Run `cd docs && ../.venv/bin/sphinx-build -W --keep-going -b html . _build/html` locally; fix every warning surfaced (broken cross-refs, missing autodoc targets, malformed RST). Common fixes: missing entries in `:toctree:`, dead `:ref:` targets after the doc rewrites in Phase 1.

**Verification:**
```bash
cd docs && ../.venv/bin/sphinx-build -W --keep-going -b html . _build/html  # → exits 0
# After RTD rebuild: check the build log on RTD dashboard shows zero warnings
```

#### [ ] M9 — CI matrix job for `examples/*.py`
- **Severity:** Medium  •  **Effort:** 1–2 hours  •  **Phase:** 2  •  **Audit ref:** §5 M9
- **Depends on:** M1 (so examples use current vocabulary)  •  **Blocks:** none  •  **File group:** .github/workflows/

**Files & lines:**
- `.github/workflows/ci.yml:120-171` — currently runs `notebooks` job on `*.ipynb`; `examples/00`–`20_*.py` are validated only via sphinx-gallery
- `examples/00_quickstart.py` … `examples/20_failure_kfold.py` — should run to completion in <60 s each

**Action:** Add an `examples` job to `ci.yml`:
```yaml
examples:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ["3.10", "3.12"]
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
    - run: pip install -e .[dev]
    - run: |
        for f in examples/*.py; do
          echo "::group::$f"
          python "$f" || exit 1
          echo "::endgroup::"
        done
```

**Verification:** Push to a branch; the new `examples` matrix job appears green in CI within ~5 min.

### Phase 2 exit gate

All of the following must pass before tagging `1.1.0`:

- [ ] All Phase 1 exit-gate items still pass.
- [ ] `python -c "import temporalcv; print(temporalcv.__version__)"` returns `1.1.0` and matches `pyproject.toml` and the latest CHANGELOG section header.
- [ ] `rg -n "brandon-behring/temporalcv" .` returns empty (or only matches in audit / this plan).
- [ ] `cd docs && ../.venv/bin/sphinx-build -W --keep-going -b html . _build/html` exits 0 (RTD `fail_on_warning: true` will pick this up).
- [ ] All result/report dataclasses are frozen except documented exceptions (audit-§4 list).
- [ ] CI `examples` matrix job green.
- [ ] CHANGELOG entry under `## [1.1.0]` enumerates Phase 2 task IDs.
- [ ] Tag `v1.1.0`; PyPI publish succeeds via Trusted Publishing on the canonical repo.

---

## 5. Phase 3 — Long-term (~1–2 weeks)

- **Target release:** `1.2.0` (refactor)
- **Risk profile:** Medium-high. Hot-spot decomposition touches statistical core; doctest harness changes how docs are validated.
- **Task count:** 9

### Phase 3 ordering notes

- **M7.a before M7.b.** Decompose hot-spot functions into private helpers first (internal-only churn), THEN consider splitting the four god-modules into submodule packages (M7.b is optional and breakier).
- **DOCTEST after B1.** Doctest harness assumes README/quickstart wording is final — Phase 1 must have shipped.
- **CITMTX absorbs L4.** The citation-matrix doc is the natural home for the internal `myga-forecasting-v2` references appendix; do them together.
- **M10 anytime.** `mirrors-mypy` migration is independent.

### Task blocks

#### [ ] M7.a — Decompose hot-spot functions into private helpers
- **Severity:** Medium  •  **Effort:** 3–5 days  •  **Phase:** 3  •  **Audit ref:** §5 M7
- **Depends on:** none  •  **Blocks:** M7.b  •  **File group:** src/temporalcv/

**Files & lines (function lengths):**
- `gate_signal_verification` 396 lines (`gates.py:286-682`)
- `dm_test` 298 lines (`statistical_tests.py:611-908`)
- `gw_test` ~273 lines, `cw_test` ~268 lines (`statistical_tests.py`)
- `gate_residual_diagnostics` ~202 lines, `gate_synthetic_ar1` ~172 lines (`gates.py`)
- `walk_forward_evaluate` ~173 lines, `NestedWalkForwardCV.fit` ~137 lines (`cv.py`)
- `compute_move_conditional_metrics` ~208 lines (`persistence.py`)
- Hub convention: 20–50 lines per function.

**Action:** For each function above, extract logical sub-steps into private helpers (`_dm_variance_hac`, `_dm_apply_harvey`, `_dm_pvalue`, etc.). Preserve the public function signatures exactly. Each helper gets a one-line docstring; aim for max function length < 100 lines.

**Verification:**
```bash
.venv/bin/python -c "
import ast, pathlib
for f in ['src/temporalcv/gates.py','src/temporalcv/statistical_tests.py','src/temporalcv/cv.py','src/temporalcv/conformal.py','src/temporalcv/persistence.py']:
    tree = ast.parse(pathlib.Path(f).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            n = node.end_lineno - node.lineno
            if n > 100:
                print(f'{f}:{node.lineno} {node.name} = {n} lines')
" | tee /tmp/long_functions.txt
# → /tmp/long_functions.txt should be empty
.venv/bin/python -m pytest tests/ --cov=temporalcv -q  # → all green; coverage unchanged
```

#### [ ] M7.b — (Optional) Split god-modules into submodule packages
- **Severity:** Medium  •  **Effort:** 2–3 days  •  **Phase:** 3  •  **Audit ref:** §5 M7 (option b)
- **Depends on:** M7.a  •  **Blocks:** none  •  **File group:** src/temporalcv/

**Files & lines:**
- `statistical_tests.py` 3,223 LOC → `statistical_tests/{dm.py, gw.py, cw.py, pt.py, multiple_testing.py, variance.py, __init__.py}`
- `gates.py` 1,802 LOC → `gates/{signal.py, boundary.py, residuals.py, reporting.py, __init__.py}`
- `cv.py` 1,943 LOC → `cv/{walk_forward.py, nested.py, financial.py, __init__.py}`
- `conformal.py` 1,547 LOC → `conformal/{split.py, adaptive.py, bellman.py, diagnostics.py, __init__.py}`

**Action:** For each module, create a package directory and re-export public names from `__init__.py` so existing imports (`from temporalcv.gates import gate_signal_verification`) keep working. Run `pytest` after each split.

**Verification:**
```bash
.venv/bin/python -c "
from temporalcv.gates import gate_signal_verification, gate_temporal_boundary
from temporalcv.statistical_tests import dm_test, gw_test, cw_test
from temporalcv.cv import WalkForwardCV
from temporalcv.conformal import SplitConformalPredictor
print('all public names still importable')
"
.venv/bin/python -m pytest tests/ -q
```

#### [ ] DOCTEST — Executable doctest harness for README and quickstart
- **Severity:** Improvement  •  **Effort:** 1 day  •  **Phase:** 3  •  **Audit ref:** §11 (codex Option B1)
- **Depends on:** B1, B2, M5  •  **Blocks:** none  •  **File group:** tests/

**Action:** Create `tests/test_documentation.py` that uses `doctest` (or `pytest --doctest-glob`) to execute every fenced ```python code block in `README.md`, `docs/quickstart.md`, and `docs/guide/common_pitfalls.md`. Add `pytest --doctest-glob="*.md" docs/ README.md` to `ci.yml`.

**Verification:** `pytest --doctest-glob="*.md" docs/ README.md -q` exits 0; an intentional break (rename `run_gates` to `run_gatez` in README and rerun) fails the harness.

#### [ ] M10 — Pre-commit `mypy` → `mirrors-mypy`
- **Severity:** Medium  •  **Effort:** 1 hour  •  **Phase:** 3  •  **Audit ref:** §5 M10
- **Depends on:** none  •  **Blocks:** none  •  **File group:** .pre-commit-config.yaml

**Files & lines:** `.pre-commit-config.yaml:17-25` currently uses `language: system` for mypy.

**Action:** Replace with a `repos:` entry pointing at `https://github.com/pre-commit/mirrors-mypy`, pin to a recent rev, and list `additional_dependencies` matching `pyproject.toml` `[dev]`. Example:
```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.11.2
  hooks:
    - id: mypy
      args: [--show-error-codes]
      additional_dependencies:
        - numpy>=1.21
        - scipy>=1.7
        - scikit-learn>=1.0
        - pandas-stubs
```

**Verification:**
```bash
pre-commit clean
pre-commit run mypy --all-files     # → passes from a clean state without local mypy install
```

#### [ ] CITMTX — Citation matrix doc (absorbs L4 internal references)
- **Severity:** Improvement (also resolves §6 L4)  •  **Effort:** 1 day  •  **Phase:** 3  •  **Audit ref:** §11 (codex Option D1) + §6 L4
- **Depends on:** L3 (citations completed)  •  **Blocks:** none  •  **File group:** docs/

**Action:** Create `docs/methodology/citation_matrix.md` with one row per method: file → DOI/arXiv → claim → caveat. Source rows from audit §7. Add a final section "Internal references" listing the `[T2] myga-forecasting-v2` annotations from `gates.py:18, 52` and `persistence.py`, with a one-paragraph explanation that these are internal validation references not publicly available. Link from README and `docs/index.rst`.

**Verification:**
```bash
ls docs/methodology/citation_matrix.md          # → exists
rg -n "myga-forecasting-v2" docs/methodology/citation_matrix.md  # → present
# Sphinx build picks up the new file without warnings (M11)
```

#### [ ] L3 — Complete Lahiri (1999) and López-Lewis (2019) citation venues
- **Severity:** Low  •  **Effort:** ½ day  •  **Phase:** 3  •  **Audit ref:** §6 L3
- **Depends on:** none  •  **Blocks:** CITMTX  •  **File group:** src/temporalcv/

**Files & lines:**
- `bagging/strategies/block_bootstrap.py:18` — Lahiri (1999) "Theoretical comparisons of block bootstrap methods" — likely *Annals of Statistics*, 27(1), 386-404; verify via Crossref
- `cv_financial.py` — López de Prado & Lewis (2019) — verify against *J. of Portfolio Management* 2019 paper or 2018 book

**Action:** Crossref-verify each citation, add full venue + DOI to the docstring.

**Verification:** `rg -n "Lahiri.*1999" src/temporalcv/` and `rg -n "López.*Lewis.*2019" src/temporalcv/` return entries with venue + DOI.

#### [ ] L5 — Optional-extra CI matrix jobs
- **Severity:** Low  •  **Effort:** 1 day  •  **Phase:** 3  •  **Audit ref:** §6 L5
- **Depends on:** none  •  **Blocks:** none  •  **File group:** .github/workflows/

**Files & lines (low-coverage modules requiring optional extras):**
- `compare/docs.py` 9%, `compare/results.py` 16%
- `benchmarks/gluonts.py` 11%, `benchmarks/m5.py` 40%, `benchmarks/fred.py` 33%

**Action:** Add CI matrix jobs that install each optional extra (`pip install -e .[gluonts]`, `.[fred]`, `.[m5]`, `.[compare]`) and run a thin smoke test. Adjust `pyproject.toml` `[project.optional-dependencies]` if those extras don't already exist.

**Verification:** Each new matrix job lights up in CI; coverage report shows movement on the previously-untouched modules.

#### [ ] NOX — `nox -s {tests,mypy,ruff,docs,security,audit}` reproducible local CI
- **Severity:** Improvement  •  **Effort:** ½–1 day  •  **Phase:** 3  •  **Audit ref:** §11 (codex Option E1)
- **Depends on:** none  •  **Blocks:** none  •  **File group:** noxfile.py

**Action:** Add a `noxfile.py` with sessions for tests (matrix py3.10–3.12), mypy, ruff, docs build, `pip-audit`, and `pre-commit`. Document `pipx install nox && nox` in CONTRIBUTING.md.

**Verification:** `nox --list` shows all sessions; `nox -s tests` exits 0 in a fresh venv.

#### [ ] B1.b — (Optional) Split `gate_signal_verification` into `gate_has_signal` + `gate_leakage_suspicion`
- **Severity:** Improvement  •  **Effort:** 2–3 days incl. deprecation shim  •  **Phase:** 3  •  **Audit ref:** §3 B1.b (alternative)
- **Depends on:** B1, M7.a  •  **Blocks:** none  •  **File group:** src/temporalcv/gates.py

**Action:** Implement two public functions. `gate_has_signal` runs the current permutation-based shuffled-target test and HALTs when signal is detected. `gate_leakage_suspicion` is a composite gate (signal + temporal-boundary violation + theoretical-bound failure) that HALTs only when leakage is genuinely suspect. Mark `gate_signal_verification` as deprecated (emit `DeprecationWarning`); keep it as a thin alias of `gate_has_signal` for one minor release. Document both in a new `docs/model_cards/leakage_detection.md`.

**Verification:**
```bash
.venv/bin/python -c "
from temporalcv.gates import gate_has_signal, gate_leakage_suspicion, gate_signal_verification
import warnings; warnings.simplefilter('error', DeprecationWarning)
try: gate_signal_verification  # should warn
except DeprecationWarning: print('deprecation works')
"
.venv/bin/python -m pytest tests/test_gates.py -q  # → all green incl. new tests
```

### Phase 3 exit gate

All of the following must pass before tagging `1.2.0`:

- [ ] All Phase 1 + Phase 2 exit-gate items still pass.
- [ ] **Max function length < 100 lines** in `gates.py`, `statistical_tests.py`, `conformal.py`, `cv.py`, `persistence.py` (M7.a success criterion). Verify via the AST script in M7.a's verification block.
- [ ] `pytest --doctest-glob="*.md" docs/ README.md -q` exits 0 (DOCTEST in place).
- [ ] `pre-commit run --all-files` from a fresh clone succeeds without locally installed mypy (M10).
- [ ] `nox -s tests mypy ruff docs` exits 0 in a fresh venv (NOX).
- [ ] `docs/methodology/citation_matrix.md` exists and is linked from README + `docs/index.rst`.
- [ ] CHANGELOG entry under `## [1.2.0]` enumerates Phase 3 task IDs and notes any deprecations (e.g., `gate_signal_verification` → `gate_has_signal` if B1.b shipped).
- [ ] Tag `v1.2.0`; PyPI publish succeeds.

---

## 6. "Already resolved — do not re-fix" appendix

These 10 items appeared in prior audits but are **already correct in the current code** at commit `5e61658`. Listed verbatim from audit §10 so a future reader does not waste effort.

| Prior finding | First raised | Where it's now correct |
|---|---|---|
| `WalkForwardCV.get_n_splits()` silent failure on `ValueError` | codex 2025-12-26 | `cv.py:735-786` — `strict=True` default raises; `versionadded` note at L759-761 |
| `WalkForwardCV` lacks `horizon` parameter | codex 2025-12-23, 2026-04-29 | `cv.py:563, 599` — `horizon: int \| None = None`; `total_separation = horizon + extra_gap` |
| `gate_signal_verification` reuses model state across shuffles | codex 2025-12-23-critique | `gates.py:410-411` — "Models are cloned for each shuffle" |
| Shuffled-target gate IID-shuffle vs cited block-permutation theory | codex 2025-12-23-critique, 2026-04-29 | `gates.py:286-304, 340-346` — default `method="permutation"` + `permutation="block"` (Künsch 1989) |
| DM test parameter names (`errors1` vs `errors_1`) | codex 2025-12-23-critique | `statistical_tests.py:611` — signature is `errors_1, errors_2, h` |
| DM test ignores t-distribution under Harvey correction | codex 2025-12-23-critique | `statistical_tests.py:879-886` — `stats.t.cdf(df=n-1)` selected when `harvey_correction=True` |
| Dead-link API docs (`load_fred_series`, `load_m5_sample`, `ComparisonRunner`) | codex 2025-12-23-critique, 2026-01-05 | Names no longer present in `docs/api/{benchmarks,compare}.md` |
| Monash benchmark loader silent truncation | codex 2025-12-23-critique | `benchmarks/monash.py:127-151, 214-238` — `was_truncated`, `official_split`, `original_series_lengths` tracked |
| Version drift (`__init__.py` 0.1.0 vs `pyproject.toml` 1.0.0) | codex-claude-sonnet 2026-01-05 | Both files now `1.0.0` |
| `BellmanConformalPredictor` cited but not implemented | (audit 2026-04-29) | `conformal.py:537` — class exists |

---

## 7. Source traceability

Generated 2026-04-29 from [`docs/audits/claude-audit-2026-04-29.md`](../../audits/claude-audit-2026-04-29.md) at commit `5e61658`. Every task block carries an `Audit ref:` line pointing at the source section.

The audit itself supersedes 11 prior audits in `docs/audits/`. If any task in this plan disagrees with the audit, the **audit is authoritative** — file an issue and revise this plan, do not silently diverge.

## 8. After completion

When all three phases have shipped:

1. Move this file: `git mv docs/plans/active/remediation_2026-04-29.md docs/plans/implemented/remediation_2026-04-29.md`
2. Add a row to `docs/plans/INDEX.md`:
   ```markdown
   | `implemented/remediation_2026-04-29.md` | 2026-04-29 audit remediation | Reference for what shipped in 1.0.1 → 1.2.0 |
   ```
3. CHANGELOG should already enumerate Phase 1/2/3 task IDs from each release; cross-link this plan from the CHANGELOG header for that range.

---

*End of remediation plan.*
