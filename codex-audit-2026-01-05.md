# codex-audit-2026-01-05.md

## Scope and method
- Reviewed methodology docs (`docs/benchmarks/methodology.md`, `docs/testing_strategy.md`, knowledge base, model cards).
- Audited docstrings and Sphinx API docs for drift and inconsistencies (cv/gates/conformal/statistical tests).
- Parsed notebooks for outdated API usage, execution-breaking code, and pedagogy clarity (no execution).
- Re-checked core code for behavior-vs-doc mismatches.
- Tests not executed in this pass.

## Executive summary
- The `gap` vs `extra_gap` drift now spans code, docstrings, Sphinx docs, model cards, tutorials, notebooks, tests, and scripts. This is a correctness and adoption blocker, not just documentation polish.
- Several notebooks contain execution-breaking code (invalid `horizon=0`, `gap=` usage, duplicate keyword arguments, `.gap` property access), which undermines pedagogy and likely breaks notebook CI runs.
- Documentation for `gate_shuffled_target` and conformal coverage metrics is internally inconsistent (strict semantics, details keys, and `coverage_gap` sign), risking incorrect interpretation of validation results.
- README dependency/CI claims still diverge from actual configuration, creating avoidable user confusion.

## Strengths
- Clear architectural intent and formalized thresholds in `SPECIFICATION.md` remain a strong correctness anchor.
- Validation gates + HALT/WARN/PASS framework are well-defined and match the library’s positioning.
- Notebook curriculum design is thoughtful (PROBLEM → FAILURE → SOLUTION), even though execution drift currently breaks it.

Sources: `SPECIFICATION.md`, `docs/testing_strategy.md`, `docs/benchmarks/methodology.md`, `notebooks/README.md`

## Findings and recommendations (ordered by severity)

### Critical: `gap` vs `extra_gap` drift across docs, docstrings, Sphinx API, tests, and scripts
Impact:
- Users see conflicting guidance depending on where they read; Sphinx auto-doc renders outdated examples directly from docstrings.
- CI/tests and example scripts will fail if they still pass `gap=` or read `cv.gap`.

Evidence:
- `WalkForwardCV` signature only accepts `extra_gap`, but docstrings still show `gap` and `cv.gap` examples. `src/temporalcv/cv.py`
- Tests and scripts still pass `gap=` and access `.gap`. `tests/test_cv.py`, `scripts/benchmark_comparison.py`
- API docs and knowledge pages still document `gap` parameters. `docs/api/gates.md`, `docs/model_cards/walk_forward_cv.md`, `docs/knowledge/leakage_audit_trail.md`, `docs/knowledge/episodes/episode_02_boundary_violations.md`

Recommendation options:
- Option A: Reintroduce `gap` as a deprecated alias for `extra_gap`, plus a computed `gap` property for backwards compatibility.
  - Pros: Immediate relief for users/tests/notebooks; smooth migration.
  - Cons: Longer deprecation tail; continued semantic ambiguity.
- Option B: Enforce `extra_gap` only and systematically update all docs, docstrings, tests, scripts, and notebooks, plus a migration guide.
  - Pros: Clean API and unambiguous semantics.
  - Cons: Large doc + notebook edit burden; hard break.
- Option C: Rename to `min_separation` or `total_gap` and treat `extra_gap` as advanced/optional.
  - Pros: Semantics become explicit.
  - Cons: Another migration wave.

Sources: `src/temporalcv/cv.py`, `tests/test_cv.py`, `scripts/benchmark_comparison.py`, `docs/api/gates.md`, `docs/model_cards/walk_forward_cv.md`, `docs/knowledge/leakage_audit_trail.md`

### Critical: Notebook execution and pedagogy are broken by API drift and syntax errors
Impact:
- Notebook CI execution likely fails on current codebase.
- Pedagogical “WRONG vs RIGHT” sections now error instead of teaching.

Evidence:
- Invalid `horizon=0` usage in multiple notebooks (current code raises `ValueError`). `notebooks/00_time_series_fundamentals.ipynb`, `notebooks/01_why_temporal_cv.ipynb`, `notebooks/02_gap_enforcement.ipynb`, `notebooks/demo.ipynb`
- `gap=` passed to `WalkForwardCV` and `gate_temporal_boundary` in notebooks (parameter no longer exists). `notebooks/01_why_temporal_cv.ipynb`, `notebooks/02_gap_enforcement.ipynb`, `notebooks/08_validation_workflow.ipynb`
- Duplicate keyword arguments (syntax errors) in notebook cells. `notebooks/02_gap_enforcement.ipynb`, `notebooks/08_validation_workflow.ipynb`
- `.gap` property accessed on `WalkForwardCV`, but no such attribute exists. `notebooks/02_gap_enforcement.ipynb`, `notebooks/demo.ipynb`

Recommendation options:
- Option A: Update notebooks to current API (`extra_gap`, `horizon=None` for 1-step) and use `SplitInfo.gap` when demonstrating gap values.
  - Pros: Notebooks execute cleanly; pedagogy remains.
  - Cons: Requires careful refactor of “WRONG” demos.
- Option B: Keep “WRONG” demos but wrap them in `try/except` or move them to markdown cells with expected error outputs.
  - Pros: Preserves teaching narrative without breaking execution.
  - Cons: More verbose notebooks.
- Option C: Add a notebook linter (static scan for `gap=` and duplicate keyword args) to CI before execution.
  - Pros: Prevents recurrence; cheap to run.
  - Cons: Extra CI complexity.

Sources: `notebooks/00_time_series_fundamentals.ipynb`, `notebooks/01_why_temporal_cv.ipynb`, `notebooks/02_gap_enforcement.ipynb`, `notebooks/08_validation_workflow.ipynb`, `notebooks/demo.ipynb`

### High: `gate_shuffled_target` docs conflict with code and reported fields
Impact:
- Users may misinterpret results or copy code that raises AttributeError.
- Documentation undermines trust in the leakage “gold standard.”

Evidence:
- `strict` described as “strict inequality for p-value” in docs, but code uses it to increase `n_shuffles` for resolution. `docs/api/gates.md`, `src/temporalcv/gates.py`
- Model card shows details keys (`shuffled_mae_mean`, `shuffled_mae_std`) that do not exist; code uses `mae_shuffled_avg` and `mae_shuffled_all`. `docs/model_cards/gate_shuffled_target.md`, `src/temporalcv/gates.py`
- Tutorial uses nonexistent attributes `result.real_mae` / `result.shuffled_mae` and compares enum to string. `docs/tutorials/feature_engineering_safety.md`, `src/temporalcv/gates.py`

Recommendation options:
- Option A: Normalize docs to the actual `GateResult` schema (`result.details[...]`) and clarify `strict` semantics.
  - Pros: Fixes broken copy-paste; aligns with code.
  - Cons: Requires updating multiple docs.
- Option B: Add convenience properties (`real_mae`, `shuffled_mae`) to `GateResult` for backward compatibility.
  - Pros: Preserves existing docs with minimal changes.
  - Cons: Adds API surface area.

Sources: `docs/api/gates.md`, `docs/model_cards/gate_shuffled_target.md`, `docs/tutorials/feature_engineering_safety.md`, `src/temporalcv/gates.py`

### High: `coverage_gap` sign is inconsistent across conformal APIs and docs
Impact:
- Same field name means opposite things depending on function, which can invert interpretation of under/over-coverage.

Evidence:
- `evaluate_interval_quality` defines `coverage_gap = coverage - target`. `src/temporalcv/conformal.py`
- `CoverageDiagnostics` uses `coverage_gap = target - overall_coverage`, and docs reflect both definitions in the same file. `src/temporalcv/conformal.py`, `docs/api/conformal.md`

Recommendation options:
- Option A: Standardize `coverage_gap` sign across all APIs (e.g., always empirical - target) and adjust warning logic accordingly.
  - Pros: Consistent semantics; less confusion.
  - Cons: Requires migration in downstream uses.
- Option B: Rename one of the fields (e.g., `undercoverage` vs `coverage_error`) to make sign explicit.
  - Pros: Clearer semantics without breaking both.
  - Cons: More verbose API.

Sources: `src/temporalcv/conformal.py`, `docs/api/conformal.md`

### Medium: Testing strategy and benchmark methodology drift from repository reality
Impact:
- Contributors may chase non-existent tests or misinterpret validation scope.

Evidence:
- Testing strategy references `tests/integration/test_e2e_*.py`, but no such tests exist. `docs/testing_strategy.md`, `tests/`
- Testing strategy claims Monash coverage, but benchmark scripts only reference M4/M5. `docs/testing_strategy.md`, `scripts/run_benchmark.py`
- Benchmark methodology instructs `pip install statsforecast`, but project extras use `temporalcv[compare]`. `docs/benchmarks/methodology.md`, `pyproject.toml`

Recommendation options:
- Option A: Update docs to reflect current test and benchmark scope.
  - Pros: Lowest effort; accurate.
  - Cons: Doesn’t expand actual coverage.
- Option B: Implement the missing e2e tests and Monash benchmark pathway.
  - Pros: Improves rigor.
  - Cons: Higher engineering cost.

Sources: `docs/testing_strategy.md`, `scripts/run_benchmark.py`, `docs/benchmarks/methodology.md`, `pyproject.toml`

### Medium: Dependency and CI claims do not match actual configuration
Impact:
- Users are told to expect pandas/core deps and OS coverage that aren’t in CI or pyproject.

Evidence:
- README lists pandas as core dependency and higher minimum versions than pyproject, and claims macOS + Python 3.9 are tested. `README.md`
- pyproject uses lower minimums and includes `statsmodels` in core deps. `pyproject.toml`
- CI matrix excludes macOS and Python 3.9. `.github/workflows/ci.yml`

Recommendation options:
- Option A: Update README to match pyproject and CI.
- Option B: Expand CI to match README and align dependencies.

Sources: `README.md`, `pyproject.toml`, `.github/workflows/ci.yml`

### Medium: Project contract drift in `CLAUDE.md`
Impact:
- Versioning and CLI exit-code contract are stale and may mislead contributors.

Evidence:
- `CLAUDE.md` still references version 0.1.0 and a CLI that does not exist. `CLAUDE.md`
- Current package version is 1.0.0-rc1. `pyproject.toml`, `src/temporalcv/__init__.py`

Recommendation options:
- Option A: Update `CLAUDE.md` to current version/structure, or remove the CLI section.
- Option B: Move contributor contract into docs and keep `CLAUDE.md` minimal.

Sources: `CLAUDE.md`, `pyproject.toml`, `src/temporalcv/__init__.py`

### Medium: Documentation duplication is creating drift
Impact:
- README, `docs/index.md`, and `docs/index.rst` disagree on examples and defaults.

Evidence:
- `docs/index.md` uses `n_shuffles=5`, while README and `docs/index.rst` recommend 100. `docs/index.md`, `docs/index.rst`, `README.md`

Recommendation options:
- Option A: Single-source core examples via MyST include.
- Option B: Checklist for synchronized updates.

Sources: `docs/index.md`, `docs/index.rst`, `README.md`

### Low: Documentation verification claims are stale
Impact:
- `docs/ASSESSMENT_CHECKLIST.md` claims no `gap=` usage but repo still has many.

Evidence:
- Checklist claim vs actual remaining `gap=` in docs/tests/notebooks. `docs/ASSESSMENT_CHECKLIST.md`, `docs/knowledge/episodes/episode_02_boundary_violations.md`, `tests/test_cv.py`, `notebooks/`

Recommendation options:
- Option A: Update checklist and link to live search or CI check.
- Option B: Add a CI lint step for deprecated API usage.

Sources: `docs/ASSESSMENT_CHECKLIST.md`, `tests/test_cv.py`, `notebooks/`

## Overlooked or underemphasized areas
- A migration guide for `gap` → `extra_gap` (including examples and notebook updates) would reduce user friction.
- Add a lightweight notebook static lint (deprecated params, duplicate keywords) before execution to prevent CI churn.
- Resolve `coverage_gap` semantics before 1.0.0 final to avoid breaking user analytics.
- Clarify Julia/Python API parity (Julia still uses `gap`). `julia/src/`

Sources: `julia/src/`, `docs/knowledge/leakage_audit_trail.md`

## Plan additions (suggested)
- Add a “gap migration” task: update docstrings, Sphinx API, model cards, tutorials, knowledge episodes, notebooks, tests, and scripts in one sweep.
- Add a “notebook stabilization” task: fix `horizon=0`, remove duplicate kwargs, update `gate_temporal_boundary` signatures, and add try/except for intentional failure demos.
- Add a “conformal semantics” task: unify `coverage_gap` sign (or rename fields) and update docs accordingly.
- Add a “gate_shuffled_target docs sync” task: align strict semantics and details keys across docs/model cards/tutorials.
- Add a “testing strategy audit” task: reconcile claimed layers with actual test paths or implement missing tests.

## Open questions for you
1) Do you want strict backward compatibility for `gap` (deprecated alias), or a clean break with a full migration?
2) Should notebooks prioritize always-running execution (wrap “WRONG” examples), or keep failing demos with narrative explanation?
3) Which `coverage_gap` convention do you want to standardize on (empirical - target or target - empirical)?
4) Should `GateResult` gain convenience properties (`real_mae`, `shuffled_mae`) or should docs switch to `details[...]` only?
5) Do you want README to match CI, or CI to match README (macOS + py3.9)?

