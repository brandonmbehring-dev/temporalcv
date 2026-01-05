# Critique: Declarative Dancing Quiche (Documentation & Release Plan)
**Date:** 2026-01-05
**Auditor:** Gemini (CLI Agent)

## Executive Summary

The plan `declarative-dancing-quiche.md` presents a robust strategy for polishing documentation and infrastructure for a v1.0 release. However, it **critically overlooks high-severity methodological flaws** identified in `codex-methodology-review.md`. 

Proceeding with this plan without addressing the core correctness issues (broken gate defaults, inconsistent gap semantics, valid-but-wrong in-sample error calculations) would result in a "polished but incorrect" library. 

**Recommendation:** The plan must be restructured. "Phase 0" must be the resolution of the High and Medium severity findings from the Codex audit. The documentation efforts should follow *after* the API and logic are stabilized.

---

## 1. Critical Methodological Gaps (The "Elephant in the Room")

The plan rates the project as "8.6/10 (Excellent)" and "Production-ready", but the internal audit (`codex-methodology-review.md`) contradicts this.

### A. Broken Defenses (High Severity)
- **The Issue:** `codex-methodology-review.md` notes that `gate_shuffled_target` defaults (permutation mode with low `n_shuffles`) make it mathematically impossible to achieve significance at $\alpha=0.05$.
- **Impact:** The library's core promise—"Definitive Leakage Detection"—is currently failing in default configurations. Documenting this behavior without fixing it cements a flaw.
- **Action:** Fix the defaults (raise `n_shuffles`, enforce `strict=True` by default, or switch to `effect_size`) before polishing the docs.

### B. Inconsistent Semantics
- **The Issue:** Gap/Horizon definitions disagree across code (`src/temporalcv/gates.py` vs `src/temporalcv/cv.py`), tests, and documentation.
- **Impact:** Users following the docs might pass validation gates while still leaking data in the CV splitter.
- **Action:** Standardize the definition (e.g., `total_gap = horizon + safety_gap`) across the codebase immediately.

### C. Version Mismatch
- **The Issue:**
  - `src/temporalcv/__init__.py`: `__version__ = "0.1.0"`
  - `pyproject.toml`: `version = "1.0.0-rc1"`
  - `CURRENT_WORK.md`: `v1.0.0-rc1`
- **Impact:** Pip installs will report a different version than the runtime package.
- **Action:** Unify versions immediately.

---

## 2. Benchmarking Strategy: Reinvention vs. Refactoring

The plan proposes creating `benchmarks/performance/run_suite.py` from scratch.

- **Observation:** `scripts/run_benchmark.py` (M4/M5 runner) and `scripts/benchmark_comparison.py` (Split speed & feature check) already exist and appear robust.
- **Critique:** Writing a new runner ignores existing tools.
- **Recommendation:**
  1. Move `scripts/run_benchmark.py` to `benchmarks/run_m4.py`.
  2. Move `scripts/benchmark_comparison.py` to `benchmarks/run_comparison.py`.
  3. Create a lightweight driver script that calls these existing robust scripts rather than rewriting them.

---

## 3. Documentation Audit

The plan correctly identifies missing RST files, but misses the opportunity to document the *methodology* changes required by the audit.

- **Pros:**
  - Excellent identification of missing `api_reference/*.rst` files.
  - Good focus on "Model Cards" and "Knowledge Tiers".
  - Practical addition of "Optional Dependencies" to README.
- **Cons:**
  - `notebooks/05_shuffled_target_gate.ipynb` (read during audit) makes claims about leakage detection that the code might not currently support if defaults are used.
  - Documentation plan does not include updating the tutorials to reflect the "Fix" items from `codex-methodology-review.md` (e.g., changing examples to use out-of-sample errors).

---

## 4. Revised Plan Proposal

I recommend inserting a **Phase 0** and **Phase 1** before the documentation work.

### Phase 0: Methodological Corrections (Urgent)
1.  **Fix Gate Defaults:** Update `gate_shuffled_target` to use safe defaults (e.g., `strict=True` or `n_shuffles=100`).
2.  **Unify Gap Semantics:** Decide on "Total Gap" vs "Extra Gap" and refactor `gates.py` and `cv.py` to match.
3.  **Fix Versioning:** Update `__init__.py` to `1.0.0-rc1`.

### Phase 1: Existing Script Refactoring
1.  Move `scripts/*.py` to `benchmarks/` to clean up the root.
2.  Ensure `run_benchmark.py` works with the new package structure.

### Phase 2: The Original Plan (Documentation)
1.  Execute the RST creation and README updates as planned.
2.  **Crucially:** Update tutorials to match the semantic fixes from Phase 0.

---

## 5. Pros & Cons of Current Plan

| Feature | Current Plan | Critique |
| :--- | :--- | :--- |
| **Focus** | Documentation Polish | Misses core logic fixes |
| **Benchmarks** | New implementation | Should leverage existing scripts |
| **Infrastructure** | Docker + Conda | Good, but secondary to correctness |
| **Feasibility** | High (12-14h) | High, but risks checking in "wrong" math |

## Conclusion

The project is **not** ready for a documentation-only sprint. The code must be aligned with the scientific methodology first. If we release v1.0 with known methodological flaws (even if documented beautifully), we undermine the library's primary value proposition of "rigorous validation."

**Next Step:** I can help you execute **Phase 0** (Fixing the code/methodology) followed by **Phase 1 & 2** (Refactoring and Documentation).
