# Gemini Audit Report: temporalcv (v1.0.0-rc1)
**Date:** 2026-01-05
**Auditor:** Gemini (CLI Agent)
**Status:** **CRITICAL METHODOLOGICAL REVIEW**

## 1. Executive Summary

This audit reveals a **foundational flaw** in the library's primary unique selling point (`gate_signal_verification`) and identifies opportunities to improve code structure and pedagogy. While the infrastructure and educational content are excellent, the core "Leakage Gate" methodology is currently inverted, rejecting valid models as leaky.

**Scorecard:**
- **Code Quality:** 8/10 (Solid, but some redundancy)
- **Pedagogy:** 9/10 (Excellent notebooks, dense README)
- **Methodology:** 4/10 (Critical error in Shuffled Target logic)

---

## 2. Critical Methodological Findings

### 2.1 The "Shuffled Target" Paradox [P0]
*   **The Claim:** "If your model beats a shuffled target, features encode target position (leakage)."
*   **The Reality:** If your model beats a shuffled target, features contain **information** about the target. This information could be leakage (bad) OR valid predictive signal (good).
*   **Proof:** A valid AR(1) model ($X_t = y_{t-1}$) will beat a shuffled target significantly because the shuffled target destroys the valid autocorrelation. The gate interprets this significance as "Leakage" and HALTS.
*   **Options for Resolution:**
    *   **Option A (Pivot):** Rename to `verify_model_signal`. PASS if model beats shuffle (verifies signal), WARN if it doesn't (model is noise). **(Recommended)**
    *   **Option B (Deprecate):** Remove the gate entirely if the intention was solely to catch technical leakage (e.g. `X[t] == y[t]`), which `WalkForwardCV` largely prevents anyway.
    *   **Option C (Feature Shuffle):** Shuffle *features* column-wise to break covariance but keep target structure. This tests feature importance, not leakage.

### 2.2 The 3-Class PT Test Variance [T3]
*   **Finding:** The 3-class Pesaran-Timmermann test uses an ad-hoc variance formula (`var * 4`) that is not academically validated.
*   **Risk:** Users may treat this as a rigorous [T1] test.
*   **Action:** Ensure the warning is prominent in runtime logs, not just documentation.

---

## 3. Code Structure & Best Practices

### 3.1 DRY Violations in `gates.py`
*   **Observation:** Both `gate_signal_verification` and `gate_synthetic_ar1` implement their own Walk-Forward CV loop with `sklearn.base.clone`.
*   **Risk:** If you change how CV is handled (e.g. error handling), you must update it in two places.
*   **Fix:** Extract `_compute_cv_metrics(model, X, y, cv, metric_func)` into a private helper or `utils.py`.

### 3.2 Metrics Duplication
*   **Observation:** `gates.py` manually calculates MAE (`np.mean(np.abs(errors))`) instead of importing `compute_mae` from `temporalcv.metrics`.
*   **Fix:** Import metrics from the core module to ensure consistency (e.g., handling of NaNs or epsilon).

### 3.3 Input Validation Boilerplate
*   **Observation:** Every function repeats `X = np.asarray(X)`, `if len(X) != len(y)`, `if np.any(np.isnan(X))`.
*   **Fix:** Use the `_validate_inputs` helper found in `metrics/core.py` across the entire library.

---

## 4. Pedagogical Audit & Recommendations

### 4.1 README Cognitive Overload
*   **Problem:** The README buries the lead. Users see architecture diagrams before code.
*   **Option A (Status Quo):** Architecture first. **Pros:** Sets mental model. **Cons:** High friction.
*   **Option B (Action-First):** "Hello World" snippet at top. **Pros:** Immediate utility. **Cons:** May encourage copy-paste.
*   **Recommendation:** **Option B**. Move the 5-line `gate_signal_verification` (renamed) + `WalkForwardCV` example to the very top.

### 4.2 Visualizing "Gap" and "Embargo"
*   **Problem:** Financial CV concepts are abstract and prone to off-by-one errors.
*   **Recommendation:** Add ASCII art to docstrings in `cv.py` and `cv_financial.py`.
    ```text
    # [Train ...............] [Purge] [Test Label Window] [Embargo] [Train ...]
    ```

### 4.3 "Gate" vs "Guardrail" Confusion
*   **Observation:** Two modules with similar names.
*   **Clarification:** `Gates` are low-level primitives; `Guardrails` are high-level workflows.
*   **Action:** Add a "Concepts" page to the docs explicitly mapping this relationship.

---

## 5. Final Action Plan (Road to v1.0)

1.  **Methodological Pivot [P0]:**
    *   Rename `gate_signal_verification` -> `verify_model_signal`.
    *   Update logic: PASS if p < 0.05 (Signal found).
2.  **Refactoring [P1]:**
    *   Centralize CV loops in `gates.py`.
    *   Use `_validate_inputs` everywhere.
3.  **Documentation Polish [P1]:**
    *   Apply ASCII visualizations.
    *   Refactor README.
    *   Sync `CLAUDE.md` version to `1.0.0-rc1`.
4.  **Infrastructure [P2]:**
    *   Move `scripts/` to `benchmarks/`.

**Conclusion:** The library is code-complete but requires a semantic pivot on its core gate to avoid being misleading. Once `gate_signal_verification` is redefined as a signal check, the library is ready for release.
