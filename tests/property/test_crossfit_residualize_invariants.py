"""Property-based tests for ``cross_fit_residualize`` invariants.

Invariants that must hold for every valid (n_samples, n_splits, data):
1. The two residual vectors share an identical NaN mask.
2. Covered rows are exactly the union of the splitter's test folds (no silent gaps).
3. On covered rows, joint residualization equals single-target ``fit_predict`` (parity).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LinearRegression

from temporalcv.cv import CrossFitCV, cross_fit_residualize


@st.composite
def residualize_params(draw: st.DrawFn) -> tuple[int, int, int]:
    """Valid (n_samples, n_splits, seed) with >= 4 samples per fold."""
    n_splits = draw(st.integers(min_value=2, max_value=8))
    n_samples = draw(st.integers(min_value=n_splits * 4, max_value=300))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return n_samples, n_splits, seed


@given(residualize_params())
@settings(max_examples=60, deadline=None)
def test_shared_mask_covered_and_parity(params: tuple[int, int, int]) -> None:
    n_samples, n_splits, seed = params
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 2))
    A = X[:, 0] + rng.standard_normal(n_samples) * 0.1
    B = X[:, 1] + rng.standard_normal(n_samples) * 0.1
    cv = CrossFitCV(n_splits=n_splits)

    a_res, b_res = cross_fit_residualize(LinearRegression(), LinearRegression(), X, A, B, cv)

    # 1. Identical NaN mask.
    assert np.array_equal(np.isnan(a_res), np.isnan(b_res))

    # 2. Covered rows == union of test folds, exactly.
    folds = list(cv.split(X))
    covered = np.unique(np.concatenate([te for _, te in folds]))
    mask = ~np.isnan(a_res)
    assert np.array_equal(np.flatnonzero(mask), covered)

    # 3. Parity with single-target fit_predict on covered rows (deterministic learner).
    a_ref = A - cv.fit_predict(LinearRegression(), X, A)
    np.testing.assert_allclose(a_res[mask], a_ref[mask], atol=1e-9)
