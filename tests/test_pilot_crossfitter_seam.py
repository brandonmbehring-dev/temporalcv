"""Tests for the v2.0 CrossFitter-seam pilot.

Covers the vertical slice introduced by the v2.0 seam pilot:
- ``Splitter`` / ``CrossFitter`` static Protocols (structural typing seam).
- ``cross_fit_residualize`` dual-variable out-of-fold residualization.
- The executable conformance suite (``check_temporal_splitter`` /
  ``check_temporal_estimator``), positive and negative.

See ``docs/adr/0001-v2-seams-and-layout.md`` and ``STYLE.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from temporalcv import (
    CrossFitCV,
    CrossFitter,
    Splitter,
    WalkForwardCV,
    check_temporal_estimator,
    check_temporal_splitter,
    cross_fit_residualize,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def linear_dgp() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Linear partially-linear DGP: y = theta*d + g(X) + eps, d = m(X) + v."""
    rng = np.random.default_rng(7)
    n = 400
    theta = 2.0
    X = rng.standard_normal((n, 3))
    d = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.1
    y = theta * d + X[:, 1] + rng.standard_normal(n) * 0.1
    return X, y, d, theta


# =============================================================================
# Static Protocol seam (structural typing)
# =============================================================================


class TestProtocolSeam:
    """The Protocol seam must discriminate capability levels by member presence."""

    def test_crossfitcv_is_splitter_and_crossfitter(self) -> None:
        cf = CrossFitCV(n_splits=3)
        assert isinstance(cf, Splitter)
        assert isinstance(cf, CrossFitter)

    def test_walkforwardcv_is_splitter_not_crossfitter(self) -> None:
        # WalkForwardCV has split/get_n_splits but no fit_predict -> Splitter only.
        wf = WalkForwardCV(n_splits=3)
        assert isinstance(wf, Splitter)
        assert not isinstance(wf, CrossFitter)

    def test_non_splitter_is_not_splitter(self) -> None:
        assert not isinstance(object(), Splitter)


# =============================================================================
# cross_fit_residualize
# =============================================================================


class TestCrossFitResidualize:
    """Dual-variable out-of-fold residualization."""

    def test_shapes_and_shared_nan_mask(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        X, y, d, _ = linear_dgp
        cv = CrossFitCV(n_splits=5)
        y_res, d_res = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)

        assert y_res.shape == (X.shape[0],)
        assert d_res.shape == (X.shape[0],)
        # The defining guarantee: both residual vectors share one NaN mask.
        np.testing.assert_array_equal(np.isnan(y_res), np.isnan(d_res))

    def test_uncovered_rows_are_fold_zero(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        X, y, d, _ = linear_dgp
        cv = CrossFitCV(n_splits=5)
        y_res, _ = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)

        covered = np.unique(np.concatenate([te for _, te in cv.split(X)]))
        uncovered = np.setdiff1d(np.arange(X.shape[0]), covered)
        assert np.all(np.isnan(y_res[uncovered]))
        assert np.all(np.isfinite(y_res[covered]))
        # Forward-only CrossFitCV: the uncovered block is the leading fold 0.
        assert uncovered.min() == 0

    def test_recovers_theta(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        X, y, d, theta = linear_dgp
        cv = CrossFitCV(n_splits=5)
        y_res, d_res = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)
        mask = ~np.isnan(y_res)
        theta_hat = float(np.polyfit(d_res[mask], y_res[mask], 1)[0])
        assert abs(theta_hat - theta) < 0.1

    def test_parity_with_single_target_fit_predict(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        """Joint residualization must equal two single-target fit_predict calls.

        For a deterministic learner the per-fold fits are identical, so
        ``cross_fit_residualize`` is exactly ``A - fit_predict(A)`` and
        ``B - fit_predict(B)`` on covered rows.
        """
        X, y, d, _ = linear_dgp
        cv = CrossFitCV(n_splits=5)
        y_res, d_res = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)

        y_ref = y - cv.fit_predict(LinearRegression(), X, y)
        d_ref = d - cv.fit_predict(LinearRegression(), X, d)
        mask = ~np.isnan(y_res)
        np.testing.assert_allclose(y_res[mask], y_ref[mask], atol=1e-10)
        np.testing.assert_allclose(d_res[mask], d_ref[mask], atol=1e-10)

    def test_accepts_any_splitter_not_just_crossfitcv(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        """Typed against ``Splitter`` — must work with WalkForwardCV too."""
        X, y, d, _ = linear_dgp
        wf = WalkForwardCV(n_splits=5, test_size=20)
        y_res, d_res = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, wf)
        np.testing.assert_array_equal(np.isnan(y_res), np.isnan(d_res))
        assert np.isfinite(y_res).any()

    def test_length_mismatch_raises(self) -> None:
        X = np.zeros((50, 2))
        with pytest.raises(ValueError, match="first-axis length"):
            cross_fit_residualize(
                LinearRegression(),
                LinearRegression(),
                X,
                np.zeros(50),
                np.zeros(49),
                CrossFitCV(n_splits=3),
            )


# =============================================================================
# Golden-parity reference: dml_ts consumer
# =============================================================================


def _dmlts_reference_residualize(
    model_y: object,
    model_t: object,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    cv: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim port of dml_ts's ``_cross_fit_nuisance_time_series`` + residualization.

    This is the golden reference the consumer (``dml_ts``) currently runs
    (``dml_ts/dml/temporal_plr_dml.py``): NaN-filled OOF prediction arrays, per-fold
    clone-fit-predict for both nuisances, uncovered rows left NaN; the DML estimator
    then forms residuals ``Y - Y_hat`` / ``T - T_hat``. Kept dependency-free here so the
    parity contract is enforced in temporalcv CI without importing dml_ts. A throwaway
    spike confirmed this matches the live dml_ts function to 1e-10 on its own splitters.
    """
    n = len(Y)
    y_hat = np.full(n, np.nan)
    t_hat = np.full(n, np.nan)
    for train_idx, test_idx in cv.split(X):  # type: ignore[attr-defined]
        ym = clone(model_y)
        ym.fit(X[train_idx], Y[train_idx])
        y_hat[test_idx] = ym.predict(X[test_idx])
        tm = clone(model_t)
        tm.fit(X[train_idx], T[train_idx])
        t_hat[test_idx] = tm.predict(X[test_idx])
    return Y - y_hat, T - t_hat


class TestDmlTsGoldenParity:
    """``cross_fit_residualize`` must match the dml_ts cross-fitting algorithm exactly.

    This gates the Track-B migration: dml_ts's bespoke ``_cross_fit_nuisance_time_series``
    is exactly ``cross_fit_residualize`` over the same splitter, so it can be retired in
    favor of the upstream seam without changing any estimate.
    """

    @pytest.mark.parametrize("n_splits", [3, 5, 6])
    def test_matches_reference_on_crossfitcv(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float], n_splits: int
    ) -> None:
        X, y, d, _ = linear_dgp
        cv = CrossFitCV(n_splits=n_splits)
        y_ref, d_ref = _dmlts_reference_residualize(
            LinearRegression(), LinearRegression(), X, y, d, cv
        )
        y_new, d_new = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)
        np.testing.assert_array_equal(np.isnan(y_ref), np.isnan(y_new))
        mask = ~np.isnan(y_new)
        np.testing.assert_allclose(y_new[mask], y_ref[mask], atol=1e-10)
        np.testing.assert_allclose(d_new[mask], d_ref[mask], atol=1e-10)

    def test_matches_reference_on_walkforward(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        X, y, d, _ = linear_dgp
        cv = WalkForwardCV(n_splits=5, test_size=20)
        y_ref, d_ref = _dmlts_reference_residualize(
            LinearRegression(), LinearRegression(), X, y, d, cv
        )
        y_new, d_new = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)
        np.testing.assert_array_equal(np.isnan(y_ref), np.isnan(y_new))
        mask = ~np.isnan(y_new)
        np.testing.assert_allclose(y_new[mask], y_ref[mask], atol=1e-10)
        np.testing.assert_allclose(d_new[mask], d_ref[mask], atol=1e-10)


# =============================================================================
# Conformance suite — positive
# =============================================================================


class _LazyNoneSplitter:
    """A valid splitter whose get_n_splits returns None (lazy, count unknown ahead of
    iteration). Exercises the contract that ``get_n_splits`` may be ``None``."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        half = n // 2
        yield np.arange(0, half, dtype=np.intp), np.arange(half, n, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> None:
        return None


class TestConformancePositive:
    """Library-provided splitters/estimators must pass their own contract."""

    def test_crossfitcv_conforms(self) -> None:
        check_temporal_splitter(CrossFitCV(n_splits=5))

    def test_walkforwardcv_conforms(self) -> None:
        check_temporal_splitter(WalkForwardCV(n_splits=5))

    def test_lazy_none_get_n_splits_accepted(self) -> None:
        # the contract permits get_n_splits() -> None for lazy splitters; the suite must
        # accept it (skipping only the count-consistency check), not reject it.
        check_temporal_splitter(_LazyNoneSplitter())

    @pytest.mark.parametrize(
        "estimator",
        [LinearRegression(), RandomForestRegressor(n_estimators=15, random_state=0)],
    )
    def test_estimators_conform(self, estimator: object) -> None:
        check_temporal_estimator(estimator)


# =============================================================================
# Conformance suite — negative (the checker must catch violations)
# =============================================================================


class _LookaheadSplitter:
    """A deliberately broken splitter: test precedes train (lookahead)."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        half = n // 2
        # test BEFORE train -> violates max(train) < min(test)
        yield np.arange(half, n, dtype=np.intp), np.arange(0, half, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1


class _MiscountSplitter:
    """get_n_splits disagrees with the number of folds actually yielded."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        third = n // 3
        yield np.arange(0, third, dtype=np.intp), np.arange(third, 2 * third, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 5  # lies: only 1 fold is produced


class TestConformanceNegative:
    """The checker must reject contract violations with AssertionError."""

    def test_lookahead_splitter_rejected(self) -> None:
        with pytest.raises(AssertionError, match="lookahead"):
            check_temporal_splitter(_LookaheadSplitter())

    def test_miscount_splitter_rejected(self) -> None:
        with pytest.raises(AssertionError, match="get_n_splits"):
            check_temporal_splitter(_MiscountSplitter())

    def test_non_estimator_rejected(self) -> None:
        with pytest.raises(AssertionError, match="fit"):
            check_temporal_estimator(object())


class _OverlapSplitter:
    """train and test indices overlap -> fails disjointness."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        # train [0, 0.6n) and test [0.5n, 0.8n) share [0.5n, 0.6n)
        yield np.arange(0, 3 * n // 5, dtype=np.intp), np.arange(n // 2, 4 * n // 5, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1


class _NonDeterministicSplitter:
    """Yields different (individually valid) folds on each split() call -> fails determinism."""

    def __init__(self) -> None:
        self._calls = 0

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        self._calls += 1
        train_end = n // 2
        test_start = train_end + self._calls  # shifts between calls
        yield np.arange(0, train_end, dtype=np.intp), np.arange(test_start, n, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1


class _EmptyFoldSplitter:
    """Yields an empty test fold -> fails non-emptiness."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        yield np.arange(0, n // 2, dtype=np.intp), np.array([], dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1


class _FloatIndexSplitter:
    """Yields float (non-integer) indices -> fails dtype check."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        yield np.arange(0, n // 2, dtype=np.intp), np.arange(n // 2, n, dtype=float)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1


class _OutOfRangeSplitter:
    """Yields a test index >= n -> fails range check."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        yield np.arange(0, n // 2, dtype=np.intp), np.arange(n // 2, n + 5, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1


class _NaNEstimator:
    """fit/predict present, but predict returns NaN -> fails finite-OOF check."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> _NaNEstimator:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), np.nan)


class _WrongShapeEstimator:
    """predict returns a 2-D array -> fails shape check."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> _WrongShapeEstimator:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((len(X), 2))


class TestConformanceNegativeExtended:
    """Each currently-unguarded splitter/estimator invariant must be rejected (review R3)."""

    def test_overlap_rejected(self) -> None:
        with pytest.raises(AssertionError, match="overlap"):
            check_temporal_splitter(_OverlapSplitter())

    def test_nondeterministic_rejected(self) -> None:
        with pytest.raises(AssertionError, match="non-deterministic"):
            check_temporal_splitter(_NonDeterministicSplitter())

    def test_empty_fold_rejected(self) -> None:
        with pytest.raises(AssertionError, match="empty"):
            check_temporal_splitter(_EmptyFoldSplitter())

    def test_float_index_rejected(self) -> None:
        with pytest.raises(AssertionError, match="integer-typed"):
            check_temporal_splitter(_FloatIndexSplitter())

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(AssertionError, match="out of range"):
            check_temporal_splitter(_OutOfRangeSplitter())

    def test_nan_estimator_rejected(self) -> None:
        with pytest.raises(AssertionError, match="non-finite"):
            check_temporal_estimator(_NaNEstimator())

    def test_wrong_shape_estimator_rejected(self) -> None:
        with pytest.raises(AssertionError, match="shape"):
            check_temporal_estimator(_WrongShapeEstimator())


# =============================================================================
# cross_fit_residualize — edge cases (review R3)
# =============================================================================


class _ZeroFoldSplitter:
    """Yields no folds at all."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return iter(())

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 0


class _GappySplitter:
    """Forward, valid, but leaves a non-contiguous interior region uncovered."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        a, b, c, d = n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5
        yield np.arange(0, a, dtype=np.intp), np.arange(a, b, dtype=np.intp)
        yield np.arange(0, c, dtype=np.intp), np.arange(c, d, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 2


class _NonCloneableModel:
    """Has fit/predict but no get_params, so sklearn.clone raises TypeError (fallback path)."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _NonCloneableModel:
        self.coef_, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(X) @ self.coef_)


class _StatefulNonCloneable:
    """Non-cloneable AND stateful: predict() reveals how many times it was fit.

    Lets the test prove the clone-fallback deep-copies a fresh instance per fold (so each
    sees exactly one fit) rather than reusing one instance (which would leak fit-count,
    i.e. corrupt OOF predictions, across folds).
    """

    def __init__(self) -> None:
        self.n_fits = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _StatefulNonCloneable:
        self.n_fits += 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), float(self.n_fits))


class TestCrossFitResidualizeEdgeCases:
    """Behaviour of cross_fit_residualize on the irregular paths (review R3)."""

    def test_zero_folds_raises(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        # C1 (review R5): zero usable folds is a loud error in the seam, not silent all-NaN
        # (which would yield NaN estimates downstream under FWL-by-formula).
        X, y, d, _ = linear_dgp
        with pytest.raises(ValueError, match="no folds"):
            cross_fit_residualize(
                LinearRegression(), LinearRegression(), X, y, d, _ZeroFoldSplitter()
            )

    def test_noncontiguous_coverage_shares_mask(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        X, y, d, _ = linear_dgp
        cv = _GappySplitter()
        y_res, d_res = cross_fit_residualize(LinearRegression(), LinearRegression(), X, y, d, cv)
        # the shared-mask guarantee must hold even when uncovered rows are non-contiguous
        np.testing.assert_array_equal(np.isnan(y_res), np.isnan(d_res))
        covered = np.unique(np.concatenate([te for _, te in cv.split(X)]))
        np.testing.assert_array_equal(np.flatnonzero(~np.isnan(y_res)), covered)

    def test_noncloneable_model_fallback(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        X, y, d, _ = linear_dgp
        # exercises the `except TypeError: fold_model = model` clone-fallback branch
        y_res, d_res = cross_fit_residualize(
            _NonCloneableModel(), _NonCloneableModel(), X, y, d, CrossFitCV(n_splits=5)
        )
        np.testing.assert_array_equal(np.isnan(y_res), np.isnan(d_res))
        assert np.isfinite(y_res[~np.isnan(y_res)]).all()

    def test_clone_fallback_isolates_fold_state(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        # C2 (review R5): the deepcopy fallback gives each fold a fresh instance, so a
        # stateful non-cloneable learner sees exactly ONE fit per fold. Its predict()
        # returns n_fits, so every covered row must read 1.0 — not 2, 3, 4... that a
        # leaked/reused single instance would produce.
        X, y, d, _ = linear_dgp
        y_res, _ = cross_fit_residualize(
            _StatefulNonCloneable(), _StatefulNonCloneable(), X, y, d, CrossFitCV(n_splits=5)
        )
        covered = ~np.isnan(y_res)
        predicted = y - y_res  # predicted value == n_fits the fold's model has seen
        np.testing.assert_allclose(predicted[covered], 1.0)

    def test_nan_in_target_with_validating_learner_raises(
        self, linear_dgp: tuple[np.ndarray, np.ndarray, np.ndarray, float]
    ) -> None:
        # H1 (review R5): the seam itself does NOT validate finiteness — this raises only
        # because the *learner* (sklearn here) rejects NaN. NaN-tolerant learners would
        # absorb it silently; the test name reflects that scoping.
        X, y, d, _ = linear_dgp
        y_bad = y.copy()
        y_bad[3] = np.nan  # row 3 lands in fold 0 -> a training fold for later folds
        with pytest.raises(ValueError, match="NaN"):
            cross_fit_residualize(
                LinearRegression(), LinearRegression(), X, y_bad, d, CrossFitCV(n_splits=5)
            )


# =============================================================================
# CrossFitter OOF / residual contract — negative (review R5 / G1)
# =============================================================================


class _LeakyCrossFitter:
    """A CrossFitter that writes finite predictions on uncovered rows (leakage)."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        half = n // 2
        yield np.arange(0, half, dtype=np.intp), np.arange(half, n, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1

    def fit_predict(self, model: object, X: object, y: object) -> np.ndarray:
        # BUG: all-finite, including the uncovered leading block [0, half)
        return np.zeros(len(y))  # type: ignore[arg-type]

    def fit_predict_residuals(self, model: object, X: object, y: object) -> np.ndarray:
        return np.asarray(y) - self.fit_predict(model, X, y)


class _InconsistentResidualCrossFitter:
    """fit_predict_residuals disagrees with y - fit_predict."""

    def split(
        self, X: object, y: object = None, groups: object = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)  # type: ignore[arg-type]
        half = n // 2
        yield np.arange(0, half, dtype=np.intp), np.arange(half, n, dtype=np.intp)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return 1

    def fit_predict(self, model: object, X: object, y: object) -> np.ndarray:
        n = len(y)  # type: ignore[arg-type]
        half = n // 2
        out = np.full(n, np.nan)
        out[half:] = 0.0  # covered rows finite, uncovered NaN (correct)
        return out

    def fit_predict_residuals(self, model: object, X: object, y: object) -> np.ndarray:
        return np.full(len(y), 999.0)  # type: ignore[arg-type]  # BUG: != y - fit_predict


class TestCrossFitterContractNegative:
    """check_temporal_splitter must enforce the CrossFitter OOF/residual contract (G1)."""

    def test_leaky_uncovered_predictions_rejected(self) -> None:
        # the NaN-on-uncovered leakage invariant — the one dml_ts most relies on
        assert isinstance(_LeakyCrossFitter(), CrossFitter)
        with pytest.raises(AssertionError, match="uncovered"):
            check_temporal_splitter(_LeakyCrossFitter())

    def test_inconsistent_residuals_rejected(self) -> None:
        assert isinstance(_InconsistentResidualCrossFitter(), CrossFitter)
        with pytest.raises(AssertionError, match="fit_predict_residuals"):
            check_temporal_splitter(_InconsistentResidualCrossFitter())
