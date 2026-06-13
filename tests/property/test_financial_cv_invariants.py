"""Property-based tests for financial CV module.

Tests invariants of PurgedKFold, CombinatorialPurgedCV, and PurgedWalkForward.
"""

from math import ceil, comb

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from temporalcv.cv_financial import (
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedWalkForward,
    compute_label_overlap,
    estimate_purge_gap,
)


# Custom strategies
@st.composite
def valid_purged_kfold_params(draw: st.DrawFn) -> dict:
    """Generate valid parameters for PurgedKFold."""
    n_samples = draw(st.integers(min_value=50, max_value=500))
    n_splits = draw(st.integers(min_value=2, max_value=min(10, n_samples // 5)))
    purge_gap = draw(st.integers(min_value=0, max_value=10))
    embargo_pct = draw(st.floats(min_value=0.0, max_value=0.1))

    # Ensure there's enough data for splits
    min_fold_size = n_samples // n_splits
    assume(min_fold_size > purge_gap)

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "purge_gap": purge_gap,
        "embargo_pct": embargo_pct,
    }


@st.composite
def valid_cpcv_params(draw: st.DrawFn) -> dict:
    """Generate valid parameters for CombinatorialPurgedCV."""
    n_samples = draw(st.integers(min_value=100, max_value=500))
    n_splits = draw(st.integers(min_value=3, max_value=6))
    n_test_splits = draw(st.integers(min_value=1, max_value=n_splits - 1))
    purge_gap = draw(st.integers(min_value=0, max_value=5))

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "purge_gap": purge_gap,
    }


@st.composite
def valid_walk_forward_params(draw: st.DrawFn) -> dict:
    """Generate provisioned parameters for PurgedWalkForward.

    Fold 0 has the smallest train window, ending at
    ``n_samples - n_splits * test_size - purge_gap``. Since #35 the fixed
    window is never truncated: fold 0 must fit the whole ``train_size``
    (``train_end - train_size >= 0``) or the splitter raises. Since #38 the
    embargo is one-sided (after each test run); in this forward-only geometry
    the train window precedes the test window, so the embargo removes nothing
    and never shaves the right edge.
    """
    n_samples = draw(st.integers(min_value=200, max_value=1000))
    n_splits = draw(st.integers(min_value=2, max_value=10))
    test_size = draw(st.integers(min_value=10, max_value=50))
    train_size = draw(st.integers(min_value=50, max_value=200))
    purge_gap = draw(st.integers(min_value=0, max_value=10))

    assume(n_samples - n_splits * test_size - purge_gap - train_size >= 0)

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "train_size": train_size,
        "test_size": test_size,
        "purge_gap": purge_gap,
    }


class TestPurgedKFoldInvariants:
    """Property tests for PurgedKFold."""

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_train_test_never_overlap(self, params: dict) -> None:
        """Train and test indices must never overlap."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
            embargo_pct=params["embargo_pct"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0, "Train and test overlap!"

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_purge_gap_respected(self, params: dict) -> None:
        """Training samples must respect purge gap from test samples."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
            embargo_pct=0.0,  # Test purge without embargo
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            # Find minimum distance from any train sample to any test sample
            train_arr = np.array(list(train_idx))
            test_arr = np.array(list(test_idx))

            # For each test index, check distance to nearest train index
            for t_idx in test_arr:
                if len(train_arr) > 0:
                    distances = np.abs(train_arr - t_idx)
                    min_dist = np.min(distances)
                    # Purging removes distances 0..purge_gap inclusive, so
                    # surviving train rows must be strictly beyond purge_gap.
                    assert min_dist > params["purge_gap"], (
                        f"Purge gap violated: min_dist={min_dist}, purge_gap={params['purge_gap']}"
                    )

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_correct_number_of_splits(self, params: dict) -> None:
        """Should produce exactly n_splits folds."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        splits = list(cv.split(X))
        assert len(splits) == params["n_splits"]

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_all_indices_valid(self, params: dict) -> None:
        """All indices must be in valid range [0, n_samples)."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            assert all(0 <= i < params["n_samples"] for i in train_idx)
            assert all(0 <= i < params["n_samples"] for i in test_idx)


class TestCombinatorialPurgedCVInvariants:
    """Property tests for CombinatorialPurgedCV."""

    @given(params=valid_cpcv_params())
    @settings(max_examples=50)
    def test_correct_number_of_paths(self, params: dict) -> None:
        """Should produce C(n_splits, n_test_splits) paths."""
        cv = CombinatorialPurgedCV(
            n_splits=params["n_splits"],
            n_test_splits=params["n_test_splits"],
            purge_gap=params["purge_gap"],
        )

        expected_paths = comb(params["n_splits"], params["n_test_splits"])
        assert cv.get_n_splits() == expected_paths

    @given(params=valid_cpcv_params())
    @settings(max_examples=50)
    def test_train_test_never_overlap(self, params: dict) -> None:
        """Train and test indices must never overlap."""
        cv = CombinatorialPurgedCV(
            n_splits=params["n_splits"],
            n_test_splits=params["n_test_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    @given(params=valid_cpcv_params())
    @settings(max_examples=50)
    def test_all_indices_valid(self, params: dict) -> None:
        """All indices must be in valid range."""
        cv = CombinatorialPurgedCV(
            n_splits=params["n_splits"],
            n_test_splits=params["n_test_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            assert all(0 <= i < params["n_samples"] for i in train_idx)
            assert all(0 <= i < params["n_samples"] for i in test_idx)


class TestPurgedWalkForwardInvariants:
    """Property tests for PurgedWalkForward."""

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_train_precedes_test(self, params: dict) -> None:
        """All train indices must precede test indices."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert np.max(train_idx) < np.min(test_idx), (
                    "Train indices must precede test indices!"
                )

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_purge_gap_creates_separation(self, params: dict) -> None:
        """Purge gap should create separation between train and test."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                gap = np.min(test_idx) - np.max(train_idx) - 1
                assert gap >= params["purge_gap"], (
                    f"Gap {gap} less than purge_gap {params['purge_gap']}"
                )

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_train_test_never_overlap(self, params: dict) -> None:
        """Train and test must never overlap."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_yields_exactly_n_splits(self, params: dict) -> None:
        """A provisioned config yields exactly get_n_splits folds (#32)."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        assert len(list(cv.split(X))) == cv.get_n_splits()

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_fixed_window_never_truncated(self, params: dict) -> None:
        """A provisioned fixed window is exactly train_size, never clipped (#35).

        purge_gap >= n_embargo is not guaranteed, so the embargo may shave
        ``max(0, n_embargo - purge_gap)`` rows off the window's right edge;
        the geometric window itself must never shrink below that.
        """
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        n_embargo = int(0.01 * params["n_samples"])
        max_embargo_bite = max(0, n_embargo - params["purge_gap"])
        for train_idx, _ in cv.split(X):
            assert len(train_idx) >= params["train_size"] - max_embargo_bite
            assert len(train_idx) <= params["train_size"]


class TestLabelOverlapInvariants:
    """Property tests for compute_label_overlap."""

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_overlap_matrix_symmetric(self, n_samples: int, horizon: int) -> None:
        """Overlap matrix must be symmetric."""
        overlap = compute_label_overlap(n_samples=n_samples, horizon=horizon)
        np.testing.assert_array_equal(overlap, overlap.T)

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_overlap_matrix_diagonal_true(self, n_samples: int, horizon: int) -> None:
        """Diagonal must always be True (self-overlap)."""
        overlap = compute_label_overlap(n_samples=n_samples, horizon=horizon)
        assert np.all(np.diag(overlap))

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_overlap_matrix_shape(self, n_samples: int, horizon: int) -> None:
        """Overlap matrix must be (n_samples, n_samples)."""
        overlap = compute_label_overlap(n_samples=n_samples, horizon=horizon)
        assert overlap.shape == (n_samples, n_samples)


class TestEstimatePurgeGapInvariants:
    """Property tests for estimate_purge_gap."""

    @given(
        horizon=st.integers(min_value=1, max_value=100),
        decay_factor=st.floats(min_value=0.1, max_value=3.0),
    )
    @settings(max_examples=100)
    def test_purge_gap_non_negative(self, horizon: int, decay_factor: float) -> None:
        """Purge gap must always be non-negative."""
        gap = estimate_purge_gap(horizon=horizon, decay_factor=decay_factor)
        assert gap >= 0

    @given(
        horizon=st.integers(min_value=1, max_value=100),
        decay_factor=st.floats(min_value=0.1, max_value=3.0),
    )
    @settings(max_examples=100)
    def test_purge_gap_integer(self, horizon: int, decay_factor: float) -> None:
        """Purge gap must be an integer."""
        gap = estimate_purge_gap(horizon=horizon, decay_factor=decay_factor)
        assert isinstance(gap, int)

    @given(horizon=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_default_decay_returns_horizon(self, horizon: int) -> None:
        """Default decay factor of 1.0 should return horizon."""
        gap = estimate_purge_gap(horizon=horizon, decay_factor=1.0)
        assert gap == horizon


def _embargo_after_each_run_respected(
    train_idx: np.ndarray, test_idx: np.ndarray, n_samples: int, embargo_pct: float
) -> bool:
    """No train index falls within ceil(embargo_pct*n) after any contiguous test run.

    The De Prado one-sided per-run embargo property (#38). The buggy pre-#38
    code used a single global ``test_max``, so the interior boundaries of a
    multi-block test set (CombinatorialPurgedCV, shuffled PurgedKFold) leaked
    into the training set.
    """
    n_embargo = ceil(embargo_pct * n_samples)
    if n_embargo == 0 or len(test_idx) == 0:
        return True
    sorted_test = np.unique(test_idx)
    run_ends = sorted_test[np.append(np.diff(sorted_test) != 1, True)]
    train_set = {int(i) for i in train_idx}
    for end in run_ends:
        for i in range(int(end) + 1, min(int(end) + 1 + n_embargo, n_samples)):
            if i in train_set:
                return False
    return True


class TestEmbargoInvariants:
    """The one-sided per-run embargo holds for every fold (#38)."""

    @given(
        n_samples=st.integers(min_value=80, max_value=400),
        n_splits=st.integers(min_value=3, max_value=6),
        n_test_splits=st.integers(min_value=1, max_value=3),
        embargo_pct=st.floats(min_value=0.0, max_value=0.15),
        purge_gap=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=150)
    def test_cpcv_embargo_after_each_run(
        self,
        n_samples: int,
        n_splits: int,
        n_test_splits: int,
        embargo_pct: float,
        purge_gap: int,
    ) -> None:
        """Multi-block test sets embargo every interior boundary, not just the global max."""
        assume(n_test_splits < n_splits)
        cv = CombinatorialPurgedCV(
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct,
        )
        try:
            splits = list(cv.split(np.zeros((n_samples, 1))))
        except ValueError:
            assume(False)  # under-provisioned config raised — not an embargo case
            return
        for train_idx, test_idx in splits:
            assert _embargo_after_each_run_respected(train_idx, test_idx, n_samples, embargo_pct)

    @given(
        n_samples=st.integers(min_value=60, max_value=400),
        n_splits=st.integers(min_value=2, max_value=8),
        embargo_pct=st.floats(min_value=0.0, max_value=0.1),
        purge_gap=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=150)
    def test_purged_kfold_embargo_after_each_run(
        self, n_samples: int, n_splits: int, embargo_pct: float, purge_gap: int
    ) -> None:
        """Unshuffled PurgedKFold embargoes after each contiguous test fold."""
        assume(n_samples // n_splits > purge_gap)
        cv = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap, embargo_pct=embargo_pct)
        try:
            splits = list(cv.split(np.zeros((n_samples, 1))))
        except ValueError:
            assume(False)
            return
        for train_idx, test_idx in splits:
            assert _embargo_after_each_run_respected(train_idx, test_idx, n_samples, embargo_pct)

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_walk_forward_embargo_is_noop(self, params: dict) -> None:
        """Forward-only geometry: the one-sided embargo removes nothing (#38)."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
            embargo_pct=0.1,  # large embargo, still a no-op since train precedes test
        )
        for detailed in cv.split_detailed(np.zeros((params["n_samples"], 1))):
            assert detailed.n_embargoed == 0
