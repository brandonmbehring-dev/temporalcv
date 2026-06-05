"""Executable conformance suite for temporalcv seams.

The contract *is* these checks: a temporal splitter or estimator conforms to the
``Splitter`` / ``CrossFitter`` seam iff it passes ``check_temporal_splitter`` /
``check_temporal_estimator``. They are exported so downstream consumers (e.g. ``dml_ts``)
can gate their own implementations against the same behavioral contract, and are run in
temporalcv CI.

Each check raises ``AssertionError`` with a diagnostic message on the first violated
invariant and returns ``None`` on success — mirroring sklearn's ``check_estimator``
family. See ``STYLE.md`` and ``docs/adr/0001-v2-seams-and-layout.md`` for the rationale,
and ``~/Claude/lever_of_archimedes/patterns/library-design-playbook.md`` ("the
conformance suite is the contract").
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from temporalcv.protocols import CrossFitter, Splitter

__all__ = ["check_temporal_splitter", "check_temporal_estimator"]


def _default_X(n_samples: int, n_features: int = 3) -> np.ndarray:
    """Deterministic design matrix so checks are reproducible without seeding."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n_samples, n_features))


def check_temporal_splitter(
    splitter: Splitter,
    *,
    X: ArrayLike | None = None,
    n_samples: int = 60,
) -> None:
    """Assert ``splitter`` satisfies the temporal-splitter contract.

    Invariants (each raises ``AssertionError`` on violation):

    1. **Structural** — satisfies the :class:`~temporalcv.Splitter` Protocol
       (``split`` + ``get_n_splits`` present).
    2. **Shape/dtype** — ``split(X)`` yields ``(train_idx, test_idx)`` pairs of non-empty
       1-D integer ndarrays with indices in ``[0, n_samples)``.
    3. **Disjointness** — train and test indices do not overlap within a fold.
    4. **No lookahead** — ``max(train_idx) < min(test_idx)`` for every fold (the defining
       property of a *temporal* splitter).
    5. **Determinism** — two iterations of ``split(X)`` produce identical folds.
    6. **Count consistency** — when ``get_n_splits(X)`` returns an int it equals the
       number of folds yielded (the contract permits ``None`` for lazy splitters).

    If ``splitter`` is also a :class:`~temporalcv.CrossFitter`, the out-of-fold
    prediction contract (shape; NaN only on uncovered rows; residual identity) is checked.

    Parameters
    ----------
    splitter : Splitter
        The splitter instance under test.
    X : ArrayLike, optional
        Design matrix to split. If ``None``, a deterministic ``(n_samples, 3)`` matrix
        is generated.
    n_samples : int, default=60
        Number of rows when ``X`` is generated.
    """
    name = type(splitter).__name__
    assert isinstance(splitter, Splitter), (
        f"{name} does not satisfy the Splitter Protocol (needs split + get_n_splits)."
    )

    X_arr = _default_X(n_samples) if X is None else np.asarray(X)
    n = X_arr.shape[0]

    folds = list(splitter.split(X_arr))
    assert len(folds) > 0, f"{name}.split yielded no folds for n_samples={n}."

    for k, fold in enumerate(folds):
        assert isinstance(fold, tuple) and len(fold) == 2, (
            f"{name}.split fold {k} is not a (train_idx, test_idx) pair."
        )
        train_idx, test_idx = fold
        for label, idx in (("train", train_idx), ("test", test_idx)):
            arr = np.asarray(idx)
            assert arr.ndim == 1, f"{name} fold {k} {label}_idx is not 1-D."
            assert np.issubdtype(arr.dtype, np.integer), (
                f"{name} fold {k} {label}_idx is not integer-typed (got {arr.dtype})."
            )
            assert arr.size > 0, f"{name} fold {k} {label}_idx is empty."
            assert arr.min() >= 0 and arr.max() < n, (
                f"{name} fold {k} {label}_idx out of range [0, {n})."
            )
        assert np.intersect1d(train_idx, test_idx).size == 0, (
            f"{name} fold {k}: train and test indices overlap."
        )
        assert int(np.max(train_idx)) < int(np.min(test_idx)), (
            f"{name} fold {k}: lookahead — max(train)={int(np.max(train_idx))} "
            f">= min(test)={int(np.min(test_idx))}."
        )

    # Determinism: a second pass must reproduce the folds exactly.
    folds2 = list(splitter.split(X_arr))
    assert len(folds2) == len(folds), f"{name}.split is non-deterministic (fold count)."
    for k, ((tr1, te1), (tr2, te2)) in enumerate(zip(folds, folds2, strict=True)):
        assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2), (
            f"{name}.split is non-deterministic at fold {k}."
        )

    # get_n_splits consistency (None allowed for lazy splitters per the contract).
    try:
        n_splits = splitter.get_n_splits(X_arr)
    except TypeError:
        n_splits = splitter.get_n_splits()
    if n_splits is not None:
        assert int(n_splits) == len(folds), (
            f"{name}.get_n_splits(X)={n_splits} != folds yielded={len(folds)}."
        )

    if isinstance(splitter, CrossFitter):
        _check_cross_fitter(splitter, X_arr, folds)


def _check_cross_fitter(
    cf: CrossFitter,
    X: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Check the out-of-fold prediction contract of a ``CrossFitter``."""
    from sklearn.linear_model import LinearRegression

    name = type(cf).__name__
    n = X.shape[0]
    rng = np.random.default_rng(0)
    y = X[:, 0] * 1.5 + rng.standard_normal(n) * 0.1

    preds = np.asarray(cf.fit_predict(LinearRegression(), X, y))
    assert preds.shape == (n,), f"{name}.fit_predict shape {preds.shape} != ({n},)."

    covered = np.unique(np.concatenate([te for _, te in folds]))
    uncovered = np.setdiff1d(np.arange(n), covered)
    assert np.all(np.isfinite(preds[covered])), (
        f"{name}.fit_predict left non-finite values on covered rows."
    )
    if uncovered.size:
        assert np.all(np.isnan(preds[uncovered])), (
            f"{name}.fit_predict produced predictions on uncovered rows (must be NaN)."
        )

    resid = np.asarray(cf.fit_predict_residuals(LinearRegression(), X, y))
    assert resid.shape == (n,), f"{name}.fit_predict_residuals shape {resid.shape} != ({n},)."
    assert np.allclose(resid[covered], (y - preds)[covered], atol=1e-8), (
        f"{name}.fit_predict_residuals != y - fit_predict on covered rows."
    )


def check_temporal_estimator(
    estimator: Any,
    *,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    n_samples: int = 60,
) -> None:
    """Assert ``estimator`` satisfies the nuisance-learner contract for cross-fitting.

    A nuisance learner must (1) expose callable ``fit``/``predict``, (2) have
    ``predict(X)`` return a finite 1-D array of length ``n_samples``, and (3) integrate
    with :meth:`~temporalcv.CrossFitCV.fit_predict` — finite out-of-fold predictions on
    every covered row. This is exactly what an estimator must satisfy to sit on the
    ``CrossFitter`` seam (e.g. as ``model_a``/``model_b`` in
    :func:`~temporalcv.cross_fit_residualize`).

    Parameters
    ----------
    estimator : object
        A scikit-learn-style regressor with ``fit(X, y)`` / ``predict(X)``.
    X, y : ArrayLike, optional
        Data to test against; deterministic defaults are generated when omitted.
    n_samples : int, default=60
        Number of rows when data is generated.
    """
    from sklearn.base import clone

    from temporalcv.cv import CrossFitCV

    name = type(estimator).__name__
    assert callable(getattr(estimator, "fit", None)), f"{name} has no callable fit()."
    assert callable(getattr(estimator, "predict", None)), f"{name} has no callable predict()."

    X_arr = _default_X(n_samples) if X is None else np.asarray(X)
    n = X_arr.shape[0]
    if y is None:
        rng = np.random.default_rng(1)
        y_arr = X_arr[:, 0] * 2.0 + rng.standard_normal(n) * 0.1
    else:
        y_arr = np.asarray(y, dtype=float)

    # Direct fit/predict contract.
    try:
        est = clone(estimator)
    except TypeError:
        est = estimator
    est.fit(X_arr, y_arr)
    pred = np.asarray(est.predict(X_arr))
    assert pred.shape == (n,), f"{name}.predict returned shape {pred.shape}, expected ({n},)."
    assert np.all(np.isfinite(pred)), f"{name}.predict returned non-finite values."

    # Integration with the cross-fitting seam.
    cv = CrossFitCV(n_splits=4)
    oof = np.asarray(cv.fit_predict(estimator, X_arr, y_arr))
    assert oof.shape == (n,), f"{name} cross-fit OOF shape {oof.shape} != ({n},)."
    covered = np.unique(np.concatenate([te for _, te in cv.split(X_arr)]))
    assert np.all(np.isfinite(oof[covered])), (
        f"{name} produced non-finite OOF predictions on covered rows."
    )
