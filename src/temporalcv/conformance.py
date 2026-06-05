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

import inspect
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from temporalcv.protocols import CrossFitter, Splitter, SupportsBootstrap, SupportsForecast

__all__ = [
    "check_temporal_splitter",
    "check_temporal_estimator",
    "check_bootstrap_strategy",
    "check_forecast_adapter",
]


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
    # Dispatch on the signature rather than catching TypeError, so a genuine TypeError
    # raised *inside* a get_n_splits(X) body propagates instead of being masked.
    accepts_x = any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in inspect.signature(splitter.get_n_splits).parameters.values()
    )
    n_splits = splitter.get_n_splits(X_arr) if accepts_x else splitter.get_n_splits()
    if n_splits is not None:
        assert int(n_splits) == len(folds), (
            f"{name}.get_n_splits(X)={n_splits} != folds yielded={len(folds)}."
        )

    if isinstance(splitter, CrossFitter):
        _check_cross_fitter(splitter, X_arr, folds)

    # Capabilities-as-tags cross-check (#14): if the splitter declares temporal_tags(), each
    # declaration must agree with observed behavior — a tag cannot silently drift from reality.
    tags_method = getattr(splitter, "temporal_tags", None)
    if callable(tags_method):
        tags = tags_method()
        assert tags.forward_only, (
            f"{name}.temporal_tags declares forward_only=False, but every fold satisfied the "
            f"no-lookahead invariant."
        )
        assert tags.deterministic, (
            f"{name}.temporal_tags declares deterministic=False, but split(X) was reproducible."
        )
        assert tags.produces_oof == isinstance(splitter, CrossFitter), (
            f"{name}.temporal_tags declares produces_oof={tags.produces_oof}, but "
            f"isinstance(splitter, CrossFitter)={isinstance(splitter, CrossFitter)}."
        )
        assert not tags.requires_groups, (
            f"{name}.temporal_tags declares requires_groups=True, but split(X) produced folds "
            f"without a groups argument."
        )


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


def check_bootstrap_strategy(
    strategy: SupportsBootstrap,
    *,
    X: ArrayLike | None = None,
    y: ArrayLike | None = None,
    n_samples: int = 40,
    n_boot: int = 5,
) -> None:
    """Assert ``strategy`` satisfies the bootstrap-strategy contract for
    :class:`~temporalcv.TimeSeriesBagger`.

    Invariants (each raises ``AssertionError`` on violation):

    1. **Structural** — satisfies the :class:`~temporalcv.SupportsBootstrap` Protocol
       (``generate_samples`` + ``transform_for_predict`` present).
    2. **Sample shape** — ``generate_samples`` returns exactly ``n_boot`` ``(X_boot, y_boot)``
       pairs; within each pair ``len(X_boot) == len(y_boot)`` and ``X_boot`` is 2-D with at
       least one feature (feature-subsetting strategies may *reduce* the feature count).
    3. **Determinism** — two calls under freshly-seeded *identical* generators produce identical
       samples (the strategy must draw all randomness from the supplied ``rng``, never global
       state).
    4. **transform_for_predict** — returns an ndarray whose row count matches its input (the
       default is identity; feature-subsetting strategies may change the column count).

    Parameters
    ----------
    strategy : SupportsBootstrap
        The resampling strategy under test.
    X, y : ArrayLike, optional
        Data to resample; deterministic defaults are generated when omitted.
    n_samples : int, default=40
        Number of rows when ``X`` is generated.
    n_boot : int, default=5
        Number of bootstrap samples requested from ``generate_samples``.
    """
    name = type(strategy).__name__
    assert isinstance(strategy, SupportsBootstrap), (
        f"{name} does not satisfy the SupportsBootstrap Protocol "
        f"(needs generate_samples + transform_for_predict)."
    )

    X_arr = _default_X(n_samples) if X is None else np.asarray(X)
    n = X_arr.shape[0]
    if y is None:
        y_arr = np.asarray(np.random.default_rng(2).standard_normal(n), dtype=float)
    else:
        y_arr = np.asarray(y, dtype=float)

    samples = strategy.generate_samples(X_arr, y_arr, n_boot, np.random.default_rng(123))
    assert len(samples) == n_boot, (
        f"{name}.generate_samples returned {len(samples)} samples, expected n_boot={n_boot}."
    )
    for k, pair in enumerate(samples):
        assert isinstance(pair, tuple) and len(pair) == 2, (
            f"{name}.generate_samples item {k} is not an (X_boot, y_boot) pair."
        )
        x_boot, y_boot = np.asarray(pair[0]), np.asarray(pair[1])
        assert len(x_boot) == len(y_boot), (
            f"{name}.generate_samples item {k}: X_boot/y_boot length mismatch "
            f"({len(x_boot)} vs {len(y_boot)})."
        )
        assert x_boot.ndim == 2 and x_boot.shape[1] >= 1, (
            f"{name}.generate_samples item {k}: X_boot must be 2-D with >=1 feature "
            f"(got shape {x_boot.shape})."
        )

    # Determinism: identical seed must reproduce the samples exactly.
    samples2 = strategy.generate_samples(X_arr, y_arr, n_boot, np.random.default_rng(123))
    for k, ((x1, y1), (x2, y2)) in enumerate(zip(samples, samples2, strict=True)):
        assert np.array_equal(np.asarray(x1), np.asarray(x2)) and np.array_equal(
            np.asarray(y1), np.asarray(y2)
        ), f"{name}.generate_samples is non-deterministic at sample {k} (uses non-rng randomness?)."

    transformed = np.asarray(strategy.transform_for_predict(X_arr, 0))
    assert transformed.shape[0] == n, (
        f"{name}.transform_for_predict changed the row count ({transformed.shape[0]} != {n})."
    )


def check_forecast_adapter(
    adapter: SupportsForecast,
    *,
    train_values: ArrayLike | None = None,
    test_size: int = 5,
    horizon: int = 5,
) -> None:
    """Assert ``adapter`` satisfies the forecast-adapter contract for the comparison runner.

    Invariants (each raises ``AssertionError`` on violation):

    1. **Structural** — satisfies the :class:`~temporalcv.SupportsForecast` Protocol
       (``model_name`` + ``package_name`` + ``fit_predict`` + ``get_params`` present).
    2. **Metadata** — ``model_name`` / ``package_name`` are non-empty strings and
       ``get_params()`` returns a dict.
    3. **fit_predict** — returns a finite ndarray whose trailing axis has length ``test_size``
       (shape ``(test_size,)`` for a single series, ``(n_series, test_size)`` for a panel).

    Parameters
    ----------
    adapter : SupportsForecast
        The forecasting adapter under test.
    train_values : ArrayLike, optional
        Training series; a deterministic length-60 ramp is generated when omitted.
    test_size, horizon : int, default=5
        Forecast length / horizon passed to ``fit_predict``.
    """
    name = type(adapter).__name__
    assert isinstance(adapter, SupportsForecast), (
        f"{name} does not satisfy the SupportsForecast Protocol "
        f"(needs model_name, package_name, fit_predict, get_params)."
    )
    assert isinstance(adapter.model_name, str) and adapter.model_name, (
        f"{name}.model_name must be a non-empty str."
    )
    assert isinstance(adapter.package_name, str) and adapter.package_name, (
        f"{name}.package_name must be a non-empty str."
    )
    assert isinstance(adapter.get_params(), dict), f"{name}.get_params() must return a dict."

    train = np.linspace(0.0, 1.0, 60) if train_values is None else np.asarray(train_values)
    preds = np.asarray(adapter.fit_predict(train, test_size, horizon))
    assert preds.shape[-1] == test_size, (
        f"{name}.fit_predict returned {preds.shape[-1]} predictions on its trailing axis, "
        f"expected test_size={test_size}."
    )
    assert np.all(np.isfinite(preds)), f"{name}.fit_predict returned non-finite predictions."
