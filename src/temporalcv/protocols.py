"""Static typed seams for temporalcv (v2.0).

These Protocols are a **static typing aid only**. ``@runtime_checkable`` verifies member
*presence*, not signatures, and is comparatively slow — use it for a one-shot boundary
check, never as a validator or on a hot path. Concrete splitters additionally subclass
sklearn ``BaseCrossValidator`` for ``cross_val_score``/``Pipeline`` interop; this Protocol
describes the shape typed consumers (e.g. ``dml_ts``) annotate against, without forcing an
import of temporalcv or sklearn.

See ``STYLE.md`` and ``docs/adr/0001-v2-seams-and-layout.md`` for the seam strategy, and
``~/Claude/lever_of_archimedes/patterns/library-design-playbook.md`` for the rationale.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Splitter",
    "CrossFitter",
    "SupportsFitPredict",
    "SupportsBootstrap",
    "SupportsForecast",
]


@runtime_checkable
class Splitter(Protocol):
    """A temporal cross-validation splitter.

    Yields ``(train_idx, test_idx)`` integer arrays in temporal order. Concrete
    implementations also subclass sklearn ``BaseCrossValidator`` so they work with
    ``cross_val_score``/``Pipeline``; this Protocol is the typed seam only.
    """

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer index arrays."""
        ...

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int | None:
        """Number of splits the splitter will produce.

        May return ``None`` for a lazy splitter whose split count is not known ahead of
        iteration; the conformance suite skips the count-consistency check in that case.
        """
        ...


@runtime_checkable
class CrossFitter(Splitter, Protocol):
    """A splitter that also produces out-of-fold predictions / residuals.

    Forward-only semantics: rows with no valid out-of-fold prediction (e.g. fold 0,
    which has no history to train on) are returned as ``NaN``. This is the seam a
    debiased/orthogonalized consumer (Double ML, partialling-out) builds on.
    """

    def fit_predict(self, model: SupportsFitPredict, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Out-of-fold predictions aligned to ``y`` (uncovered rows are ``NaN``)."""
        ...

    def fit_predict_residuals(
        self, model: SupportsFitPredict, X: ArrayLike, y: ArrayLike
    ) -> np.ndarray:
        """Out-of-fold residuals ``y - y_hat`` (uncovered rows are ``NaN``)."""
        ...


@runtime_checkable
class SupportsFitPredict(Protocol):
    """A model with the scikit-learn ``fit``/``predict`` interface.

    The canonical **estimator** seam: nuisance learners (e.g. ``model_a``/``model_b`` in
    :func:`~temporalcv.cross_fit_residualize`), bagging base estimators, and gate models
    all annotate against this one Protocol. ``@runtime_checkable`` checks member
    *presence* only (a static aid; never a hot-path validator) — use
    :func:`~temporalcv.check_temporal_estimator` for the behavioral contract.

    Inputs are typed ``ArrayLike`` (consumers may pass arrays, lists, or frames);
    ``predict`` returns a concrete ``np.ndarray`` as scikit-learn estimators do.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> Any:
        """Fit the model to training data; returns the fitted estimator."""
        ...

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict targets for ``X``."""
        ...


@runtime_checkable
class SupportsBootstrap(Protocol):
    """A time-series bootstrap resampling strategy — the accept-seam for
    :class:`~temporalcv.TimeSeriesBagger`.

    This Protocol is the typed seam consumers annotate against. Our own strategies additionally
    subclass the owned shared-impl base :class:`~temporalcv.BootstrapStrategy` (which supplies the
    default :meth:`transform_for_predict` and instantiation fail-fast); a third party may instead
    satisfy this Protocol structurally, without importing temporalcv.

    Inputs are concrete ``np.ndarray`` because this is an **internal** seam: ``TimeSeriesBagger``
    normalizes its public ``ArrayLike`` input via ``np.asarray`` *before* delegating to the
    strategy, so a strategy always receives a materialized array. The ``ArrayLike`` boundary lives
    on the bagger, not here. Use :func:`~temporalcv.check_bootstrap_strategy` for the behavioral
    contract (``@runtime_checkable`` verifies member *presence* only).
    """

    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate ``n_samples`` ``(X_boot, y_boot)`` samples preserving time-series structure."""
        ...

    def transform_for_predict(self, X: np.ndarray, estimator_idx: int, /) -> np.ndarray:
        """Transform ``X`` for the ``estimator_idx``-th estimator at predict time (default: identity).

        The index is **positional-only**: the contract is the argument position, not its name (the
        shared base names it ``_estimator_idx``; :class:`~temporalcv.FeatureBagging` names it
        ``estimator_idx``).
        """
        ...


@runtime_checkable
class SupportsForecast(Protocol):
    """A forecasting-package adapter — the accept-seam for the model-comparison runner
    (:func:`~temporalcv.run_comparison` and friends).

    This Protocol is the typed seam consumers annotate against. Our own adapters additionally
    subclass the owned shared-impl base :class:`~temporalcv.ForecastAdapter` (which supplies the
    default :meth:`get_params` and instantiation fail-fast); a third party may instead satisfy this
    Protocol structurally. Use :func:`~temporalcv.check_forecast_adapter` for the behavioral
    contract (``@runtime_checkable`` verifies member *presence* only).
    """

    @property
    def model_name(self) -> str:
        """Human-readable model name (e.g. ``"AutoARIMA"``)."""
        ...

    @property
    def package_name(self) -> str:
        """Originating package (e.g. ``"statsforecast"``)."""
        ...

    def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
        """Fit on ``train_values`` and return predictions of length ``test_size``."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Model hyperparameters (default: empty mapping)."""
        ...
