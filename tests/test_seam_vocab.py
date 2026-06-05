"""Tests for the v2.0 seam-vocabulary completion (issues #12 + #14).

Covers:
- ``SupportsBootstrap`` / ``SupportsForecast`` static Protocols (the bootstrap-strategy and
  forecast-adapter accept-seams), positive and negative structural conformance.
- The executable conformance checks ``check_bootstrap_strategy`` / ``check_forecast_adapter``,
  positive on every concrete impl and negative on deliberately-broken ones.
- ``TemporalTags`` capability descriptor (#14) and the ``temporal_tags()`` cross-validation that
  ``check_temporal_splitter`` performs (declared tags must match observed behavior).

See ``docs/adr/0001-v2-seams-and-layout.md``, ``STYLE.md``, and the hub
``library-design-playbook.md`` ("Protocol for what you accept; ABC/mixin for what you share;
declare capabilities as data/tags; the conformance suite is the contract").
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from temporalcv import (
    BlockedTimeSeriesCV,
    CrossFitCV,
    CrossFitter,
    FeatureBagging,
    MovingBlockBootstrap,
    ResidualBootstrap,
    StationaryBootstrap,
    SupportsBootstrap,
    SupportsForecast,
    TemporalTags,
    TimeSeriesBagger,
    TimeSeriesCrossValidator,
    WalkForwardCV,
    check_bootstrap_strategy,
    check_forecast_adapter,
    check_temporal_splitter,
)
from temporalcv.bagging.base import BootstrapStrategy
from temporalcv.compare.adapters.multi_series import MultiSeriesAdapter, ProgressAdapter
from temporalcv.compare.base import ForecastAdapter, NaiveAdapter, SeasonalNaiveAdapter

ALL_STRATEGIES = [
    MovingBlockBootstrap(block_length=5),
    StationaryBootstrap(),
    FeatureBagging(),
    ResidualBootstrap(),
]
ALL_ADAPTERS = [
    NaiveAdapter(),
    SeasonalNaiveAdapter(season_length=12),
    MultiSeriesAdapter(NaiveAdapter()),
    ProgressAdapter(NaiveAdapter()),
]
TAGGED_SPLITTERS = [
    WalkForwardCV(n_splits=3),
    TimeSeriesCrossValidator(n_splits=3),
    BlockedTimeSeriesCV(n_splits=3),
    CrossFitCV(n_splits=3),
]


# =============================================================================
# #12 — static Protocol seams (structural typing), positive + negative
# =============================================================================


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: type(s).__name__)
def test_concrete_strategy_satisfies_supports_bootstrap(strategy: SupportsBootstrap) -> None:
    assert isinstance(strategy, SupportsBootstrap)


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: type(a).__name__)
def test_concrete_adapter_satisfies_supports_forecast(adapter: SupportsForecast) -> None:
    assert isinstance(adapter, SupportsForecast)


def test_supports_bootstrap_negative_missing_transform() -> None:
    class _OnlyGenerate:
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            return []

    # Missing transform_for_predict -> not a SupportsBootstrap.
    assert not isinstance(_OnlyGenerate(), SupportsBootstrap)


def test_supports_forecast_negative_missing_fit_predict() -> None:
    class _NoFitPredict:
        @property
        def model_name(self) -> str:
            return "x"

        @property
        def package_name(self) -> str:
            return "y"

        def get_params(self) -> dict[str, Any]:
            return {}

    # Missing fit_predict -> not a SupportsForecast.
    assert not isinstance(_NoFitPredict(), SupportsForecast)


def test_supports_bootstrap_is_structural_no_subclassing_required() -> None:
    """A duck-typed strategy that never imports temporalcv still conforms."""

    class _DuckStrategy:
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(X, y) for _ in range(n_samples)]

        def transform_for_predict(self, X: np.ndarray, estimator_idx: int) -> np.ndarray:
            return X

    assert isinstance(_DuckStrategy(), SupportsBootstrap)


# =============================================================================
# #12 — conformance checks, positive + negative
# =============================================================================


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: type(s).__name__)
def test_check_bootstrap_strategy_positive(strategy: SupportsBootstrap) -> None:
    check_bootstrap_strategy(strategy)  # must not raise


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: type(a).__name__)
def test_check_forecast_adapter_positive(adapter: SupportsForecast) -> None:
    check_forecast_adapter(adapter)  # must not raise


def test_check_bootstrap_strategy_negative_wrong_sample_count() -> None:
    class _IgnoresNBoot(BootstrapStrategy):
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            return [(X, y)]  # ignores n_samples -> always 1

    with pytest.raises(AssertionError, match="expected n_boot"):
        check_bootstrap_strategy(_IgnoresNBoot())


def test_check_bootstrap_strategy_negative_nondeterministic() -> None:
    class _GlobalRandom(BootstrapStrategy):
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            # Draws from global state, NOT the supplied rng -> non-deterministic.
            out = []
            for _ in range(n_samples):
                idx = np.random.default_rng().integers(0, len(X), len(X))
                out.append((X[idx], y[idx]))
            return out

    with pytest.raises(AssertionError, match="non-deterministic"):
        check_bootstrap_strategy(_GlobalRandom())


def test_check_forecast_adapter_negative_wrong_length() -> None:
    class _WrongLength(ForecastAdapter):
        @property
        def model_name(self) -> str:
            return "Wrong"

        @property
        def package_name(self) -> str:
            return "test"

        def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
            return np.zeros(test_size + 1)  # one too many

    with pytest.raises(AssertionError, match="expected test_size"):
        check_forecast_adapter(_WrongLength())


# =============================================================================
# #12 — consumer integration (the retyped param accepts the seam)
# =============================================================================


def test_bagger_accepts_supports_bootstrap_strategy() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 3))
    y = X[:, 0] + rng.standard_normal(40) * 0.1
    bagger = TimeSeriesBagger(
        Ridge(), MovingBlockBootstrap(block_length=5), n_estimators=5, random_state=0
    )
    bagger.fit(X, y)
    preds = bagger.predict(X)
    assert preds.shape == (40,)
    assert np.all(np.isfinite(preds))


# =============================================================================
# #14 — TemporalTags descriptor
# =============================================================================


def test_temporal_tags_is_frozen() -> None:
    tags = WalkForwardCV(n_splits=3).temporal_tags()
    with pytest.raises(AttributeError):
        tags.forward_only = False  # type: ignore[misc]


def test_temporal_tags_to_dict() -> None:
    tags = CrossFitCV(n_splits=3).temporal_tags()
    assert tags.to_dict() == {
        "forward_only": True,
        "deterministic": True,
        "produces_oof": True,
        "requires_groups": False,
    }


def test_temporal_tags_has_no_schema_version() -> None:
    """TemporalTags is a capability descriptor, not a versioned result object."""
    assert not hasattr(TemporalTags, "SCHEMA_VERSION")


@pytest.mark.parametrize(
    ("splitter", "expected_oof"),
    [
        (WalkForwardCV(n_splits=3), False),
        (TimeSeriesCrossValidator(n_splits=3), False),
        (BlockedTimeSeriesCV(n_splits=3), False),
        (CrossFitCV(n_splits=3), True),
    ],
    ids=lambda x: type(x).__name__ if hasattr(x, "split") else str(x),
)
def test_temporal_tags_values(splitter: Any, expected_oof: bool) -> None:
    tags = splitter.temporal_tags()
    assert tags.forward_only is True
    assert tags.deterministic is True
    assert tags.requires_groups is False
    assert tags.produces_oof is expected_oof
    # produces_oof must agree with actual CrossFitter membership.
    assert tags.produces_oof == isinstance(splitter, CrossFitter)


# =============================================================================
# #14 — conformance cross-validates declared tags against observed behavior
# =============================================================================


@pytest.mark.parametrize("splitter", TAGGED_SPLITTERS, ids=lambda s: type(s).__name__)
def test_check_temporal_splitter_passes_with_honest_tags(splitter: Any) -> None:
    check_temporal_splitter(splitter)  # must not raise


def test_check_temporal_splitter_skips_when_no_tags() -> None:
    """The tag cross-check is optional: a tag-less splitter still passes."""

    class _TaglessSplitter:
        def split(
            self,
            X: Any,
            y: Any = None,
            groups: Any = None,
        ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
            n = len(X)
            cut = n // 2
            yield np.arange(cut, dtype=np.intp), np.arange(cut, n, dtype=np.intp)

        def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
            return 1

    assert not hasattr(_TaglessSplitter(), "temporal_tags")
    check_temporal_splitter(_TaglessSplitter())  # must not raise


def _wfcv_with_tags(tags: TemporalTags) -> Any:
    class _LyingCV(WalkForwardCV):
        def temporal_tags(self) -> TemporalTags:
            return tags

    return _LyingCV(n_splits=3)


def test_lying_produces_oof_tag_is_caught() -> None:
    # Claims OOF but is not a CrossFitter.
    splitter = _wfcv_with_tags(TemporalTags(True, True, True, False))
    with pytest.raises(AssertionError, match="produces_oof"):
        check_temporal_splitter(splitter)


def test_lying_forward_only_tag_is_caught() -> None:
    splitter = _wfcv_with_tags(TemporalTags(False, True, False, False))
    with pytest.raises(AssertionError, match="forward_only"):
        check_temporal_splitter(splitter)


def test_lying_deterministic_tag_is_caught() -> None:
    splitter = _wfcv_with_tags(TemporalTags(True, False, False, False))
    with pytest.raises(AssertionError, match="deterministic"):
        check_temporal_splitter(splitter)


def test_lying_requires_groups_tag_is_caught() -> None:
    splitter = _wfcv_with_tags(TemporalTags(True, True, False, True))
    with pytest.raises(AssertionError, match="requires_groups"):
        check_temporal_splitter(splitter)


# =============================================================================
# Review remediation — fail-loud tag declarations (#14 / SF-1)
# =============================================================================


def test_temporal_tags_as_property_is_cross_checked() -> None:
    """A splitter exposing temporal_tags as a PROPERTY is still cross-checked, not silently skipped."""

    class _PropTagCV(WalkForwardCV):
        @property
        def temporal_tags(self) -> TemporalTags:  # type: ignore[override]
            return TemporalTags(True, True, True, False)  # lies: produces_oof

    with pytest.raises(AssertionError, match="produces_oof"):
        check_temporal_splitter(_PropTagCV(n_splits=3))


def test_temporal_tags_malformed_raises() -> None:
    """temporal_tags yielding a non-TemporalTags fails loud (not silently skipped)."""

    class _BadTagCV(WalkForwardCV):
        def temporal_tags(self) -> Any:
            return {"forward_only": True}  # not a TemporalTags

    with pytest.raises(AssertionError, match="must be a TemporalTags"):
        check_temporal_splitter(_BadTagCV(n_splits=3))


# =============================================================================
# Review remediation — un-guarded conformance invariants (negative tests)
# =============================================================================


def test_check_forecast_adapter_negative_nonfinite() -> None:
    class _NaNAdapter(ForecastAdapter):
        @property
        def model_name(self) -> str:
            return "NaN"

        @property
        def package_name(self) -> str:
            return "test"

        def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
            out = np.zeros(test_size)
            out[0] = np.nan
            return out

    with pytest.raises(AssertionError, match="non-finite"):
        check_forecast_adapter(_NaNAdapter())


def test_check_forecast_adapter_negative_scalar_return() -> None:
    class _ScalarAdapter(ForecastAdapter):
        @property
        def model_name(self) -> str:
            return "Scalar"

        @property
        def package_name(self) -> str:
            return "test"

        def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
            return np.asarray(3.0)  # 0-d scalar

    with pytest.raises(AssertionError, match="0-d/scalar"):
        check_forecast_adapter(_ScalarAdapter())


def test_check_forecast_adapter_negative_get_params_not_dict() -> None:
    class _BadParams(ForecastAdapter):
        @property
        def model_name(self) -> str:
            return "BadParams"

        @property
        def package_name(self) -> str:
            return "test"

        def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
            return np.zeros(test_size)

        def get_params(self) -> dict[str, Any]:
            return ["not", "a", "dict"]  # type: ignore[return-value]

    with pytest.raises(AssertionError, match="must return a dict"):
        check_forecast_adapter(_BadParams())


def test_check_forecast_adapter_negative_empty_model_name() -> None:
    class _EmptyName(ForecastAdapter):
        @property
        def model_name(self) -> str:
            return ""

        @property
        def package_name(self) -> str:
            return "test"

        def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
            return np.zeros(test_size)

    with pytest.raises(AssertionError, match="non-empty"):
        check_forecast_adapter(_EmptyName())


def test_check_forecast_adapter_panel_2d() -> None:
    """The advertised panel (n_series, test_size) output path is exercised."""
    rng = np.random.default_rng(0)
    panel = rng.standard_normal((3, 60))  # (n_series, T)
    check_forecast_adapter(MultiSeriesAdapter(NaiveAdapter()), train_values=panel)


def test_check_bootstrap_strategy_negative_1d_xboot() -> None:
    class _OneDX(BootstrapStrategy):
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            # First column only -> 1-D X_boot, but length still matches y (so the 2-D check, not
            # the length check, is what fires).
            return [(np.asarray(X)[:, 0], np.asarray(y)) for _ in range(n_samples)]

    with pytest.raises(AssertionError, match="2-D"):
        check_bootstrap_strategy(_OneDX())


def test_check_bootstrap_strategy_negative_pair_shape() -> None:
    class _NotPairs(BootstrapStrategy):
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            return [np.asarray(X) for _ in range(n_samples)]  # type: ignore[misc]

    with pytest.raises(AssertionError, match="pair"):
        check_bootstrap_strategy(_NotPairs())


def test_check_bootstrap_strategy_negative_transform_changes_rows() -> None:
    class _DropsRow(BootstrapStrategy):
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            X, y = np.asarray(X), np.asarray(y)
            idx = rng.integers(0, len(X), len(X))
            return [(X[idx], y[idx]) for _ in range(n_samples)]

        def transform_for_predict(self, X: np.ndarray, estimator_idx: int) -> np.ndarray:
            return np.asarray(X)[:-1]  # drops a row

    with pytest.raises(AssertionError, match="row count"):
        check_bootstrap_strategy(_DropsRow())


def test_check_bootstrap_strategy_negative_ignores_rng() -> None:
    """A strategy that ignores the supplied rng (constant output) is caught by the rng probe."""

    class _ConstantStrategy(BootstrapStrategy):
        def generate_samples(
            self, X: np.ndarray, y: np.ndarray, n_samples: int, rng: np.random.Generator
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            X, y = np.asarray(X), np.asarray(y)
            return [(X.copy(), y.copy()) for _ in range(n_samples)]  # ignores rng

    with pytest.raises(AssertionError, match="ignores the supplied Generator"):
        check_bootstrap_strategy(_ConstantStrategy())


def test_supports_forecast_negative_missing_property() -> None:
    class _NoModelName:
        @property
        def package_name(self) -> str:
            return "y"

        def fit_predict(self, train_values: np.ndarray, test_size: int, horizon: int) -> np.ndarray:
            return np.zeros(test_size)

        def get_params(self) -> dict[str, Any]:
            return {}

    # Missing the model_name property -> not a SupportsForecast.
    assert not isinstance(_NoModelName(), SupportsForecast)
