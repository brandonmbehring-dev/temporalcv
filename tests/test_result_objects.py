"""Library-wide v2.0 result-object contract: frozen, slotted, versioned, JSON-serializable.

Every result/output value object in temporalcv (outside the benchmarks dataset holders) must
be a frozen, slotted dataclass exposing a ``to_dict()`` that carries an integer
``schema_version`` and is ``json.dumps``-able. This parametrized registry is the single
enforcement point — when a new result object is added, register a factory in
``RESULT_FACTORIES`` below.

Companion suites: identity eq/hash for the array/dict-bearing subset lives in
``test_result_eq_hash_safety.py``; cv-result value-roundtrip details in
``test_cv_result_objects.py``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable

import numpy as np
import pytest

from temporalcv.compare.base import ComparisonReport, ComparisonResult, ModelResult
from temporalcv.conformal import CoverageDiagnostics, PredictionInterval
from temporalcv.cv import NestedCVResult, SplitInfo, SplitResult, WalkForwardResults
from temporalcv.cv_financial import PurgedSplit
from temporalcv.diagnostics.influence import InfluenceDiagnostic
from temporalcv.diagnostics.sensitivity import GapSensitivityResult
from temporalcv.gates import GateResult, GateStatus, StratifiedValidationReport, ValidationReport
from temporalcv.inference.block_bootstrap_ci import BlockBootstrapResult
from temporalcv.inference.wild_bootstrap import WildBootstrapResult
from temporalcv.metrics.event import BrierScoreResult, PRAUCResult
from temporalcv.metrics.volatility_weighted import VolatilityStratifiedResult
from temporalcv.persistence import MoveConditionalResult
from temporalcv.regimes import StratifiedMetricsResult
from temporalcv.statistical_tests import (
    BidirectionalEncompassingResult,
    CWTestResult,
    DMTestResult,
    EncompassingTestResult,
    GWTestResult,
    MultiHorizonResult,
    MultiModelComparisonResult,
    MultiModelHorizonResult,
    PTTestResult,
    RealityCheckResult,
    SPATestResult,
)


def _split_result() -> SplitResult:
    return SplitResult(
        split_idx=0,
        train_start=0,
        train_end=9,
        test_start=11,
        test_end=15,
        predictions=np.array([1.0, 2.0, 3.0]),
        actuals=np.array([1.1, 1.9, 3.2]),
    )


def _nested_cv_result() -> NestedCVResult:
    return NestedCVResult(
        best_params={"alpha": 1.0},
        outer_scores=np.array([0.1, 0.2, 0.15]),
        mean_outer_score=0.15,
        std_outer_score=0.04,
        inner_cv_results=[{"fold": 0}],
        n_outer_splits=3,
        n_inner_splits=2,
        scoring="neg_mae",
        best_params_per_fold=[{"alpha": 1.0}],
        params_stability=1.0,
    )


def _dm() -> DMTestResult:
    return DMTestResult(
        statistic=np.float64(1.5),
        pvalue=np.float64(0.03),
        h=1,
        n=100,
        loss="squared",
        alternative="two-sided",
        harvey_adjusted=True,
        mean_loss_diff=np.float64(0.1),
    )


def _encompassing() -> EncompassingTestResult:
    return EncompassingTestResult(
        lambda_coef=0.1,
        statistic=1.0,
        pvalue=0.05,
        encompasses=True,
        optimal_weight_b=0.1,
        direction="a_encompasses_b",
        n=100,
        h=1,
    )


def _multi_model() -> MultiModelComparisonResult:
    return MultiModelComparisonResult(
        pairwise_results={("A", "B"): _dm()},
        best_model="A",
        bonferroni_alpha=0.025,
        original_alpha=0.05,
        model_rankings=[("A", 0.1), ("B", 0.2)],
        significant_pairs=[("A", "B")],
    )


def _model_result() -> ModelResult:
    return ModelResult(
        model_name="A",
        package="p",
        metrics={"mae": 0.5},
        predictions=np.array([1.0, 2.0]),
        runtime_seconds=1.0,
    )


def _comparison_result() -> ComparisonResult:
    return ComparisonResult(dataset_name="d", models=[_model_result()], primary_metric="mae")


def _validation_report() -> ValidationReport:
    return ValidationReport(gates=[GateResult(name="g", status=GateStatus.PASS, message="ok")])


# Every result object in the library. Add new ones here.
RESULT_FACTORIES: list[tuple[str, Callable[[], object]]] = [
    # cv.py
    (
        "SplitInfo",
        lambda: SplitInfo(split_idx=0, train_start=0, train_end=9, test_start=11, test_end=15),
    ),
    ("SplitResult", _split_result),
    ("WalkForwardResults", lambda: WalkForwardResults(splits=[_split_result()])),
    ("NestedCVResult", _nested_cv_result),
    # statistical_tests
    ("DMTestResult", _dm),
    (
        "PTTestResult",
        lambda: PTTestResult(
            statistic=1.0, pvalue=0.05, accuracy=0.6, expected=0.5, n=100, n_classes=2
        ),
    ),
    (
        "GWTestResult",
        lambda: GWTestResult(
            statistic=1.0,
            pvalue=0.05,
            r_squared=0.1,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.1,
        ),
    ),
    (
        "CWTestResult",
        lambda: CWTestResult(
            statistic=1.0,
            pvalue=0.05,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=True,
            mean_loss_diff=0.1,
            mean_loss_diff_adjusted=0.05,
            adjustment_magnitude=0.01,
        ),
    ),
    ("MultiModelComparisonResult", _multi_model),
    (
        "MultiHorizonResult",
        lambda: MultiHorizonResult(
            horizons=(1, 2),
            dm_results={1: _dm()},
            model_1_name="m1",
            model_2_name="m2",
            n_per_horizon={1: 100},
            loss="squared",
            alternative="two-sided",
            alpha=0.05,
        ),
    ),
    (
        "MultiModelHorizonResult",
        lambda: MultiModelHorizonResult(
            horizons=(1,),
            model_names=("A", "B"),
            pairwise_by_horizon={1: _multi_model()},
            alpha=0.05,
        ),
    ),
    ("EncompassingTestResult", _encompassing),
    (
        "BidirectionalEncompassingResult",
        lambda: BidirectionalEncompassingResult(
            a_encompasses_b=_encompassing(),
            b_encompasses_a=_encompassing(),
            recommendation="use_a",
            combined_weight_b=None,
        ),
    ),
    (
        "RealityCheckResult",
        lambda: RealityCheckResult(
            statistic=2.0,
            pvalue=0.01,
            best_model="A",
            individual_statistics={"A": 1.0},
            mean_losses={"A": 0.5},
            n_bootstrap=100,
            block_size=5,
            n=50,
        ),
    ),
    (
        "SPATestResult",
        lambda: SPATestResult(
            statistic=2.0,
            pvalue=0.01,
            pvalue_consistent=0.02,
            pvalue_lower=0.005,
            best_model="A",
            individual_statistics={"A": 1.0},
            mean_losses={"A": 0.5},
            n_bootstrap=100,
            block_size=5,
            n=50,
        ),
    ),
    # gates
    (
        "GateResult",
        lambda: GateResult(name="g", status=GateStatus.PASS, message="ok", details={"p": 0.5}),
    ),
    ("ValidationReport", _validation_report),
    (
        "StratifiedValidationReport",
        lambda: StratifiedValidationReport(
            overall=_validation_report(),
            by_regime={"r": _validation_report()},
            regime_counts={"r": 10},
            masked_regimes=[],
        ),
    ),
    # conformal
    (
        "PredictionInterval",
        lambda: PredictionInterval(
            point=np.array([1.0]),
            lower=np.array([0.5]),
            upper=np.array([1.5]),
            confidence=0.95,
            method="split",
        ),
    ),
    (
        "CoverageDiagnostics",
        lambda: CoverageDiagnostics(
            overall_coverage=0.94,
            target_coverage=0.95,
            coverage_gap=-0.01,
            undercoverage_warning=False,
            coverage_by_window={"w": 0.9},
            coverage_by_regime=None,
            n_observations=100,
        ),
    ),
    # metrics
    (
        "BrierScoreResult",
        lambda: BrierScoreResult(
            brier_score=0.1,
            reliability=0.05,
            resolution=0.2,
            uncertainty=0.25,
            n_samples=100,
            n_classes=2,
        ),
    ),
    (
        "PRAUCResult",
        lambda: PRAUCResult(
            pr_auc=0.8, baseline=0.3, precision_at_50_recall=0.7, n_positive=30, n_negative=70
        ),
    ),
    (
        "VolatilityStratifiedResult",
        lambda: VolatilityStratifiedResult(
            overall_mae=0.5,
            low_vol_mae=0.3,
            med_vol_mae=0.5,
            high_vol_mae=0.7,
            volatility_normalized_mae=0.4,
            n_low=10,
            n_med=10,
            n_high=10,
            vol_thresholds=(0.1, 0.3),
        ),
    ),
    # regimes / persistence
    (
        "StratifiedMetricsResult",
        lambda: StratifiedMetricsResult(
            overall_mae=0.5,
            overall_rmse=0.6,
            n_total=100,
            by_regime={"b": {"mae": 0.4, "n": 50}},
            masked_regimes=[],
        ),
    ),
    (
        "MoveConditionalResult",
        lambda: MoveConditionalResult(
            mae_up=0.3,
            mae_down=0.4,
            mae_flat=0.2,
            n_up=20,
            n_down=15,
            n_flat=65,
            skill_score=0.1,
            move_threshold=0.01,
        ),
    ),
    # compare
    ("ModelResult", _model_result),
    ("ComparisonResult", _comparison_result),
    ("ComparisonReport", lambda: ComparisonReport(results=[_comparison_result()])),
    # diagnostics / inference / financial (the M1 five)
    (
        "PurgedSplit",
        lambda: PurgedSplit(
            train_indices=np.array([0, 1, 2]),
            test_indices=np.array([5, 6]),
            n_purged=2,
            n_embargoed=1,
        ),
    ),
    (
        "InfluenceDiagnostic",
        lambda: InfluenceDiagnostic(
            observation_influence=np.array([0.1, 0.2]),
            observation_high_mask=np.array([False, True]),
            block_influence=np.array([0.15]),
            block_high_mask=np.array([False]),
            block_indices=[(0, 2)],
            n_high_influence_obs=1,
            n_high_influence_blocks=0,
            influence_threshold=2.0,
        ),
    ),
    (
        "GapSensitivityResult",
        lambda: GapSensitivityResult(
            gap_values=np.array([0, 1]),
            metrics=np.array([0.5, 0.6]),
            metric_name="mae",
            break_even_gap=None,
            sensitivity_score=0.1,
            degradation_threshold=0.1,
            baseline_metric=0.5,
            baseline_gap=0,
        ),
    ),
    (
        "BlockBootstrapResult",
        lambda: BlockBootstrapResult(
            estimate=1.0,
            ci_lower=0.8,
            ci_upper=1.2,
            alpha=0.05,
            std_error=0.1,
            n_bootstrap=100,
            block_length=5,
            bootstrap_distribution=np.array([0.9, 1.0, 1.1]),
        ),
    ),
    (
        "WildBootstrapResult",
        lambda: WildBootstrapResult(
            estimate=1.0,
            se=0.1,
            ci_lower=0.8,
            ci_upper=1.2,
            p_value=0.04,
            n_bootstrap=100,
            n_clusters=5,
            weight_type="rademacher",
            bootstrap_distribution=np.array([0.9, 1.0]),
        ),
    ),
]

_IDS = [name for name, _ in RESULT_FACTORIES]
_FACTORIES = [factory for _, factory in RESULT_FACTORIES]


@pytest.mark.parametrize("factory", _FACTORIES, ids=_IDS)
class TestResultObjectContract:
    """Every registered result object satisfies the v2.0 value-object contract."""

    def test_is_frozen(self, factory: Callable[[], object]) -> None:
        obj = factory()
        first_field = next(iter(dataclasses.fields(obj))).name  # type: ignore[arg-type]
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, first_field, None)

    def test_is_slotted(self, factory: Callable[[], object]) -> None:
        obj = factory()
        assert hasattr(type(obj), "__slots__")
        assert not hasattr(obj, "__dict__")

    def test_to_dict_carries_schema_version(self, factory: Callable[[], object]) -> None:
        obj = factory()
        schema_version = type(obj).SCHEMA_VERSION  # type: ignore[attr-defined]
        d = obj.to_dict()  # type: ignore[attr-defined]
        assert d["schema_version"] == schema_version
        assert isinstance(d["schema_version"], int)

    def test_to_dict_is_json_serializable(self, factory: Callable[[], object]) -> None:
        obj = factory()
        json.dumps(obj.to_dict())  # type: ignore[attr-defined]  # must not raise
