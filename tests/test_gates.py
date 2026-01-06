"""
Tests for temporalcv.gates module.

Tests validation gate framework including:
- GateStatus enum behavior
- GateResult dataclass
- ValidationReport aggregation
- Individual gate functions
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import (
    GateStatus,
    GateResult,
    ValidationReport,
    gate_signal_verification,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
    gate_temporal_boundary,
    run_gates,
)


# =============================================================================
# Fixtures
# =============================================================================


class MockModel:
    """Mock model for testing gate functions."""

    def __init__(self, prediction_type: str = "mean") -> None:
        """
        Initialize mock model.

        Parameters
        ----------
        prediction_type : str
            "mean" - predicts mean of training y
            "leaky" - learns X->y mapping that only works for original ordering
            "random" - predicts random values
            "lag1" - predicts y shifted by 1 (optimal for AR process)
        """
        self.prediction_type = prediction_type
        self._mean: float = 0.0
        self._coeffs: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockModel":
        """Fit model to data."""
        self._mean = float(np.mean(y))
        if self.prediction_type == "leaky":
            # Learn coefficients - this will work on original X but not when
            # fit on shuffled y (different coefficients each time)
            X = np.asarray(X)
            y = np.asarray(y)
            # Simple least squares (add small regularization)
            XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
            Xty = X.T @ y
            self._coeffs = np.linalg.solve(XtX, Xty)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        X = np.asarray(X)
        n = len(X)

        if self.prediction_type == "mean":
            return np.full(n, self._mean)
        elif self.prediction_type == "leaky":
            # Use learned coefficients
            if self._coeffs is not None:
                return X @ self._coeffs
            return np.zeros(n)
        elif self.prediction_type == "random":
            return np.random.randn(n)
        elif self.prediction_type == "lag1":
            # Optimal predictor for AR(1): use first column (lag 1)
            return 0.95 * X[:, 0] if X.ndim > 1 else 0.95 * X
        else:
            return np.zeros(n)


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple test data for gates."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 5))
    y = rng.standard_normal(n)
    return X, y


# =============================================================================
# GateStatus Tests
# =============================================================================


class TestGateStatus:
    """Tests for GateStatus enum."""

    def test_status_values(self) -> None:
        """All status values should be accessible."""
        assert GateStatus.HALT.value == "HALT"
        assert GateStatus.WARN.value == "WARN"
        assert GateStatus.PASS.value == "PASS"
        assert GateStatus.SKIP.value == "SKIP"

    def test_status_comparison(self) -> None:
        """Status comparison should work."""
        assert GateStatus.HALT != GateStatus.PASS
        assert GateStatus.HALT == GateStatus.HALT


# =============================================================================
# GateResult Tests
# =============================================================================


class TestGateResult:
    """Tests for GateResult dataclass."""

    def test_basic_creation(self) -> None:
        """GateResult should be creatable with minimal args."""
        result = GateResult(
            name="test_gate",
            status=GateStatus.PASS,
            message="Test passed",
        )
        assert result.name == "test_gate"
        assert result.status == GateStatus.PASS
        assert result.message == "Test passed"
        assert result.metric_value is None
        assert result.threshold is None
        assert result.details == {}
        assert result.recommendation == ""

    def test_full_creation(self) -> None:
        """GateResult should store all fields."""
        result = GateResult(
            name="test_gate",
            status=GateStatus.HALT,
            message="Test failed",
            metric_value=0.5,
            threshold=0.2,
            details={"key": "value"},
            recommendation="Fix the issue",
        )
        assert result.metric_value == 0.5
        assert result.threshold == 0.2
        assert result.details["key"] == "value"
        assert result.recommendation == "Fix the issue"

    def test_str_format(self) -> None:
        """String format should include status and name."""
        result = GateResult(
            name="test_gate",
            status=GateStatus.HALT,
            message="Test failed",
        )
        s = str(result)
        assert "[HALT]" in s
        assert "test_gate" in s
        assert "Test failed" in s


# =============================================================================
# ValidationReport Tests
# =============================================================================


class TestValidationReport:
    """Tests for ValidationReport aggregation."""

    def test_empty_report(self) -> None:
        """Empty report should have PASS status."""
        report = ValidationReport()
        assert report.status == "PASS"
        assert report.failures == []
        assert report.warnings == []

    def test_single_pass(self) -> None:
        """Report with single PASS should be PASS."""
        report = ValidationReport(
            gates=[
                GateResult(name="g1", status=GateStatus.PASS, message="ok")
            ]
        )
        assert report.status == "PASS"

    def test_single_halt(self) -> None:
        """Report with single HALT should be HALT."""
        report = ValidationReport(
            gates=[
                GateResult(name="g1", status=GateStatus.HALT, message="fail")
            ]
        )
        assert report.status == "HALT"
        assert len(report.failures) == 1

    def test_halt_overrides_pass(self) -> None:
        """HALT should override PASS."""
        report = ValidationReport(
            gates=[
                GateResult(name="g1", status=GateStatus.PASS, message="ok"),
                GateResult(name="g2", status=GateStatus.HALT, message="fail"),
            ]
        )
        assert report.status == "HALT"

    def test_warn_without_halt(self) -> None:
        """WARN should show when no HALT."""
        report = ValidationReport(
            gates=[
                GateResult(name="g1", status=GateStatus.PASS, message="ok"),
                GateResult(name="g2", status=GateStatus.WARN, message="caution"),
            ]
        )
        assert report.status == "WARN"
        assert len(report.warnings) == 1

    def test_halt_overrides_warn(self) -> None:
        """HALT should override WARN."""
        report = ValidationReport(
            gates=[
                GateResult(name="g1", status=GateStatus.WARN, message="caution"),
                GateResult(name="g2", status=GateStatus.HALT, message="fail"),
            ]
        )
        assert report.status == "HALT"

    def test_skip_ignored(self) -> None:
        """SKIP should not affect status."""
        report = ValidationReport(
            gates=[
                GateResult(name="g1", status=GateStatus.SKIP, message="skipped"),
                GateResult(name="g2", status=GateStatus.PASS, message="ok"),
            ]
        )
        assert report.status == "PASS"

    def test_summary_format(self) -> None:
        """Summary should be readable."""
        report = ValidationReport(
            gates=[
                GateResult(
                    name="g1",
                    status=GateStatus.HALT,
                    message="fail",
                    recommendation="fix it",
                )
            ]
        )
        summary = report.summary()
        assert "VALIDATION REPORT" in summary
        assert "HALT" in summary
        assert "g1" in summary


# =============================================================================
# gate_signal_verification Tests
# =============================================================================


class TestGateShuffledTarget:
    """Tests for shuffled target gate."""

    def test_pass_on_mean_predictor(self, simple_data: tuple) -> None:
        """Mean predictor should pass (doesn't use temporal info)."""
        X, y = simple_data
        model = MockModel("mean")

        # Use IID permutation for this test - it tests that a mean predictor
        # doesn't beat a random baseline, which is clearer with IID shuffling
        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            permutation="iid",  # Explicit IID for backward compatibility
            random_state=42,
        )

        # Mean predictor shouldn't beat shuffled baseline significantly
        assert result.status in (GateStatus.PASS, GateStatus.WARN)
        assert "signal_verification" in result.name

    def test_halt_on_leaky_predictor(self) -> None:
        """Leaky predictor should HALT when X strongly predicts y."""
        rng = np.random.default_rng(42)
        n = 100

        # Create data with strong X->y relationship (the "leak")
        X = rng.standard_normal((n, 5))
        true_coeffs = np.array([2.0, -1.5, 1.0, 0.5, -0.8])
        y = X @ true_coeffs + rng.standard_normal(n) * 0.1  # Low noise

        model = MockModel("leaky")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            threshold=0.05,
            method="effect_size",  # Use effect size mode for this test
            random_state=42,
        )

        # Model on real data will fit well (low MAE)
        # Model on shuffled will fit poorly (high MAE)
        # So improvement ratio should be positive and trigger HALT
        assert result.status == GateStatus.HALT
        assert result.metric_value is not None
        assert result.metric_value > 0.05

    def test_details_contain_metrics(self, simple_data: tuple) -> None:
        """Result details should contain all metrics."""
        X, y = simple_data
        model = MockModel("mean")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            random_state=42,
        )

        assert "mae_real" in result.details
        assert "mae_shuffled_avg" in result.details
        assert "mae_shuffled_all" in result.details
        assert "n_shuffles" in result.details

    def test_reproducible_with_seed(self, simple_data: tuple) -> None:
        """Results should be reproducible with same seed."""
        X, y = simple_data
        model = MockModel("mean")

        result1 = gate_signal_verification(model, X, y, n_shuffles=3, random_state=42)
        result2 = gate_signal_verification(model, X, y, n_shuffles=3, random_state=42)

        assert result1.metric_value == result2.metric_value

    def test_block_permutation_preserves_local_structure(self) -> None:
        """Block permutation should preserve within-block ordering."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)
        model = MockModel("mean")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            permutation="block",
            block_size=10,  # Explicit block size
            random_state=42,
        )

        # Verify block permutation was used
        assert result.details["permutation"] == "block"
        assert result.details["block_size"] == 10

    def test_auto_block_size_uses_cube_root(self) -> None:
        """Auto block size should use n^(1/3) per Kunsch (1989)."""
        rng = np.random.default_rng(42)
        n = 125  # 125^(1/3) = 5
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)
        model = MockModel("mean")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=2,
            permutation="block",
            block_size="auto",
            random_state=42,
        )

        # 125^(1/3) = 5
        assert result.details["block_size"] == 5

    def test_iid_permutation_no_block_size(self) -> None:
        """IID permutation should report None for block_size."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        model = MockModel("mean")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=2,
            permutation="iid",
            random_state=42,
        )

        assert result.details["permutation"] == "iid"
        assert result.details["block_size"] is None

    def test_invalid_permutation_raises(self) -> None:
        """Invalid permutation value should raise ValueError."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        model = MockModel("mean")

        with pytest.raises(ValueError, match="permutation must be"):
            gate_signal_verification(
                model=model,
                X=X,
                y=y,
                permutation="invalid",  # type: ignore
                random_state=42,
            )

    def test_strict_mode_uses_199_shuffles(self) -> None:
        """strict=True should override n_shuffles to 199 for p-value resolution of 0.005."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        model = MockModel("mean")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=5,  # Request 5, but strict=True should override
            strict=True,
            random_state=42,
        )

        # Verify effective n_shuffles is 199
        assert result.details["n_shuffles"] == 5  # Original request preserved
        assert result.details["n_shuffles_effective"] == 199  # Overridden to 199
        assert result.details["strict"] is True
        assert result.details["min_pvalue"] == pytest.approx(0.005, rel=0.01)

    def test_strict_mode_does_not_downgrade(self) -> None:
        """strict=True should not reduce n_shuffles if already >= 199."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        model = MockModel("mean")

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=250,  # Already above 199
            strict=True,
            random_state=42,
        )

        # Should use original since it's already >= 199
        assert result.details["n_shuffles_effective"] == 250

    def test_power_warning_for_low_n_shuffles(self) -> None:
        """Low n_shuffles without strict=True should emit power warning."""
        import warnings

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        model = MockModel("mean")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gate_signal_verification(
                model=model,
                X=X,
                y=y,
                n_shuffles=5,
                strict=False,
                random_state=42,
            )

        # Should have power warning
        power_warnings = [x for x in w if "limited statistical power" in str(x.message)]
        assert len(power_warnings) >= 1
        assert "Phipson & Smyth" in str(power_warnings[0].message)


# =============================================================================
# gate_synthetic_ar1 Tests
# =============================================================================


class TestGateSyntheticAR1:
    """Tests for synthetic AR(1) gate."""

    def test_pass_on_mean_predictor(self) -> None:
        """Mean predictor should pass (doesn't beat theoretical)."""
        model = MockModel("mean")

        result = gate_synthetic_ar1(
            model=model,
            phi=0.95,
            sigma=1.0,
            n_samples=200,
            n_lags=3,
            random_state=42,
        )

        # Mean predictor is worse than optimal
        assert result.status == GateStatus.PASS

    def test_pass_on_lag1_predictor(self) -> None:
        """Optimal lag-1 predictor should pass (matches theory)."""
        model = MockModel("lag1")

        result = gate_synthetic_ar1(
            model=model,
            phi=0.95,
            sigma=1.0,
            n_samples=500,
            n_lags=3,
            random_state=42,
        )

        # Optimal predictor should be close to theoretical
        assert result.status == GateStatus.PASS
        assert result.metric_value is not None
        # Ratio should be near 1.0 (within tolerance)
        assert 0.5 < result.metric_value < 2.0

    def test_details_contain_parameters(self) -> None:
        """Result should contain AR(1) parameters."""
        model = MockModel("mean")

        result = gate_synthetic_ar1(
            model=model,
            phi=0.9,
            sigma=2.0,
            n_samples=100,
            random_state=42,
        )

        assert result.details["phi"] == 0.9
        assert result.details["sigma"] == 2.0
        assert "theoretical_mae" in result.details
        assert "model_mae" in result.details

    def test_theoretical_mae_calculation(self) -> None:
        """Theoretical MAE should be sigma * sqrt(2/pi)."""
        model = MockModel("mean")

        result = gate_synthetic_ar1(
            model=model,
            phi=0.95,
            sigma=2.0,
            random_state=42,
        )

        expected_theoretical = 2.0 * np.sqrt(2 / np.pi)
        assert abs(result.details["theoretical_mae"] - expected_theoretical) < 0.01


# =============================================================================
# gate_suspicious_improvement Tests
# =============================================================================


class TestGateSuspiciousImprovement:
    """Tests for suspicious improvement gate."""

    def test_pass_on_small_improvement(self) -> None:
        """Small improvement should pass."""
        result = gate_suspicious_improvement(
            model_metric=0.95,
            baseline_metric=1.0,
            threshold=0.20,
        )

        assert result.status == GateStatus.PASS
        assert result.metric_value == pytest.approx(0.05)

    def test_warn_on_moderate_improvement(self) -> None:
        """Moderate improvement should warn."""
        result = gate_suspicious_improvement(
            model_metric=0.85,
            baseline_metric=1.0,
            threshold=0.20,
            warn_threshold=0.10,
        )

        assert result.status == GateStatus.WARN

    def test_halt_on_large_improvement(self) -> None:
        """Large improvement should halt."""
        result = gate_suspicious_improvement(
            model_metric=0.70,
            baseline_metric=1.0,
            threshold=0.20,
        )

        assert result.status == GateStatus.HALT
        assert "suspicious" in result.name.lower()

    def test_skip_on_zero_baseline(self) -> None:
        """Zero baseline should skip."""
        result = gate_suspicious_improvement(
            model_metric=0.5,
            baseline_metric=0.0,
        )

        assert result.status == GateStatus.SKIP

    def test_negative_improvement_passes(self) -> None:
        """Model worse than baseline should pass."""
        result = gate_suspicious_improvement(
            model_metric=1.2,
            baseline_metric=1.0,
        )

        assert result.status == GateStatus.PASS
        # Negative improvement
        assert result.metric_value is not None
        assert result.metric_value < 0


# =============================================================================
# gate_temporal_boundary Tests
# =============================================================================


class TestGateTemporalBoundary:
    """Tests for temporal boundary gate."""

    def test_pass_on_valid_boundary(self) -> None:
        """Valid boundary should pass."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=102,
            horizon=2,
            extra_gap=0,
        )

        assert result.status == GateStatus.PASS

    def test_halt_on_invalid_boundary(self) -> None:
        """Invalid boundary should halt."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=100,
            horizon=2,
            extra_gap=0,
        )

        assert result.status == GateStatus.HALT
        assert "temporal" in result.name.lower()

    def test_gap_enforcement(self) -> None:
        """Additional gap should be enforced."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=102,
            horizon=2,
            extra_gap=2,  # Need 4 total
        )

        assert result.status == GateStatus.HALT

    def test_details_contain_indices(self) -> None:
        """Details should contain all indices."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=104,
            horizon=2,
            extra_gap=1,
        )

        assert result.details["train_end_idx"] == 99
        assert result.details["test_start_idx"] == 104
        assert result.details["horizon"] == 2
        assert result.details["extra_gap"] == 1


# =============================================================================
# run_gates Tests
# =============================================================================


class TestRunGates:
    """Tests for gate aggregation."""

    def test_aggregates_results(self) -> None:
        """run_gates should aggregate results."""
        results = [
            GateResult(name="g1", status=GateStatus.PASS, message="ok"),
            GateResult(name="g2", status=GateStatus.PASS, message="ok"),
        ]

        report = run_gates(results)

        assert len(report.gates) == 2
        assert report.status == "PASS"

    def test_empty_list(self) -> None:
        """Empty list should produce PASS report."""
        report = run_gates([])
        assert report.status == "PASS"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple gates."""

    def test_full_validation_workflow(self, simple_data: tuple) -> None:
        """Full validation workflow should work."""
        X, y = simple_data
        model = MockModel("mean")

        # Run multiple gates
        gate_results = [
            gate_signal_verification(model, X, y, n_shuffles=2, random_state=42),
            gate_suspicious_improvement(
                model_metric=0.95, baseline_metric=1.0
            ),
            gate_temporal_boundary(
                train_end_idx=79, test_start_idx=82, horizon=2
            ),
        ]

        report = run_gates(gate_results)

        # Should have all gates
        assert len(report.gates) == 3

        # Summary should be readable
        summary = report.summary()
        assert "VALIDATION REPORT" in summary

    def test_leakage_detection_workflow(self) -> None:
        """Leakage should be detected by gates."""
        rng = np.random.default_rng(123)
        n = 100

        # Create data with strong X->y relationship (simulates leakage)
        X = rng.standard_normal((n, 5))
        true_coeffs = np.array([2.0, -1.5, 1.0, 0.5, -0.8])
        y = X @ true_coeffs + rng.standard_normal(n) * 0.1

        model = MockModel("leaky")

        result = gate_signal_verification(
            model, X, y, n_shuffles=3, threshold=0.05,
            method="effect_size",  # Use effect_size mode for this test
            random_state=42
        )

        # Leakage should be caught
        assert result.status == GateStatus.HALT


# =============================================================================
# Bootstrap CI Integration Tests
# =============================================================================


class TestBootstrapCIIntegration:
    """Tests for bootstrap CI integration with gates."""

    def test_shuffled_target_ci_disabled_by_default(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CI should not be computed by default."""
        X, y = simple_data
        model = MockModel("mean")

        result = gate_signal_verification(
            model, X, y, n_shuffles=3, method="effect_size", random_state=42
        )

        assert "ci_lower" not in result.details
        assert "ci_upper" not in result.details

    def test_shuffled_target_ci_enabled(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CI should be computed when bootstrap_ci=True."""
        X, y = simple_data
        model = MockModel("mean")

        result = gate_signal_verification(
            model, X, y,
            n_shuffles=3,
            method="effect_size",
            random_state=42,
            bootstrap_ci=True,
            n_bootstrap=50,
            bootstrap_alpha=0.05,
        )

        assert "ci_lower" in result.details
        assert "ci_upper" in result.details
        assert "ci_alpha" in result.details
        assert "bootstrap_std" in result.details
        assert "n_bootstrap" in result.details
        assert "bootstrap_block_length" in result.details

        # CI should be reasonable
        assert result.details["ci_lower"] < result.details["ci_upper"]
        assert result.details["ci_alpha"] == 0.05
        assert result.details["n_bootstrap"] == 50

    def test_synthetic_ar1_ci_disabled_by_default(self) -> None:
        """CI should not be computed by default for synthetic AR1."""
        model = MockModel("lag1")

        result = gate_synthetic_ar1(
            model, phi=0.9, sigma=1.0, n_samples=100, random_state=42
        )

        assert "ci_lower" not in result.details
        assert "ci_upper" not in result.details

    def test_synthetic_ar1_ci_enabled(self) -> None:
        """CI should be computed when bootstrap_ci=True for synthetic AR1."""
        model = MockModel("lag1")

        result = gate_synthetic_ar1(
            model,
            phi=0.9,
            sigma=1.0,
            n_samples=100,
            random_state=42,
            bootstrap_ci=True,
            n_bootstrap=50,
            bootstrap_alpha=0.10,
        )

        assert "ci_lower" in result.details
        assert "ci_upper" in result.details
        assert "ci_alpha" in result.details
        assert result.details["ci_alpha"] == 0.10

        # CI should be reasonable
        assert result.details["ci_lower"] < result.details["ci_upper"]

    def test_ci_contains_point_estimate(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Point estimate should be within CI for well-behaved data."""
        X, y = simple_data
        model = MockModel("mean")

        result = gate_signal_verification(
            model, X, y,
            n_shuffles=3,
            method="effect_size",
            random_state=42,
            bootstrap_ci=True,
            n_bootstrap=100,
        )

        mae_real = result.details["mae_real"]
        ci_lower = result.details["ci_lower"]
        ci_upper = result.details["ci_upper"]

        # Point estimate should be in CI (or very close)
        assert ci_lower <= mae_real <= ci_upper

    def test_ci_reproducibility(
        self, simple_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Same seed should produce same CI."""
        X, y = simple_data
        model = MockModel("mean")

        result1 = gate_signal_verification(
            model, X, y,
            n_shuffles=3,
            method="effect_size",
            random_state=42,
            bootstrap_ci=True,
            n_bootstrap=50,
        )

        result2 = gate_signal_verification(
            model, X, y,
            n_shuffles=3,
            method="effect_size",
            random_state=42,
            bootstrap_ci=True,
            n_bootstrap=50,
        )

        assert result1.details["ci_lower"] == result2.details["ci_lower"]
        assert result1.details["ci_upper"] == result2.details["ci_upper"]
