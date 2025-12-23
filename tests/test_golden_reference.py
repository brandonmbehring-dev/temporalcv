"""
Golden reference tests - verify outputs match pre-computed reference values.

These tests use frozen reference data to catch regressions in statistical
computations. Reference values were computed using R's forecast package
for DM test validation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from temporalcv.statistical_tests import dm_test
from temporalcv.inference.wild_bootstrap import wild_cluster_bootstrap


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def golden_refs():
    """Load golden reference data."""
    fixture_path = Path(__file__).parent / "fixtures" / "golden_reference.json"
    with open(fixture_path) as f:
        return json.load(f)


# =============================================================================
# DM Test Golden Reference
# =============================================================================


class TestDMGoldenReference:
    """Verify DM test against pre-computed reference values."""

    def test_dm_case_001_equal_errors(self, golden_refs):
        """Equal forecasters should not reject null."""
        case = golden_refs["dm_test"]["case_001_equal_errors"]
        inputs = case["inputs"]
        expected = case["expected"]

        result = dm_test(
            np.array(inputs["errors1"]),
            np.array(inputs["errors2"]),
            h=inputs["h"],
            loss=inputs["loss"],
            harvey_correction=inputs["harvey_correction"],
        )

        # Check p-value is in expected range (not rejecting null)
        assert expected["pvalue_range"][0] <= result.pvalue <= expected["pvalue_range"][1], (
            f"p-value {result.pvalue:.4f} not in expected range {expected['pvalue_range']}"
        )

    def test_dm_case_002_model1_better(self, golden_refs):
        """Model 1 clearly better should reject null."""
        case = golden_refs["dm_test"]["case_002_model1_better"]
        inputs = case["inputs"]
        expected = case["expected"]

        result = dm_test(
            np.array(inputs["errors1"]),
            np.array(inputs["errors2"]),
            h=inputs["h"],
            loss=inputs["loss"],
            harvey_correction=inputs["harvey_correction"],
        )

        # Should reject null (model 1 is better)
        assert result.pvalue < 0.05, (
            f"p-value {result.pvalue:.4f} should be < 0.05 (model 1 is clearly better)"
        )

        # Statistic should be negative (model 1 has lower loss)
        assert result.statistic < 0, (
            f"Statistic {result.statistic:.4f} should be negative (model 1 is better)"
        )

    def test_dm_case_003_multistep(self, golden_refs):
        """Multi-step HAC-adjusted test."""
        case = golden_refs["dm_test"]["case_003_multistep"]
        inputs = case["inputs"]
        expected = case["expected"]

        result = dm_test(
            np.array(inputs["errors1"]),
            np.array(inputs["errors2"]),
            h=inputs["h"],
            loss=inputs["loss"],
            harvey_correction=inputs["harvey_correction"],
        )

        # Check p-value is in expected range
        assert expected["pvalue_range"][0] <= result.pvalue <= expected["pvalue_range"][1], (
            f"p-value {result.pvalue:.4f} not in expected range {expected['pvalue_range']}"
        )


# =============================================================================
# Wild Bootstrap Golden Reference
# =============================================================================


class TestWildBootstrapGoldenReference:
    """Verify wild bootstrap against pre-computed reference values."""

    def test_wild_case_001_positive_mean(self, golden_refs):
        """Positive mean fold statistics should reject H0: mean=0."""
        case = golden_refs["wild_bootstrap"]["case_001_positive_mean"]
        inputs = case["inputs"]
        expected = case["expected"]

        result = wild_cluster_bootstrap(
            np.array(inputs["fold_statistics"]),
            n_bootstrap=inputs["n_bootstrap"],
            random_state=inputs["random_state"],
        )

        # Check estimate matches expected
        tol = case["tolerance"]["estimate"]
        assert abs(result.estimate - expected["estimate"]) < tol, (
            f"Estimate {result.estimate:.4f} not within {tol} of expected {expected['estimate']}"
        )

        # Check p-value indicates rejection
        assert result.p_value < 0.05, (
            f"p-value {result.p_value:.4f} should be < 0.05 for clearly positive mean"
        )

    def test_wild_case_002_zero_mean(self, golden_refs):
        """Zero-centered fold statistics should not reject H0."""
        case = golden_refs["wild_bootstrap"]["case_002_zero_mean"]
        inputs = case["inputs"]
        expected = case["expected"]

        result = wild_cluster_bootstrap(
            np.array(inputs["fold_statistics"]),
            n_bootstrap=inputs["n_bootstrap"],
            random_state=inputs["random_state"],
        )

        # Check estimate is near zero
        assert abs(result.estimate) < 0.1, (
            f"Estimate {result.estimate:.4f} should be near zero"
        )

        # Check p-value indicates non-rejection
        assert result.p_value > 0.10, (
            f"p-value {result.p_value:.4f} should be > 0.10 for zero-mean data"
        )
