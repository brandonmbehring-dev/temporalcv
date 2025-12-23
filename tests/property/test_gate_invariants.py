"""Property-based tests for validation gate invariants.

These tests verify invariants that should ALWAYS hold:
1. Gate composition priority: HALT > WARN > PASS
2. Gate status is always valid enum member
3. GateResult always has required fields
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from temporalcv.gates import (
    GateStatus,
    GateResult,
    run_gates,
    gate_suspicious_improvement,
)


# === Strategies ===

@st.composite
def valid_gate_result(draw):
    """Generate a valid GateResult."""
    status = draw(st.sampled_from(list(GateStatus)))
    return GateResult(
        name=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        status=status,
        message=draw(st.text(max_size=200)),
        details=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()),
            max_size=5
        )),
    )


@st.composite
def gate_result_list(draw, min_size=1, max_size=10):
    """Generate a list of valid GateResults."""
    return draw(st.lists(valid_gate_result(), min_size=min_size, max_size=max_size))


@st.composite
def positive_errors(draw):
    """Generate valid positive error values for suspicious improvement gate."""
    model_error = draw(st.floats(min_value=0.001, max_value=1.0))
    baseline_error = draw(st.floats(min_value=0.001, max_value=1.0))
    return model_error, baseline_error


# === Gate Composition Invariants ===

class TestGateCompositionInvariants:
    """Test that gate aggregation always respects priority order."""

    @given(gate_result_list(min_size=1))
    @settings(max_examples=100)
    def test_halt_dominates_all(self, gates):
        """If any gate is HALT, aggregate must be HALT."""
        report = run_gates(gates)

        has_halt = any(g.status == GateStatus.HALT for g in gates)
        if has_halt:
            assert report.status == "HALT"

    @given(gate_result_list(min_size=1))
    @settings(max_examples=100)
    def test_warn_dominates_pass(self, gates):
        """If no HALT and any WARN, aggregate must be WARN."""
        report = run_gates(gates)

        has_halt = any(g.status == GateStatus.HALT for g in gates)
        has_warn = any(g.status == GateStatus.WARN for g in gates)

        if not has_halt and has_warn:
            assert report.status == "WARN"

    @given(gate_result_list(min_size=1))
    @settings(max_examples=100)
    def test_pass_only_when_all_pass(self, gates):
        """PASS status only when all gates are PASS or SKIP."""
        report = run_gates(gates)

        all_pass_or_skip = all(
            g.status in (GateStatus.PASS, GateStatus.SKIP) for g in gates
        )
        has_pass = any(g.status == GateStatus.PASS for g in gates)

        if all_pass_or_skip and has_pass:
            assert report.status == "PASS"

    @given(gate_result_list(min_size=1))
    @settings(max_examples=100)
    def test_report_contains_all_gates(self, gates):
        """Aggregate report should contain all input gates."""
        report = run_gates(gates)
        assert len(report.gates) == len(gates)


# === Suspicious Improvement Gate Invariants ===

class TestSuspiciousImprovementInvariants:
    """Test that suspicious improvement gate behaves correctly."""

    @given(positive_errors())
    @settings(max_examples=100)
    def test_zero_or_negative_improvement_never_halts(self, errors):
        """If model is worse or equal, should never HALT."""
        model_metric, baseline_metric = errors
        assume(model_metric >= baseline_metric)  # No improvement

        result = gate_suspicious_improvement(
            model_metric=model_metric,
            baseline_metric=baseline_metric,
            threshold=0.20,
            warn_threshold=0.10,
        )

        assert result.status != GateStatus.HALT

    @given(st.floats(min_value=0.001, max_value=0.5))
    @settings(max_examples=50)
    def test_extreme_improvement_always_halts(self, baseline):
        """If improvement is >90%, should always HALT (threshold is 20%)."""
        model_metric = baseline * 0.05  # 95% improvement
        baseline_metric = baseline

        result = gate_suspicious_improvement(
            model_metric=model_metric,
            baseline_metric=baseline_metric,
            threshold=0.20,
        )

        assert result.status == GateStatus.HALT

    @given(
        st.floats(min_value=0.001, max_value=0.5),
        st.floats(min_value=0.05, max_value=0.50),
    )
    @settings(max_examples=50)
    def test_threshold_respected(self, baseline, threshold):
        """Gate should respect the threshold parameter."""
        # Create model with exactly threshold+1% improvement
        improvement = threshold + 0.01
        model_metric = baseline * (1 - improvement)

        result = gate_suspicious_improvement(
            model_metric=model_metric,
            baseline_metric=baseline,
            threshold=threshold,
        )

        assert result.status == GateStatus.HALT


# === GateResult Invariants ===

class TestGateResultInvariants:
    """Test that GateResult always has valid structure."""

    @given(valid_gate_result())
    @settings(max_examples=100)
    def test_status_is_valid_enum(self, gate_result):
        """Status must always be a valid GateStatus enum."""
        assert isinstance(gate_result.status, GateStatus)

    @given(valid_gate_result())
    @settings(max_examples=100)
    def test_name_is_string(self, gate_result):
        """Gate name must always be a non-empty string."""
        assert isinstance(gate_result.name, str)
        assert len(gate_result.name.strip()) > 0

    @given(valid_gate_result())
    @settings(max_examples=100)
    def test_message_is_string(self, gate_result):
        """Message must always be a string."""
        assert isinstance(gate_result.message, str)


# === Priority Order Invariants ===

class TestPriorityOrderInvariants:
    """Test that priority order is consistent."""

    def test_enum_order_is_halt_warn_pass_skip(self):
        """Verify the enum defines correct priority order."""
        # HALT should have highest priority (lowest value if ordered)
        statuses = [GateStatus.HALT, GateStatus.WARN, GateStatus.PASS, GateStatus.SKIP]

        # Create gates with each status
        gates = [
            GateResult(name=f"gate_{i}", status=s, message="test")
            for i, s in enumerate(statuses)
        ]

        # Verify HALT dominates when present
        assert run_gates(gates).status == "HALT"

        # Remove HALT, WARN should dominate
        assert run_gates(gates[1:]).status == "WARN"

        # Remove WARN, PASS should dominate
        assert run_gates(gates[2:]).status == "PASS"

        # Only SKIP - note: when only skips, returns PASS (nothing failed)
        # This is correct behavior - SKIP doesn't indicate failure
        assert run_gates([gates[3]]).status in ("PASS", "SKIP")
