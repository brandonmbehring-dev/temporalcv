"""Tests for the v2.0 result-object contract on cv.py result types.

Every result object is a frozen, slotted value object with an explicit JSON-serializable
``to_dict()`` carrying a ``schema_version`` (the locked playbook pattern). See
``docs/adr/0001-v2-seams-and-layout.md`` and hub ``library-design-playbook.md``.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import numpy as np
import pytest

from temporalcv.cv import NestedCVResult, SplitInfo, SplitResult, WalkForwardResults


def _split_result(idx: int = 0) -> SplitResult:
    return SplitResult(
        split_idx=idx,
        train_start=0,
        train_end=9,
        test_start=11,
        test_end=15,
        predictions=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        actuals=np.array([1.1, 1.9, 3.2, 3.8, 5.1]),
    )


def _nested_result() -> NestedCVResult:
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


ALL_RESULTS = [
    SplitInfo(split_idx=0, train_start=0, train_end=9, test_start=11, test_end=15),
    _split_result(),
    WalkForwardResults(splits=[_split_result()], cv_config={"n_splits": 5}),
    _nested_result(),
]


class TestFrozenSlotted:
    """Result objects are immutable value objects without a per-instance __dict__."""

    @pytest.mark.parametrize("obj", ALL_RESULTS)
    def test_is_frozen(self, obj: object) -> None:
        field_name = next(iter(dataclasses.fields(obj))).name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field_name, None)

    @pytest.mark.parametrize("obj", ALL_RESULTS)
    def test_is_slotted(self, obj: object) -> None:
        assert hasattr(type(obj), "__slots__")
        assert not hasattr(obj, "__dict__")

    @pytest.mark.parametrize("obj", ALL_RESULTS)
    def test_cannot_add_new_attribute(self, obj: object) -> None:
        # Adding an undeclared attribute must be blocked. Note: for
        # @dataclass(frozen=True, slots=True) CPython raises TypeError (the generated
        # frozen __setattr__ reaches a stale-class super() for non-field names) rather
        # than FrozenInstanceError — either way the assignment is rejected, which is the
        # contract we care about. Declared-field immutability (test_is_frozen) is the
        # clean FrozenInstanceError path.
        with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            obj.some_undeclared_attr = 1  # type: ignore[attr-defined]


class TestToDict:
    """to_dict() is JSON-serializable and carries a schema_version."""

    @pytest.mark.parametrize("obj", ALL_RESULTS)
    def test_to_dict_has_schema_version(self, obj: object) -> None:
        d = obj.to_dict()  # type: ignore[attr-defined]
        assert d["schema_version"] == type(obj).SCHEMA_VERSION  # type: ignore[attr-defined]
        assert isinstance(d["schema_version"], int)

    @pytest.mark.parametrize("obj", ALL_RESULTS)
    def test_to_dict_is_json_serializable(self, obj: object) -> None:
        # Must not raise: arrays must already be converted to lists, etc.
        json.dumps(obj.to_dict())  # type: ignore[attr-defined]

    def test_split_result_to_dict_arrays_become_lists(self) -> None:
        d = _split_result().to_dict()
        assert isinstance(d["predictions"], list)
        assert isinstance(d["actuals"], list)
        assert d["predictions"] == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_walkforward_to_dict_nests_split_dicts(self) -> None:
        wfr = WalkForwardResults(splits=[_split_result(0), _split_result(1)], cv_config=None)
        d = wfr.to_dict()
        assert d["n_splits"] == 2
        assert len(d["splits"]) == 2
        assert all(s["schema_version"] == SplitResult.SCHEMA_VERSION for s in d["splits"])

    def test_nested_to_dict_outer_scores_list(self) -> None:
        d = _nested_result().to_dict()
        assert isinstance(d["outer_scores"], list)
        assert d["best_params"] == {"alpha": 1.0}

    def test_dates_serialize_to_iso(self) -> None:
        # The datetime -> ISO branch of _date_to_json (review R3: previously 0 coverage).
        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=9,
            test_start=11,
            test_end=15,
            train_start_date=datetime(2020, 1, 1),
            test_end_date=datetime(2020, 3, 1),
        )
        d = info.to_dict()
        assert d["train_start_date"] == "2020-01-01T00:00:00"
        assert d["test_end_date"] == "2020-03-01T00:00:00"
        assert d["train_end_date"] is None  # passthrough for unset dates
        json.dumps(d)  # must remain JSON-serializable with dates present


class TestValuePreservation:
    """to_dict() preserves the underlying values."""

    def test_splitinfo_roundtrip_values(self) -> None:
        info = SplitInfo(split_idx=2, train_start=0, train_end=49, test_start=52, test_end=61)
        d = info.to_dict()
        assert d["split_idx"] == 2
        assert d["train_size"] == 50
        assert d["test_size"] == 10
        assert d["gap"] == 2
