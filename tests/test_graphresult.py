"""Tests for GraphResult (W2.6)."""

from __future__ import annotations

import numpy as np
import pytest

from pymargins._graph._plan import Plan
from pymargins._result._graphresult import GraphResult
from pymargins._result._margins import MarginsResult


def _make_margins_result(**kwargs):
    defaults = {
        "estimate": np.array([1.0]),
        "std_error": np.array([0.1]),
        "conf_int_lower": np.array([0.8]),
        "conf_int_upper": np.array([1.2]),
        "method": "delta",
        "level": 0.95,
        "n_obs": 100,
        "gradient": np.array([1.0]),
        "cov_params": np.array([[0.01]]),
    }
    defaults.update(kwargs)
    return MarginsResult(**defaults)


def test_conf_int_no_level_parameter():
    plan = Plan()
    mr = _make_margins_result()
    gr = GraphResult(mr, plan)
    # Calling with level should raise TypeError
    with pytest.raises(TypeError):
        gr.conf_int(level=0.90)


def test_summary_contains_plan_hash():
    plan = Plan()
    mr = _make_margins_result()
    gr = GraphResult(mr, plan)
    summary = gr.summary()
    assert plan.hash in summary


def test_conf_int_bonferroni_widens():
    """Bonferroni correction must widen intervals monotonically."""
    plan = Plan()
    mr = _make_margins_result(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.2]),
        conf_int_lower=np.array([0.8, 1.6]),
        conf_int_upper=np.array([1.2, 2.4]),
        gradient=np.array([[1.0, 0.0], [0.0, 1.0]]),
        cov_params=np.array([[0.01, 0.0], [0.0, 0.04]]),
    )
    gr = GraphResult(mr, plan)
    lo_none, hi_none = gr.conf_int()
    lo_bonf, hi_bonf = gr.conf_int(correction="bonferroni")
    # Bonferroni should widen (lower lower, higher upper)
    assert np.all(lo_bonf <= lo_none)
    assert np.all(hi_bonf >= hi_none)
    # And strictly for at least one component
    assert np.any(lo_bonf < lo_none) or np.any(hi_bonf > hi_none)


def test_conf_int_sidak_widens():
    """Sidak correction must widen intervals monotonically."""
    plan = Plan()
    mr = _make_margins_result(
        estimate=np.array([1.0, 2.0]),
        std_error=np.array([0.1, 0.2]),
        conf_int_lower=np.array([0.8, 1.6]),
        conf_int_upper=np.array([1.2, 2.4]),
        gradient=np.array([[1.0, 0.0], [0.0, 1.0]]),
        cov_params=np.array([[0.01, 0.0], [0.0, 0.04]]),
    )
    gr = GraphResult(mr, plan)
    lo_none, hi_none = gr.conf_int()
    lo_sidak, hi_sidak = gr.conf_int(correction="sidak")
    assert np.all(lo_sidak <= lo_none)
    assert np.all(hi_sidak >= hi_none)


def test_conf_int_level_rejected():
    """conf_int(level=...) must raise TypeError on the doctrine surface."""
    plan = Plan()
    mr = _make_margins_result()
    gr = GraphResult(mr, plan)
    with pytest.raises(TypeError):
        gr.conf_int(level=0.90)


def test_round_trip_disk():
    import os
    import tempfile

    plan = Plan(method_resolved="delta")
    mr = _make_margins_result()
    gr = GraphResult(mr, plan)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "result.pkl")
        gr.to_disk(path)
        gr2 = GraphResult.from_disk(path)
        np.testing.assert_array_equal(gr2.estimate, gr.estimate)
        assert gr2._plan.hash == gr._plan.hash
