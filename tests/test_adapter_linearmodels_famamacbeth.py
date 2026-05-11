"""Tests for linearmodels FamaMacBeth adapter."""

import numpy as np
import pandas as pd
import pytest

from linearmodels.panel import FamaMacBeth

from pymargins import Margins
from pymargins._adapters.linearmodels_panel import LinearmodelsPanelAdapter


@pytest.fixture
def panel_df():
    np.random.seed(42)
    n = 200
    t = 5
    entities = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)
    x1 = np.random.randn(n * t)
    y = 1.0 + 2.0 * x1 + np.random.randn(n * t) * 0.5
    return pd.DataFrame({
        "entity": entities,
        "time": times,
        "y": y,
        "x1": x1,
    }).set_index(["entity", "time"])


def test_auto_detect_famamacbeth(panel_df):
    mod = FamaMacBeth.from_formula("y ~ x1", data=panel_df)
    res = mod.fit()
    m = Margins(res)
    assert isinstance(m.adapter, LinearmodelsPanelAdapter)


def test_margins_predict(panel_df):
    mod = FamaMacBeth.from_formula("y ~ x1", data=panel_df)
    res = mod.fit()
    m = Margins(res)

    pred = m.predict()
    native_pred = res.predict(exog=panel_df[["x1"]])
    assert np.isclose(float(pred.estimate), float(native_pred.mean().iloc[0]), rtol=1e-4)


def test_margins_dydx(panel_df):
    mod = FamaMacBeth.from_formula("y ~ x1", data=panel_df)
    res = mod.fit()
    m = Margins(res)

    slope = m.dydx("x1")
    assert np.isclose(float(slope.estimate), float(res.params["x1"]), rtol=2e-2)
