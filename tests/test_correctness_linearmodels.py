"""Analytical correctness tests for linearmodels adapters.

For linear models (OLS, PanelOLS, IV2SLS, AbsorbingLS), marginal effects
are analytically equal to the coefficients.  These tests verify that
pymargins recovers the coefficient values within float32 finite-difference
tolerance via dydx(), predict(), and contrasts().
"""

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS, AbsorbingLS
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

from pymargins import GComputation
from pymargins.scenarios import pairwise

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def panel_df():
    """Small panel dataset with known structure."""
    np.random.seed(42)
    n = 200
    t = 5
    entities = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)
    x1 = np.random.randn(n * t)
    x2 = np.random.randn(n * t)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + 0.5 * entities + np.random.randn(n * t) * 0.5
    df = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    ).set_index(["entity", "time"])
    return df


@pytest.fixture
def iv_df():
    """Cross-sectional dataset for IV."""
    np.random.seed(42)
    n = 500
    z = np.random.randn(n)
    w = np.random.randn(n)
    x = 0.5 * z + 0.3 * w + np.random.randn(n) * 0.3
    y = 1.0 + 2.0 * x + np.random.randn(n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "w": w})
    return df


@pytest.fixture
def absorb_df():
    """Dataset for high-dimensional fixed effects."""
    np.random.seed(42)
    n = 300
    entities = np.random.randint(0, 50, n)
    times = np.random.randint(0, 5, n)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + 0.5 * entities + np.random.randn(n) * 0.5
    df = pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "entity": entities,
            "time": times,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fd_slope(beta, X, eps, col_idx):
    """Finite-difference slope ground truth in float32."""
    Xp = np.array(X, dtype=np.float32)
    Xm = np.array(X, dtype=np.float32)
    Xp[:, col_idx] += eps
    Xm[:, col_idx] -= eps
    mu_p = Xp @ np.array(beta, dtype=np.float32)
    mu_m = Xm @ np.array(beta, dtype=np.float32)
    return float(np.mean((mu_p - mu_m) / (2.0 * eps)))


# ---------------------------------------------------------------------------
# PooledOLS
# ---------------------------------------------------------------------------


def test_pooledols_dydx_matches_coefficient_within_fd_tolerance(panel_df):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    slope_x1 = m.dydx("x1")
    slope_x2 = m.dydx("x2")

    # Float32 finite-difference can introduce ~0.1% relative error.
    assert np.isclose(float(slope_x1.estimate), float(res.params["x1"]), rtol=2e-2)
    assert np.isclose(float(slope_x2.estimate), float(res.params["x2"]), rtol=2e-2)


def test_pooledols_predict_at_mean_matches_native(panel_df):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    pred = m.predict()
    # PooledOLS predict() expects exog with MultiIndex
    native_pred = res.predict(exog=panel_df[["x1", "x2"]])
    assert np.isclose(
        float(pred.estimate), float(native_pred.mean().iloc[0]), rtol=1e-4
    )


def test_pooledols_pairwise_contrast_matches_manual(panel_df):
    mod = PooledOLS.from_formula("y ~ x1 + x2", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    x1_mean = panel_df["x1"].mean()
    x1_sd = panel_df["x1"].std()

    scenarios, contrasts = pairwise("x1", [x1_mean + x1_sd, x1_mean - x1_sd])
    contrast_res = m.contrasts(scenarios=scenarios, contrasts=contrasts)

    # For a linear model, the contrast in predictions is exactly beta * delta_x
    expected = float(res.params["x1"]) * (2 * x1_sd)
    assert np.isclose(float(contrast_res.estimate), expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# PanelOLS (with FE)
# ---------------------------------------------------------------------------


def test_panelols_dydx_matches_coefficient_within_fd_tolerance(panel_df):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    slope_x1 = m.dydx("x1")
    slope_x2 = m.dydx("x2")

    assert np.isclose(float(slope_x1.estimate), float(res.params["x1"]), rtol=2e-2)
    assert np.isclose(float(slope_x2.estimate), float(res.params["x2"]), rtol=2e-2)


def test_panelols_predict_at_mean_matches_native(panel_df):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    pred = m.predict()
    native_pred = res.predict(exog=panel_df[["x1", "x2"]])
    assert np.isclose(
        float(pred.estimate), float(native_pred.mean().iloc[0]), rtol=1e-4
    )


def test_panelols_pairwise_contrast_matches_manual(panel_df):
    mod = PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    x1_mean = panel_df["x1"].mean()
    x1_sd = panel_df["x1"].std()

    scenarios, contrasts = pairwise("x1", [x1_mean + x1_sd, x1_mean - x1_sd])
    contrast_res = m.contrasts(scenarios=scenarios, contrasts=contrasts)

    expected = float(res.params["x1"]) * (2 * x1_sd)
    assert np.isclose(float(contrast_res.estimate), expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# RandomEffects
# ---------------------------------------------------------------------------


def test_random_effects_dydx_matches_coefficient_within_fd_tolerance(panel_df):
    mod = RandomEffects.from_formula("y ~ x1 + x2", data=panel_df)
    res = mod.fit()
    m = GComputation(res)

    slope_x1 = m.dydx("x1")
    slope_x2 = m.dydx("x2")

    assert np.isclose(float(slope_x1.estimate), float(res.params["x1"]), rtol=2e-2)
    assert np.isclose(float(slope_x2.estimate), float(res.params["x2"]), rtol=2e-2)


# ---------------------------------------------------------------------------
# IV2SLS
# ---------------------------------------------------------------------------


def test_iv2sls_dydx_matches_coefficient_within_fd_tolerance(iv_df):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_df)
    res = mod.fit()
    from pymargins._adapters.linearmodels_iv import LinearmodelsIVAdapter

    adapter = LinearmodelsIVAdapter(res, training_data=iv_df)
    m = GComputation(adapter=adapter)

    slope_x = m.dydx("x")
    assert np.isclose(float(slope_x.estimate), float(res.params["x"]), rtol=2e-2)


def test_iv2sls_predict_at_mean_matches_native(iv_df):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_df)
    res = mod.fit()
    from pymargins._adapters.linearmodels_iv import LinearmodelsIVAdapter

    adapter = LinearmodelsIVAdapter(res, training_data=iv_df)
    m = GComputation(adapter=adapter)

    pred = m.predict()
    native_pred = res.predict(data=iv_df)
    assert np.isclose(
        float(pred.estimate), float(native_pred.mean().iloc[0]), rtol=1e-4
    )


def test_iv2sls_pairwise_contrast_matches_manual(iv_df):
    mod = IV2SLS.from_formula("y ~ 1 + [x ~ z + w]", data=iv_df)
    res = mod.fit()
    from pymargins._adapters.linearmodels_iv import LinearmodelsIVAdapter

    adapter = LinearmodelsIVAdapter(res, training_data=iv_df)
    m = GComputation(adapter=adapter)

    x_mean = iv_df["x"].mean()
    x_sd = iv_df["x"].std()

    scenarios, contrasts = pairwise("x", [x_mean + x_sd, x_mean - x_sd])
    contrast_res = m.contrasts(scenarios=scenarios, contrasts=contrasts)

    expected = float(res.params["x"]) * (2 * x_sd)
    assert np.isclose(float(contrast_res.estimate), expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# AbsorbingLS
# ---------------------------------------------------------------------------


def test_absorbingls_dydx_matches_coefficient_within_fd_tolerance(absorb_df):
    mod = AbsorbingLS(
        absorb_df[["y"]],
        absorb_df[["x1", "x2"]],
        absorb=absorb_df[["entity", "time"]],
    )
    res = mod.fit()
    from pymargins._adapters.linearmodels_absorbing import LinearmodelsAbsorbingAdapter

    adapter = LinearmodelsAbsorbingAdapter(res)
    m = GComputation(adapter=adapter)

    slope_x1 = m.dydx("x1")
    slope_x2 = m.dydx("x2")

    assert np.isclose(float(slope_x1.estimate), float(res.params["x1"]), rtol=2e-2)
    assert np.isclose(float(slope_x2.estimate), float(res.params["x2"]), rtol=2e-2)


def test_absorbingls_pairwise_contrast_matches_manual(absorb_df):
    mod = AbsorbingLS(
        absorb_df[["y"]],
        absorb_df[["x1", "x2"]],
        absorb=absorb_df[["entity", "time"]],
    )
    res = mod.fit()
    from pymargins._adapters.linearmodels_absorbing import LinearmodelsAbsorbingAdapter

    adapter = LinearmodelsAbsorbingAdapter(res)
    m = GComputation(adapter=adapter)

    x1_mean = absorb_df["x1"].mean()
    x1_sd = absorb_df["x1"].std()

    scenarios, contrasts = pairwise("x1", [x1_mean + x1_sd, x1_mean - x1_sd])
    contrast_res = m.contrasts(scenarios=scenarios, contrasts=contrasts)

    expected = float(res.params["x1"]) * (2 * x1_sd)
    assert np.isclose(float(contrast_res.estimate), expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# Cross-model sanity: all linear adapters agree on AME ≈ coefficient
# ---------------------------------------------------------------------------


def test_all_linear_adapters_ame_equals_coefficient_within_tolerance(
    panel_df, iv_df, absorb_df
):
    """For any linear model, the AME of x1 should approximate its coefficient."""
    models = [
        ("PooledOLS", PooledOLS.from_formula("y ~ x1 + x2", data=panel_df).fit(), None),
        (
            "PanelOLS",
            PanelOLS.from_formula("y ~ x1 + x2 + EntityEffects", data=panel_df).fit(),
            None,
        ),
        (
            "RandomEffects",
            RandomEffects.from_formula("y ~ x1 + x2", data=panel_df).fit(),
            None,
        ),
    ]

    for name, res, training_data in models:
        if training_data is None:
            m = GComputation(res)
        else:
            from pymargins._adapters.linearmodels_iv import LinearmodelsIVAdapter

            adapter = LinearmodelsIVAdapter(res, training_data=training_data)
            m = GComputation(adapter=adapter)

        slope = m.dydx("x1")
        coef = float(res.params["x1"])
        assert np.isclose(float(slope.estimate), coef, rtol=2e-2), (
            f"{name} AME mismatch: {float(slope.estimate):.6f} vs {coef:.6f}"
        )
