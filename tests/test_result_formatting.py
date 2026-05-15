"""Tests for MarginsResult formatting: summary, to_latex, to_html."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins


@pytest.fixture
def df_logit():
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treatment": rng.binomial(1, 0.5, size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })
    lp = -2.0 + 0.05 * df["age"] + 0.8 * df["treatment"] + 0.3 * (df["sex"] == "M")
    df["outcome"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    return df


@pytest.fixture
def fit_logit(df_logit):
    return smf.glm(
        "outcome ~ age + treatment + C(sex)",
        data=df_logit,
        family=sm.families.Binomial(),
    ).fit()


@pytest.fixture
def df_ols():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "age": rng.normal(50, 10, size=n),
        "treatment": rng.binomial(1, 0.5, size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })
    df["y"] = (
        10.0
        + 0.2 * df["age"]
        + 3.0 * df["treatment"]
        + 1.5 * (df["sex"] == "M")
        + rng.standard_normal(n) * 2.0
    )
    return df


@pytest.fixture
def fit_ols(df_ols):
    return smf.ols("y ~ age + treatment + C(sex)", data=df_ols).fit()


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------

def test_summary_contains_title_and_columns(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    s = pred.summary()
    assert "Margins Result" in s
    assert "delta" in s
    assert "estimate" in s
    assert "std err" in s
    assert "z" in s
    assert "P>|z|" in s
    assert "[95% Conf. Int.]" in s
    assert "=" in s
    assert "-" in s


def test_summary_shows_data_rows(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    s = pred.summary()
    assert "treatment=0" in s
    assert "treatment=1" in s


def test_summary_stars(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    s = pred.summary(stars=True)
    # Predictions are far from zero, so p-values are ~0 -> should see ***
    assert "***" in s


def test_summary_no_stars_by_default(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    s = pred.summary()
    assert "***" not in s


def test_summary_truncation(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    s = pred.summary(max_rows=1)
    assert "..." in s


def test_summary_footer_kappa_and_fallback(fit_logit):
    # Force high kappa to trigger fallback
    m = Margins.log_scale(fit_logit, kappa_threshold=0.0)
    rr = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}, "label": "treated"},
            {"atexog": {"treatment": 0}, "label": "control"},
        ],
        contrasts=[+1, -1],
    )
    s = rr.summary()
    if rr.fallback_triggered:
        assert "WARNING" in s
        assert "Fallback" in s
    if rr.kappa is not None:
        assert "κ:" in s


def test_summary_scale_note_for_non_identity(fit_logit):
    m = Margins.log_scale(fit_logit)
    pred = m.predict()
    s = pred.summary()
    assert "inference scale" in s
    assert "reporting scale" in s


def test_summary_for_dydx(fit_ols):
    m = Margins.linear_scale(fit_ols, at="overall")
    ame = m.dydx("age")
    s = ame.summary()
    assert "Margins Result" in s
    assert "[0]" in s or "age" in s


# ---------------------------------------------------------------------------
# to_latex()
# ---------------------------------------------------------------------------

def test_to_latex_basic_structure(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    latex = pred.to_latex()
    assert r"\begin{tabular}" in latex
    assert r"\end{tabular}" in latex
    assert r"\hline" in latex
    assert "estimate" in latex
    assert "std err" in latex


def test_to_latex_with_caption_and_label(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    latex = pred.to_latex(caption="My Caption", label="tab:test")
    assert r"\begin{table}" in latex
    assert r"\caption{My Caption}" in latex
    assert r"\label{tab:test}" in latex
    assert r"\end{table}" in latex


def test_to_latex_stars(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    latex = pred.to_latex(stars=True)
    assert "***" in latex


# ---------------------------------------------------------------------------
# to_html()
# ---------------------------------------------------------------------------

def test_to_html_basic_structure(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    html = pred.to_html()
    assert "<table" in html
    assert "</table>" in html
    assert "<thead>" in html
    assert "<tbody>" in html
    assert "estimate" in html
    assert "std err" in html


def test_to_html_with_caption(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    html = pred.to_html(caption="My Caption")
    assert "<caption>My Caption</caption>" in html


def test_to_html_stars(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    html = pred.to_html(stars=True)
    assert "***" in html


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_materialized_result_summary_still_works(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    mat = pred.materialize()
    s = mat.summary()
    assert "Margins Result" in s
    # Materialized results lack gradient/draws, so z/p columns may be absent


def test_summary_custom_float_fmt(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    s = pred.summary(float_fmt=".2f")
    # Check that estimates are formatted to 2 decimals
    import re
    matches = re.findall(r"\d+\.\d{2}", s)
    assert len(matches) > 0


def test_materialized_to_frame_has_no_p_value(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    mat = pred.materialize()
    frame = mat.to_frame()
    assert "statistic" not in frame.columns
    assert "p_value" not in frame.columns


def test_materialized_test_raises(fit_ols):
    m = Margins.linear_scale(fit_ols, at="typical")
    pred = m.predict(atexog={"treatment": [0, 1]})
    mat = pred.materialize()
    with pytest.raises(ValueError):
        mat.test()


def test_summary_with_2d_estimate(fit_logit):
    """Summary must work when estimate is 2D (multi-outcome x multi-scenario)."""
    m = Margins.linear_scale(fit_logit, at="overall")
    res = m.predict(atexog={"age": [25, 45, 65], "treatment": [0, 1]})
    # This produces a 2D estimate (6 scenarios)
    s = res.summary()
    assert isinstance(s, str)
    assert "age=25" in s or "estimate" in s
