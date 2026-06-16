"""Tests for bootstrap progress bar."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import GComputation


@pytest.fixture
def fit_ols():
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({"x": rng.standard_normal(n), "y": rng.standard_normal(n)})
    return smf.ols("y ~ x", data=df).fit()


def test_progress_bar_true_runs(fit_ols, capsys):
    """progress_bar=True should run without error and produce tqdm output."""
    m = GComputation(fit_ols, method="bootstrap", B=5, seed=42, progress_bar=True)
    result = m.predict(atexog={"x": 0})
    assert np.isfinite(result.estimate)
    captured = capsys.readouterr()
    assert "Bootstrap refit" in captured.err


def test_progress_bar_false_runs(fit_ols, capsys):
    """progress_bar=False should run without error and not produce tqdm output."""
    m = GComputation(fit_ols, method="bootstrap", B=5, seed=42, progress_bar=False)
    result = m.predict(atexog={"x": 0})
    assert np.isfinite(result.estimate)
    captured = capsys.readouterr()
    assert "Bootstrap refit" not in captured.err


def test_progress_bar_default_is_false(fit_ols):
    """Default progress_bar should be False."""
    m = GComputation(fit_ols)
    assert m._progress_bar is False
