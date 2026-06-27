"""Oracle test fixtures and assertion helpers.

Design §4.2, req §7. Added in 0.4.0 (R1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from ._tolerances import TOL_COEF, TOL_EST, TOL_SE

GOLDEN_DIR = Path(__file__).parent / "golden"
DATA_DIR = Path(__file__).parent / "data"


def load_golden(case_id: str) -> dict[str, Any]:
    """Load one R-generated golden JSON."""
    path = GOLDEN_DIR / f"{case_id}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def oracle_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "oracle_main.csv")


@pytest.fixture(scope="session")
def fit_ols(oracle_df: pd.DataFrame):
    return smf.ols("y_cont ~ treat + x1 + x2", data=oracle_df).fit()


@pytest.fixture(scope="session")
def fit_logit(oracle_df: pd.DataFrame):
    return smf.glm(
        "y_bin ~ treat + x1 + x2",
        data=oracle_df,
        family=sm.families.Binomial(),
    ).fit(tol=1e-12, maxiter=200)


@pytest.fixture(scope="session")
def fit_logit_weighted(oracle_df: pd.DataFrame):
    return smf.glm(
        "y_bin ~ treat + x1 + x2",
        data=oracle_df,
        family=sm.families.Binomial(),
        freq_weights=oracle_df["w"].values,
    ).fit(tol=1e-12, maxiter=200)


@pytest.fixture(scope="session")
def fit_probit(oracle_df: pd.DataFrame):
    return smf.probit("y_bin ~ treat + x1 + x2", data=oracle_df).fit(
        tol=1e-12, maxiter=200
    )


@pytest.fixture(scope="session")
def fit_poisson(oracle_df: pd.DataFrame):
    return smf.glm(
        "y_count ~ treat + x1 + x2",
        data=oracle_df,
        family=sm.families.Poisson(),
    ).fit(tol=1e-12, maxiter=200)


def assert_coef_aligned(fit, golden: dict[str, Any]) -> None:
    """Fit-alignment gate. A failure here is misalignment, NOT a defect."""
    np.testing.assert_allclose(
        np.asarray(fit.params),
        np.asarray(golden["quantities"]["coefficients"]),
        rtol=TOL_COEF,
        atol=0.0,
    )


def assert_matches_golden(
    golden: dict[str, Any],
    *,
    estimate=None,
    std_error=None,
    conf_low=None,
    conf_high=None,
) -> None:
    """Compare computed quantities to a golden, respecting per-case overrides."""
    tol = golden.get("tolerances", {})
    est_tol = tol.get("estimate", TOL_EST)
    se_tol = tol.get("std_error", TOL_SE)
    # When SE conventions differ, CI endpoints differ proportionally;
    # inherit the SE tolerance unless the golden explicitly pins CI.
    ci_tol = tol.get("conf_int", se_tol)

    q = golden["quantities"]
    if estimate is not None:
        np.testing.assert_allclose(
            np.asarray(estimate),
            np.asarray(q["estimate"]),
            rtol=est_tol,
            atol=0.0,
        )
    if std_error is not None:
        np.testing.assert_allclose(
            np.asarray(std_error),
            np.asarray(q["std_error"]),
            rtol=se_tol,
            atol=0.0,
        )
    if conf_low is not None:
        np.testing.assert_allclose(
            np.asarray(conf_low),
            np.asarray(q["conf_low"]),
            rtol=ci_tol,
            atol=0.0,
        )
    if conf_high is not None:
        np.testing.assert_allclose(
            np.asarray(conf_high),
            np.asarray(q["conf_high"]),
            rtol=ci_tol,
            atol=0.0,
        )
