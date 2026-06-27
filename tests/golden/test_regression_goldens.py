"""Layer-4 regression goldens: reproduction of anchor cells within tolerance.

Design §7.4, R7.4. These arrays were recorded by tools/record_goldens.py
from the new engine; regeneration requires a ledger entry.

The recorded values are environment-specific in their low-order bits: the
GLM/logit ``expit`` path and the simulation MVN-sampling path drift by ~1e-13
across Python/numpy/scipy/BLAS builds (ledger D20/D22). We therefore compare
within the project's documented oracle tolerances instead of byte-for-byte, so
the gate holds across the 3.10/3.12/3.14 CI matrix while still catching any
genuine logic regression (which would exceed these tolerances by orders of
magnitude). The numbers themselves remain oracle-validated by the layer-1/2
analytic + R-golden suites.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

jax.config.update("jax_enable_x64", True)

from pymargins import steps
from pymargins.estimators import GComputation

GOLDEN_DIR = Path(__file__).parent
MANIFEST = GOLDEN_DIR / "manifest.json"

SEED = 12345

# Comparison tolerances — mirror tests/oracle/_tolerances.py (the layer-1/2
# correctness authority). Changing any value = ledger entry. A small atol
# guards entries near zero where rtol alone is meaningless.
TOL_EST = 1e-6
TOL_SE = 1e-5
TOL_CI = 1e-5
ATOL = 1e-9


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, size=n).astype(float),
            "y_cont": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treat": rng.binomial(1, 0.5, size=n).astype(float),
            "cluster": np.repeat(np.arange(40), 5),
        }
    )


@pytest.fixture(scope="session")
def df():
    return _make_df()


@pytest.fixture(scope="session")
def fit_ols(df):
    return smf.ols("y_cont ~ treat + x1 + x2", data=df).fit()


@pytest.fixture(scope="session")
def fit_glm(df):
    return smf.glm("y ~ treat + x1 + x2", data=df, family=sm.families.Binomial()).fit()


@pytest.fixture(scope="session")
def weights(df):
    rng = np.random.default_rng(7)
    return rng.uniform(0.5, 1.5, size=len(df))


def _load_manifest():
    return json.loads(MANIFEST.read_text())


def _load_cell(cell_id: str):
    path = GOLDEN_DIR / f"{cell_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _run_cell(cell_id: str, fit_ols, fit_glm, weights):
    """Replay the same estimator + query that the recorder used."""
    parts = cell_id.split("_")
    family = parts[0]
    method = parts[1]

    fit = fit_ols if family == "ols" else fit_glm

    if "cluster_bootstrap" in cell_id:
        df = _make_df()
        cluster_ids = df["cluster"].values
        est = GComputation(
            steps.input(df, cluster=cluster_ids),
            outcome=fit,
            at="overall",
            method="bootstrap",
            B=200,
            seed=SEED,
        )
    elif method == "bootstrap":
        est = GComputation(fit, at="overall", method="bootstrap", B=200, seed=SEED)
    else:
        est = GComputation(fit, at="overall", method=method, seed=SEED)

    if "weights" in cell_id:
        est = GComputation(
            fit, at="overall", method="delta", weights=weights, seed=SEED
        )

    if cell_id.endswith("_predict"):
        return est.predict()
    if cell_id.endswith("_dydx"):
        return est.dydx("x1")
    if cell_id.endswith("_contrasts"):
        return est.contrasts(
            scenarios=[{"atexog": {"treat": 1}}, {"atexog": {"treat": 0}}],
            contrasts=[1, -1],
        )
    if cell_id.endswith("_evaluate"):
        return est.evaluate(
            scenarios=[{"atexog": {"treat": 1}}, {"atexog": {"treat": 0}}],
            compose=lambda x: x[1] - x[0],
        )
    if "_elasticity_" in cell_id:
        query = parts[-1]
        return getattr(est, query)("x1")
    raise ValueError(f"Unknown cell kind: {cell_id}")


@pytest.mark.parametrize("cell", _load_manifest()["cells"], ids=lambda c: c["cell_id"])
def test_regression_golden(cell, fit_ols, fit_glm, weights):
    recorded = _load_cell(cell["cell_id"])

    assert "estimate" in recorded
    assert recorded["estimate"].dtype == np.float64

    result = _run_cell(cell["cell_id"], fit_ols, fit_glm, weights)

    np.testing.assert_allclose(
        np.asarray(result.estimate), recorded["estimate"], rtol=TOL_EST, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(result.std_error), recorded["std_error"], rtol=TOL_SE, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(result.conf_int_lower),
        recorded["conf_int_lower"],
        rtol=TOL_CI,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.conf_int_upper),
        recorded["conf_int_upper"],
        rtol=TOL_CI,
        atol=ATOL,
    )

    if "draws" in recorded:
        assert result.draws is not None
        np.testing.assert_allclose(
            np.asarray(result.draws), recorded["draws"], rtol=TOL_SE, atol=ATOL
        )


def test_manifest_matches_files_on_disk():
    """Every manifest cell has a matching npz; no orphan npzs exist."""
    manifest = _load_manifest()
    cell_ids = {c["cell_id"] for c in manifest["cells"]}
    on_disk = {p.stem for p in GOLDEN_DIR.glob("*.npz")}
    assert cell_ids == on_disk, f"manifest≠disk: {cell_ids ^ on_disk}"


def test_record_script_is_available():
    """The recorder must be present for regeneration."""
    script = GOLDEN_DIR.parent.parent / "tools" / "record_goldens.py"
    assert script.exists()
