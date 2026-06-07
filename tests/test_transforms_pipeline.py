"""Tests for the transform pipeline plumbing (Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import Margins
from pymargins._transforms import IdentityStage

# ---------------------------------------------------------------------------
# Regression guard: IdentityStage yields identical results to no pipeline
# ---------------------------------------------------------------------------


def test_identity_pipeline_yields_identical_results():
    """An IdentityStage pipeline through bootstrap must be inert."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(scale=0.5, size=n)
    fit = smf.ols("y ~ x1 + x2", data=df).fit()

    m_no_pipe = Margins(fit, method="bootstrap", n_boot=50, n_jobs=1, rng_seed=7)
    r_no_pipe = m_no_pipe.predict(atexog={"x1": 0})

    m_pipe = Margins(
        fit,
        transforms=[IdentityStage()],
        method="bootstrap",
        n_boot=50,
        n_jobs=1,
        rng_seed=7,
    )
    r_pipe = m_pipe.predict(atexog={"x1": 0})

    assert np.isclose(r_pipe.estimate, r_no_pipe.estimate)
    assert np.isclose(r_pipe.std_error, r_no_pipe.std_error)
    assert np.isclose(r_pipe.conf_int_lower, r_no_pipe.conf_int_lower)
    assert np.isclose(r_pipe.conf_int_upper, r_no_pipe.conf_int_upper)


# ---------------------------------------------------------------------------
# Recording stage: prepare_resample called once per replicate
# ---------------------------------------------------------------------------


class _RecordingStage:
    requires_resampling = False
    alters_rows = False
    emits_columns = ()
    source_data = None

    def __init__(self):
        self.calls = 0

    def prepare(self, data):
        return data

    def prepare_resample(self, data):
        self.calls += 1
        return data


def test_recording_stage_called_once_per_replicate():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    stage = _RecordingStage()
    m = Margins(
        fit,
        transforms=[stage],
        method="bootstrap",
        n_boot=30,
        n_jobs=1,
        rng_seed=1,
    )
    _ = m.predict()
    assert stage.calls == 30


# ---------------------------------------------------------------------------
# Index handling: None when alters_rows, else preserved
# ---------------------------------------------------------------------------


class _RowAlteringStage:
    requires_resampling = False
    alters_rows = True
    emits_columns = ()
    source_data = None

    def prepare(self, data):
        return data.iloc[::2].reset_index(drop=True)

    def prepare_resample(self, data):
        return data.iloc[::2].reset_index(drop=True)


class _NonRowAlteringStage:
    requires_resampling = False
    alters_rows = False
    emits_columns = ()
    source_data = None

    def prepare(self, data):
        return data

    def prepare_resample(self, data):
        return data


def test_alters_rows_forces_index_none():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(
        fit,
        transforms=[_RowAlteringStage()],
        method="bootstrap",
        n_boot=10,
        n_jobs=1,
        rng_seed=1,
    )
    _ = m.predict()
    # If any bootstrap refit succeeded, its adapter should have _pymargins_bootstrap_idx = None
    # We inspect the cache directly.
    from pymargins.margins._inference_glue import _bootstrap_states_bank

    states, _, _ = _bootstrap_states_bank(m)
    assert len(states) > 0
    for _b, adapter in states:
        assert adapter._pymargins_bootstrap_idx is None


def test_non_alters_rows_preserves_index():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(
        fit,
        transforms=[_NonRowAlteringStage()],
        method="bootstrap",
        n_boot=10,
        n_jobs=1,
        rng_seed=1,
    )
    _ = m.predict()
    from pymargins.margins._inference_glue import _bootstrap_states_bank

    states, _, _ = _bootstrap_states_bank(m)
    assert len(states) > 0
    for _b, adapter in states:
        assert adapter._pymargins_bootstrap_idx is not None


# ---------------------------------------------------------------------------
# Bank key: differs with transforms, stable for transforms=None
# ---------------------------------------------------------------------------


def test_bank_key_differs_when_transforms_present():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m1 = Margins(fit, method="bootstrap", n_boot=50, rng_seed=1)
    m2 = Margins(
        fit, transforms=[IdentityStage()], method="bootstrap", n_boot=50, rng_seed=1
    )

    from pymargins.margins._inference_glue import _bootstrap_bank_key

    k1 = _bootstrap_bank_key(m1)
    k2 = _bootstrap_bank_key(m2)
    assert k1 != k2


def test_bank_key_identical_for_two_none_sessions():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m1 = Margins(fit, method="bootstrap", n_boot=50, rng_seed=1)
    m2 = Margins(fit, method="bootstrap", n_boot=50, rng_seed=1)

    from pymargins.margins._inference_glue import _bootstrap_bank_key

    k1 = _bootstrap_bank_key(m1)
    k2 = _bootstrap_bank_key(m2)
    assert k1 == k2


# ---------------------------------------------------------------------------
# n_jobs>1: picklable stage uses processes, unpicklable falls back
# ---------------------------------------------------------------------------


class _UnpicklableStage:
    """Holds a local lambda → cannot be pickled."""

    requires_resampling = False
    alters_rows = False
    emits_columns = ()
    source_data = None

    def __init__(self):
        self._fn = lambda x: x

    def prepare(self, data):
        return data

    def prepare_resample(self, data):
        return self._fn(data)


def test_picklable_stage_process_pool():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(
        fit,
        transforms=[IdentityStage()],
        method="bootstrap",
        n_boot=20,
        n_jobs=2,
        rng_seed=1,
    )
    r = m.predict()
    assert r.n_boot_effective == 20


def test_unpicklable_stage_falls_back_to_threads():
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": 1.0 + 0.5 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ x", data=df).fit()

    m = Margins(
        fit,
        transforms=[_UnpicklableStage()],
        method="bootstrap",
        n_boot=20,
        n_jobs=2,
        rng_seed=1,
    )
    with pytest.warns(RuntimeWarning, match="cannot be pickled"):
        r = m.predict()
    assert r.n_boot_effective == 20
