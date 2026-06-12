"""Bank caching tests.

Design §9.4, req §5. Added in 0.4.0 (R1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins._engine._banks import BankRetentionError, BankSet


@pytest.fixture
def fit_logit():
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, size=n).astype(float),
            "x": rng.normal(size=n),
        }
    )
    return smf.logit("y ~ x", data=df).fit(disp=0)


def test_indices_built_once():
    bank = BankSet(plan_hash="h", branch_id=0, seed=42)
    a = bank.resample_indices(n_obs=10, B=3)
    b = bank.resample_indices(n_obs=10, B=3)
    assert a is b
    assert len(a) == 3


def test_draws_built_once():
    bank = BankSet(plan_hash="h", branch_id=0, seed=42)
    beta = np.array([0.0, 1.0])
    cov = np.eye(2) * 0.01
    a = bank.sim_draws(beta=beta, cov=cov, n_sim=5)
    b = bank.sim_draws(beta=beta, cov=cov, n_sim=5)
    assert a is b


def test_states_replayed(fit_logit, monkeypatch):
    bank = BankSet(plan_hash="h", branch_id=0, seed=42)
    adapter = fit_logit.model
    data = fit_logit.model.data.frame
    indices = bank.resample_indices(n_obs=len(data), B=2)

    from pymargins._inference import _bootstrap

    calls = []
    original = _bootstrap._harvest_bootstrap_states

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(_bootstrap, "_harvest_bootstrap_states", spy)

    bank.bootstrap_states(adapter=adapter, data=data, indices=indices)
    bank.bootstrap_states(adapter=adapter, data=data, indices=indices)

    assert len(calls) == 1


def test_distinct_seeds_distinct_banks():
    bank_a = BankSet(plan_hash="h", branch_id=0, seed=1)
    bank_b = BankSet(plan_hash="h", branch_id=0, seed=2)
    a = bank_a.resample_indices(n_obs=10, B=3)
    b = bank_b.resample_indices(n_obs=10, B=3)
    assert not all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_bank_retention_error_message():
    err = BankRetentionError()
    assert "replicate products" in str(err)
