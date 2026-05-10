"""Tests for strict mode (IMPLEMENTATION_GUIDE.md §2.1)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymargins import Margins


@pytest.fixture
def fit_ols():
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "y": rng.standard_normal(n),
    })
    return smf.ols("y ~ x1 + x2", data=df).fit()


def test_strict_mode_rejects_defaults(fit_ols):
    """strict=True without explicit config args should raise ValueError."""
    with pytest.raises(ValueError, match="strict=True: argument 'phi' must be explicitly given"):
        Margins(fit_ols, strict=True)


def test_strict_mode_accepts_explicit_args(fit_ols):
    """strict=True with all config args explicitly given should succeed."""
    m = Margins(
        fit_ols,
        strict=True,
        phi=None,
        phi_inv=None,
        vcov=None,
        weights=None,
        at="overall",
        level=0.95,
        method="delta",
        kappa_threshold=0.3,
        rng_seed=None,
        n_sim=4000,
        n_boot=1000,
        n_jobs=1,
        gradient_backend="autodiff",
        fd_step=1e-6,
        diagnostics=True,
        cluster=None,
        block_size=None,
        bootstrap_config=None,
    )
    assert m.strict is True
    assert m.at == "overall"


def test_strict_mode_rejects_auto_gradient_backend(fit_ols):
    """strict=True with gradient_backend='auto' should raise ValueError."""
    with pytest.raises(ValueError, match="strict=True: gradient_backend='auto' is not allowed"):
        Margins(
            fit_ols,
            strict=True,
            phi=None,
            phi_inv=None,
            vcov=None,
            weights=None,
            at="overall",
            level=0.95,
            method="delta",
            kappa_threshold=0.3,
            rng_seed=None,
            n_sim=4000,
            n_boot=1000,
            n_jobs=1,
            gradient_backend="auto",
            fd_step=1e-6,
            diagnostics=True,
            cluster=None,
            block_size=None,
            bootstrap_config=None,
        )


def test_strict_mode_rejects_next_missing_arg(fit_ols):
    """strict=True should raise on the first missing arg after phi is provided."""
    with pytest.raises(ValueError, match="strict=True: argument 'phi_inv' must be explicitly given"):
        Margins(fit_ols, strict=True, phi=None)


def test_strict_mode_false_uses_defaults(fit_ols):
    """strict=False (default) should silently apply defaults."""
    m = Margins(fit_ols)
    assert m.strict is False
    assert m.at == "overall"
    assert m.level == 0.95
    assert m.method == "delta"


def test_strict_mode_with_explicit_nondefault_vcov(fit_ols):
    """strict=True should accept an explicit non-default vcov string."""
    m = Margins(
        fit_ols,
        strict=True,
        phi=None,
        phi_inv=None,
        vcov="HC0",
        weights=None,
        at="overall",
        level=0.95,
        method="delta",
        kappa_threshold=0.3,
        rng_seed=None,
        n_sim=4000,
        n_boot=1000,
        n_jobs=1,
        gradient_backend="autodiff",
        fd_step=1e-6,
        diagnostics=True,
        cluster=None,
        block_size=None,
        bootstrap_config=None,
    )
    assert m.vcov_spec == "HC0"


def test_strict_mode_convenience_constructor_fails_cleanly(fit_ols):
    """Convenience constructors (e.g. log_scale) do not set every strict arg;
    they should fail with a clear message so users know to use the main ctor."""
    with pytest.raises(ValueError, match="strict=True: argument 'vcov' must be explicitly given"):
        Margins.log_scale(fit_ols, strict=True)
