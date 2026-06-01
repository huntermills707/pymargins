"""Tests for _gradients.py coverage gaps."""

import numpy as np
import pytest

from pymargins._gradients import _jax_link_inverse, _jax_link_inverse_deriv


def _make_link(name, **kwargs):
    class FakeLink:
        pass

    FakeLink.__name__ = name
    link = FakeLink()
    for k, v in kwargs.items():
        setattr(link, k, v)
    return link


# ---------------------------------------------------------------------------
# _jax_link_inverse for less-common links
# ---------------------------------------------------------------------------


def test_link_inverse_cloglog():
    f = _jax_link_inverse(_make_link("CLogLog"))
    assert np.isfinite(float(f(0.0)))


def test_link_inverse_loglog():
    f = _jax_link_inverse(_make_link("LogLog"))
    assert np.isfinite(float(f(0.0)))


def test_link_inverse_logc():
    f = _jax_link_inverse(_make_link("LogC"))
    assert np.isfinite(float(f(-1.0)))


def test_link_inverse_power_p0():
    f = _jax_link_inverse(_make_link("Power", power=0.0))
    assert np.isfinite(float(f(1.0)))


def test_link_inverse_power_p2():
    f = _jax_link_inverse(_make_link("Power", power=2.0))
    assert np.isfinite(float(f(1.0)))


def test_link_inverse_inverse_power():
    f = _jax_link_inverse(_make_link("InversePower"))
    assert np.isfinite(float(f(1.0)))


def test_link_inverse_inverse_squared():
    f = _jax_link_inverse(_make_link("InverseSquared"))
    assert np.isfinite(float(f(1.0)))


def test_link_inverse_sqrt():
    f = _jax_link_inverse(_make_link("Sqrt"))
    assert np.isfinite(float(f(4.0)))


def test_link_inverse_cauchy():
    f = _jax_link_inverse(_make_link("Cauchy"))
    assert np.isfinite(float(f(0.0)))


def test_link_inverse_nb():
    f = _jax_link_inverse(_make_link("NegativeBinomial", alpha=1.0))
    assert np.isfinite(float(f(1.0)))


def test_link_inverse_unknown_raises():
    with pytest.raises(NotImplementedError, match="No JAX mapping for link"):
        _jax_link_inverse(_make_link("UnknownLink"))


# ---------------------------------------------------------------------------
# _jax_link_inverse_deriv for less-common links
# ---------------------------------------------------------------------------


def test_link_inverse_deriv_cloglog():
    d = _jax_link_inverse_deriv(_make_link("CLogLog"))
    assert np.isfinite(float(d(0.0)))


def test_link_inverse_deriv_loglog():
    d = _jax_link_inverse_deriv(_make_link("LogLog"))
    assert np.isfinite(float(d(0.0)))


def test_link_inverse_deriv_logc():
    d = _jax_link_inverse_deriv(_make_link("LogC"))
    assert np.isfinite(float(d(-1.0)))


def test_link_inverse_deriv_power_p0():
    d = _jax_link_inverse_deriv(_make_link("Power", power=0.0))
    assert np.isfinite(float(d(1.0)))


def test_link_inverse_deriv_power_p2():
    d = _jax_link_inverse_deriv(_make_link("Power", power=2.0))
    assert np.isfinite(float(d(1.0)))


def test_link_inverse_deriv_inverse_power():
    d = _jax_link_inverse_deriv(_make_link("InversePower"))
    assert np.isfinite(float(d(1.0)))


def test_link_inverse_deriv_inverse_squared():
    d = _jax_link_inverse_deriv(_make_link("InverseSquared"))
    assert np.isfinite(float(d(1.0)))


def test_link_inverse_deriv_sqrt():
    d = _jax_link_inverse_deriv(_make_link("Sqrt"))
    assert np.isfinite(float(d(4.0)))


def test_link_inverse_deriv_cauchy():
    d = _jax_link_inverse_deriv(_make_link("Cauchy"))
    assert np.isfinite(float(d(0.0)))


def test_link_inverse_deriv_nb():
    d = _jax_link_inverse_deriv(_make_link("NegativeBinomial", alpha=1.0))
    assert np.isfinite(float(d(1.0)))


def test_link_inverse_deriv_unknown_raises():
    with pytest.raises(NotImplementedError, match="No JAX mapping for link derivative"):
        _jax_link_inverse_deriv(_make_link("UnknownLink"))
