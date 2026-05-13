"""
pymargins._result._arithmetic

Arithmetic operators and composability helpers for MarginsResult.
"""

from __future__ import annotations
from typing import Optional

import jax.numpy as jnp
import numpy as np
from scipy import stats

from ._margins import MarginsResult


# ---------------------------------------------------------------------------
# Same-session compatibility check
# ---------------------------------------------------------------------------

def _check_compatible(self: MarginsResult, other: MarginsResult) -> None:
    """Verify two results came from the same session and are composable."""
    self_sess = self._session_obj()
    other_sess = other._session_obj()
    if self_sess is None or other_sess is None:
        raise ValueError(
            "Composition requires both results to carry a session reference."
        )
    if self_sess is not other_sess:
        raise ValueError(
            "Cannot compose results from different Margins sessions. "
            "Different sessions may have different inference scales and "
            "covariances; composition is not well-defined."
        )


# ---------------------------------------------------------------------------
# Dunder methods
# ---------------------------------------------------------------------------

def __sub__(self: MarginsResult, other: MarginsResult) -> MarginsResult:
    """Difference of two estimands with proper joint inference.

    Computes the delta-method variance of the difference using the joint
    gradient and the shared Σ̂ from the session. Available only when
    both results carry gradients (delta-method results).
    """
    self._check_compatible(other)
    return _combine_results(
        self, other, lambda a, b: a - b,
        grad_combine=lambda g1, g2: g1 - g2,
        label_combine=lambda l1, l2: f"({l1}) - ({l2})",
    )


def __add__(self: MarginsResult, other: MarginsResult) -> MarginsResult:
    """Add two estimands with proper joint inference via the delta method."""
    self._check_compatible(other)
    return _combine_results(
        self, other, lambda a, b: a + b,
        grad_combine=lambda g1, g2: g1 + g2,
        label_combine=lambda l1, l2: f"({l1}) + ({l2})",
    )


def __mul__(self: MarginsResult, other) -> MarginsResult:
    """Scale the estimand by a scalar, with inference-aware transforms."""
    if isinstance(other, MarginsResult):
        raise ValueError(
            "Product of two MarginsResults is nonlinear; use evaluate() "
            "with a custom compose function instead."
        )
    try:
        scalar = float(other)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"MarginsResult multiplication requires a scalar, got {type(other).__name__}"
        ) from exc

    # Scale-aware: SE/gradient/draws are on the inference scale;
    # estimate and CI bounds are on the reporting scale.
    if self.phi is not None and self.phi_inv is not None:
        new_est = np.asarray(self.phi(scalar * self.phi_inv(self.estimate)))
        lo = np.asarray(self.phi(scalar * self.phi_inv(self.conf_int_lower)))
        hi = np.asarray(self.phi(scalar * self.phi_inv(self.conf_int_upper)))
        if scalar < 0:
            lo, hi = hi, lo
        new_lo, new_hi = lo, hi
        new_draws = (
            np.asarray(self.phi(scalar * self.phi_inv(self.draws)))
            if self.draws is not None else None
        )
    else:
        new_est = self.estimate * scalar
        new_lo = (self.conf_int_lower * scalar
                  if scalar > 0 else self.conf_int_upper * scalar)
        new_hi = (self.conf_int_upper * scalar
                  if scalar > 0 else self.conf_int_lower * scalar)
        new_draws = self.draws * scalar if self.draws is not None else None

    return MarginsResult(
        estimate=new_est,
        std_error=self.std_error * abs(scalar),
        conf_int_lower=new_lo,
        conf_int_upper=new_hi,
        method=self.method,
        level=self.level,
        n_obs=self.n_obs,
        kappa=self.kappa,
        delta_sim_disagreement=self.delta_sim_disagreement,
        fallback_triggered=self.fallback_triggered,
        fallback_reason=self.fallback_reason,
        estimand_metadata={**self.estimand_metadata,
                           "labels": [f"({l})*{scalar}"
                                      for l in self.estimand_metadata.get("labels", [])]},
        gradient=(self.gradient * scalar
                  if self.gradient is not None else None),
        draws=new_draws,
        draws_inf=(self.draws_inf * scalar if self.draws_inf is not None else None),
        cov_params=self.cov_params,
        phi=self.phi,
        phi_inv=self.phi_inv,
        session=self.session,
        ci_method=self.ci_method,
        bootstrap_extras=self.bootstrap_extras,
    )


def __truediv__(self: MarginsResult, other) -> MarginsResult:
    """Scale the estimand by the reciprocal of a scalar."""
    if isinstance(other, MarginsResult):
        raise ValueError(
            "Ratio of two MarginsResults is nonlinear; use evaluate() "
            "with a custom compose function (e.g., compose=lambda p: p[0]/p[1]) "
            "for proper inference."
        )
    return self.__mul__(1.0 / float(other))


# ---------------------------------------------------------------------------
# Cosmetic transformations (don't affect inference)
# ---------------------------------------------------------------------------

def scaled(self: MarginsResult, by: float, units: Optional[str] = None) -> MarginsResult:
    """Cosmetic rescaling of the estimate and CI for reporting.

    Multiplies the estimate, SE, and CI bounds by `by`. Useful for unit
    conversion or percentage display. Does not affect any inference
    properties — the underlying gradients/draws are also rescaled so
    composability is preserved.

    Parameters
    ----------
    by : float
        Scaling factor. May be negative (flips direction).

    units : str, optional
        Label for the new scale; recorded in metadata.

    Returns
    -------
    scaled : MarginsResult
    """
    new = self * by
    if units:
        new.estimand_metadata["units"] = units
    return new


# ---------------------------------------------------------------------------
# Internal: result combination helper
# ---------------------------------------------------------------------------

def _join_fallback_reasons(a_reason, b_reason):
    """Combine fallback reasons from two results."""
    if not a_reason and not b_reason:
        return None
    if a_reason and not b_reason:
        return a_reason
    if b_reason and not a_reason:
        return b_reason
    return f"{a_reason}; {b_reason}"


def _combine_results(
    a: MarginsResult,
    b: MarginsResult,
    estimate_combine,
    grad_combine,
    label_combine,
) -> MarginsResult:
    """Combine two results from the same session via a linear operation."""
    # Inference-scale estimates and combined gradient
    a_inf = a.phi_inv(a.estimate) if a.phi_inv is not None else a.estimate
    b_inf = b.phi_inv(b.estimate) if b.phi_inv is not None else b.estimate
    combined_inf = estimate_combine(a_inf, b_inf)

    if a.gradient is None or b.gradient is None:
        raise ValueError(
            "Composition currently requires delta-method results (with "
            "gradients). Simulation/bootstrap composition would require "
            "matched draws; use the draws array manually or re-run with "
            "method='delta'."
        )

    new_grad = grad_combine(a.gradient, b.gradient)

    # Guard against vector-valued results — composition only defined for scalars
    if jnp.ndim(new_grad) != 1:
        raise NotImplementedError(
            "Composition is only supported for scalar estimands. "
            "For vector results, compose elementwise or use evaluate()."
        )

    # New SE and CI from delta on the combined gradient.
    # Both results are from the same session and were produced with the
    # same vcov_spec, so a.cov_params is the canonical Σ̂ for composition.
    if a.cov_params is None:
        raise ValueError(
            "Composition requires Σ̂ on the result (cov_params). The "
            "originating session should have populated it; if this result "
            "was constructed manually, supply cov_params."
        )
    cov = jnp.asarray(a.cov_params)
    var = jnp.dot(jnp.asarray(new_grad), cov @ jnp.asarray(new_grad))
    se = float(jnp.sqrt(var))

    z = stats.norm.ppf(0.5 + a.level / 2.0)
    lo_inf = combined_inf - z * se
    hi_inf = combined_inf + z * se

    if a.phi is not None:
        estimate_report = np.asarray(a.phi(combined_inf))
        lower_report = np.asarray(a.phi(lo_inf))
        upper_report = np.asarray(a.phi(hi_inf))
    else:
        estimate_report = combined_inf
        lower_report = lo_inf
        upper_report = hi_inf

    a_label = (a.estimand_metadata.get("labels", [""])[0]
               if a.estimand_metadata.get("labels") else "A")
    b_label = (b.estimand_metadata.get("labels", [""])[0]
               if b.estimand_metadata.get("labels") else "B")

    return MarginsResult(
        estimate=np.asarray(estimate_report),
        std_error=np.asarray(se),
        conf_int_lower=np.asarray(lower_report),
        conf_int_upper=np.asarray(upper_report),
        method=a.method,
        level=a.level,
        n_obs=max(a.n_obs, b.n_obs),
        kappa=None,  # not recomputed for combined results
        delta_sim_disagreement=None,
        fallback_triggered=a.fallback_triggered or b.fallback_triggered,
        fallback_reason=_join_fallback_reasons(a.fallback_reason, b.fallback_reason),
        estimand_metadata={"labels": [label_combine(a_label, b_label)]},
        gradient=new_grad,
        draws=None,
        cov_params=a.cov_params,
        phi=a.phi,
        phi_inv=a.phi_inv,
        session=a.session,
    )


# ---------------------------------------------------------------------------
# Patch methods onto MarginsResult
# ---------------------------------------------------------------------------

MarginsResult._check_compatible = _check_compatible
MarginsResult.__sub__ = __sub__
MarginsResult.__add__ = __add__
MarginsResult.__mul__ = __mul__
MarginsResult.__truediv__ = __truediv__
MarginsResult.scaled = scaled
