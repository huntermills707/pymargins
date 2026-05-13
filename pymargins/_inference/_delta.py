from __future__ import annotations
from typing import Optional
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from .._gradients import gradient
from .._delta import delta_se, delta_confint
from .._kappa import kappa, kappa_vector, delta_simulation_disagreement
from .._estimands import is_jax_differentiable
from ._config import InferenceConfig
from ._simulation import _run_simulation


def _run_delta(h, adapter, config, estimand_metadata):
    """Delta-method inference with optional κ-based fallback to simulation."""
    beta = adapter.coefficients()
    Sigma = config.cov_params if config.cov_params is not None else adapter.covariance()

    estimate = h(beta)
    grad = gradient(h, beta, backend=config.gradient_backend, fd_step=config.fd_step)

    # Curvature diagnostic
    k = None
    fallback_triggered = False
    fallback_reason = None
    if config.diagnostics:
        if jnp.ndim(estimate) == 0:
            k = kappa(h, beta, Sigma,
                      backend=config.gradient_backend, fd_step=config.fd_step)
        else:
            k = kappa_vector(h, beta, Sigma,
                             backend=config.gradient_backend, fd_step=config.fd_step)

        max_k = float(k) if jnp.ndim(k) == 0 else float(jnp.nanmax(jnp.asarray(k)))
        if max_k > config.kappa_threshold:
            # Auto-fallback to simulation
            warnings.warn(
                f"Delta-method curvature κ={max_k:.3f} exceeds threshold "
                f"({config.kappa_threshold}); falling back to simulation.",
                UserWarning, stacklevel=3,
            )
            sim_result = _run_simulation(
                h, adapter, config, estimand_metadata,
                fallback_reason=f"kappa={max_k:.3f}>threshold={config.kappa_threshold}",
                skip_kappa=True,
            )
            sim_result["kappa"] = np.asarray(k) if k is not None else None
            return sim_result

    # Construct CI on inference scale, then back-transform via phi
    lower, upper = delta_confint(
        estimate, grad, Sigma,
        level=config.level, phi=config.phi,
    )
    se = delta_se(grad, Sigma)

    # Optional comparison against simulation. Works for scalar and vector
    # estimands; for vectors it returns the max per-component disagreement.
    delta_sim_disagreement = None
    if config.diagnostics:
        try:
            delta_sim_disagreement = delta_simulation_disagreement(
                estimate, grad, Sigma, h, beta,
                level=config.level,
                n_sim=min(config.n_sim, 1000),  # Smaller sample for diagnostic
                rng_seed=config.rng_seed,
                phi=config.phi,
            )
        except (ValueError, TypeError, jax.errors.JAXTypeError) as exc:
            warnings.warn(f"delta-simulation diagnostic failed: {exc}", RuntimeWarning)

    estimate_report = config.phi(estimate) if config.phi is not None else estimate

    return {
        "estimate": np.asarray(estimate_report),
        "std_error": np.asarray(se),
        "conf_int_lower": np.asarray(lower),
        "conf_int_upper": np.asarray(upper),
        "method": "delta",
        "level": config.level,
        "kappa": np.asarray(k) if k is not None else None,
        "delta_sim_disagreement": delta_sim_disagreement,
        "fallback_triggered": False,
        "fallback_reason": None,
        "gradient": np.asarray(grad),
        "draws": None,
        "draws_inf": None,
        "estimand_metadata": estimand_metadata or {},
        "ci_method": None,
        "bootstrap_extras": None,
    }
