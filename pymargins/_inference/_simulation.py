from __future__ import annotations
from typing import Optional
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from .._kappa import kappa, kappa_vector
from .._estimands import is_jax_differentiable
from ._config import InferenceConfig


def _run_simulation(h, adapter, config, estimand_metadata, *, fallback_reason=None, skip_kappa=False):
    """Krinsky–Robb simulation: sample β̃ ~ N(β̂, Σ̂), evaluate h, take
    quantiles for CIs."""
    beta = adapter.coefficients()
    Sigma = config.cov_params if config.cov_params is not None else adapter.covariance()

    rng = np.random.default_rng(
        [config.rng_seed, 0] if config.rng_seed is not None else None
    )
    Sigma_np = np.asarray(Sigma)
    beta_np = np.asarray(beta)
    eigvals = np.linalg.eigvalsh(Sigma_np)
    if np.any(eigvals < -1e-8):
        raise ValueError(
            "Covariance matrix (vcov) is not positive semi-definite. "
            "Simulation draws would produce NaN. "
            f"Minimum eigenvalue: {np.min(eigvals):.3e}. "
            "Check your vcov specification or use a different inference method."
        )
    draws_beta = rng.multivariate_normal(beta_np, Sigma_np, size=config.n_sim)
    if not np.all(np.isfinite(draws_beta)):
        raise ValueError(
            "Simulation draws contain NaN or Inf. The covariance matrix may be "
            "singular or numerically unstable."
        )

    estimate = h(beta)
    try:
        h_draws_inf = np.asarray(jax.vmap(h)(jnp.asarray(draws_beta)))
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerBoolConversionError,
        jax.errors.TracerIntegerConversionError,
        jax.errors.UnexpectedTracerError,
    ):
        # vmap couldn't trace h; fall back to a Python loop. Genuine
        # shape/type bugs in h (TypeError/ValueError) are intentionally not
        # caught — they surface immediately with their original traceback.
        h_draws_inf = np.array([np.asarray(h(np.asarray(b))) for b in draws_beta])

    se = np.std(h_draws_inf, axis=0, ddof=1)

    # Curvature diagnostic. Gated on JAX-differentiability because explicit
    # method="simulation" is how non-differentiable estimands reach the engine.
    # skip_kappa=True avoids redundant recomputation when the delta path already
    # computed κ and is falling back to simulation.
    k = None
    if not skip_kappa and config.diagnostics and is_jax_differentiable(h, beta):
        try:
            if jnp.ndim(estimate) == 0:
                k = kappa(h, beta, Sigma,
                          backend=config.gradient_backend, fd_step=config.fd_step)
            else:
                k = kappa_vector(h, beta, Sigma,
                                 backend=config.gradient_backend, fd_step=config.fd_step)
        except (ValueError, TypeError, jax.errors.JAXTypeError) as exc:
            warnings.warn(f"kappa diagnostic failed: {exc}", RuntimeWarning)

    # Apply phi to draws and estimate for reporting
    if config.phi is not None:
        try:
            h_draws = np.asarray(config.phi(jnp.asarray(h_draws_inf)))
        except (TypeError, ValueError):
            h_draws = np.asarray(config.phi(np.asarray(h_draws_inf)))
        try:
            estimate_report = np.asarray(config.phi(estimate))
        except (TypeError, ValueError):
            estimate_report = np.asarray(config.phi(np.asarray(estimate)))
    else:
        h_draws = h_draws_inf
        estimate_report = np.asarray(estimate)

    alpha = (1.0 - config.level) / 2.0
    lower = np.quantile(h_draws, alpha, axis=0)
    upper = np.quantile(h_draws, 1.0 - alpha, axis=0)

    return {
        "estimate": estimate_report,
        "std_error": se,
        "conf_int_lower": lower,
        "conf_int_upper": upper,
        "method": "simulation",
        "level": config.level,
        "kappa": np.asarray(k) if k is not None else None,
        "delta_sim_disagreement": None,
        "fallback_triggered": fallback_reason is not None,
        "fallback_reason": fallback_reason,
        "gradient": None,
        "draws": h_draws,
        "draws_inf": h_draws_inf,
        "estimand_metadata": estimand_metadata or {},
        "ci_method": None,
        "bootstrap_extras": None,
    }
