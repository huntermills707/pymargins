"""Scale serialization helpers for result disk persistence.

Moved from ``pymargins._result._margins`` in the R6 audit so that
``GraphResult`` can avoid importing the legacy ``MarginsResult`` module.
"""

from __future__ import annotations

import warnings

_KNOWN_PHI_MAP = {
    "jax.numpy.exp": ("jax.numpy", "exp"),
    "jax.numpy.log": ("jax.numpy", "log"),
    "jax.numpy.expm1": ("jax.numpy", "expm1"),
    "jax.numpy.tanh": ("jax.numpy", "tanh"),
    "jax.scipy.special.expit": ("jax.scipy.special", "expit"),
}


def _phi_to_name(phi):
    """Map a known phi/phi_inv function to a serializable name."""
    if phi is None:
        return None
    # Try to match by identity against known JAX functions
    try:
        import jax.numpy as jnp

        if phi is jnp.exp:
            return "jax.numpy.exp"
        if phi is jnp.log:
            return "jax.numpy.log"
        if phi is jnp.expm1:
            return "jax.numpy.expm1"
        if phi is jnp.tanh:
            return "jax.numpy.tanh"
    except Exception:
        pass
    try:
        from jax.scipy.special import expit

        if phi is expit:
            return "jax.scipy.special.expit"
    except Exception:
        pass
    # Not a known function — caller will warn
    return None


def _name_to_phi(name):
    """Reconstruct a phi/phi_inv function from its serialized name."""
    if name is None:
        return None
    module, attr = _KNOWN_PHI_MAP.get(name, (None, None))
    if module is None:
        warnings.warn(
            f"Unknown phi name {name!r}; returning None.", UserWarning, stacklevel=2
        )
        return None
    try:
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)
    except Exception as exc:
        warnings.warn(
            f"Could not reconstruct phi {name!r}: {exc}. Returning None.",
            UserWarning,
            stacklevel=2,
        )
        return None
