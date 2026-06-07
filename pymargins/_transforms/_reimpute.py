"""Reimpute stage for bootstrap-then-impute multiple imputation."""

from __future__ import annotations

import hashlib
import warnings

import pandas as pd


def reimpute(imputer, *, incomplete: pd.DataFrame, warn_on_deterministic: bool = True):
    """Create a ``reimpute`` pipeline stage.

    On every bootstrap replicate the *imputer* is called fresh on the
    resampled incomplete data.  This injects imputation-model parameter
    uncertainty into the bootstrap distribution, making even nominally
    "improper" imputers proper enough for valid inference.

    Parameters
    ----------
    imputer : callable
        ``imputer(frame) -> frame``.  Must accept a DataFrame and return
        a DataFrame of the same shape with missing values filled.  The
        callable is expected to **fit-and-impute** (re-derive), not
        apply a frozen fitted model.
    incomplete : pd.DataFrame
        The *incomplete* data (with missingness).  The bootstrap resamples
        this frame, not the adapter's training data, so that every
        replicate has missing cells to impute.
    warn_on_deterministic : bool, default True
        Whether to run the cheap determinism guard at construction.
        Disable if you know your imputer is deterministic and you want
        to avoid the double-run overhead.

    Returns
    -------
    Stage
        A stage with ``requires_resampling=True`` (bootstrap-only) and
        ``source_data=incomplete``.

    Warns
    -----
    UserWarning
        If calling *imputer* twice on the same frame yields byte-identical
        output, the imputer is deterministic-given-data (no residual draw).
        MI variance will be too narrow; consider a stochastic imputer such
        as ``IterativeImputer(sample_posterior=True)``.
    UserWarning
        If the imputer exposes a ``random_state`` attribute that is
        ``None``, reproducibility is not guaranteed even with a fixed
        session ``rng_seed``.  Set ``random_state`` to an integer on the
        imputer for deterministic draws.
    """
    if warn_on_deterministic:
        _warn_if_deterministic(imputer, incomplete)

    _warn_if_unseeded(imputer)

    stage = _ReimputeStage(imputer, incomplete)
    return stage


class _ReimputeStage:
    requires_resampling = True
    alters_rows = False
    emits_columns = ()

    def __init__(self, imputer, incomplete: pd.DataFrame):
        self._imputer = imputer
        self.source_data = incomplete
        # Stable hash key for bank discrimination (F10).
        # Uses type only (coarse but stable across sessions); if users need
        # finer discrimination they should pass distinct callables.
        self._pymargins_hash_key = hashlib.sha256(
            f"{type(imputer).__module__}.{type(imputer).__name__}".encode()
        ).hexdigest()[:16]

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def prepare_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        seed = getattr(self, "_pymargins_replicate_seed", None)
        if seed is not None and hasattr(self._imputer, "set_params"):
            try:
                self._imputer.set_params(random_state=seed)
            except Exception:
                pass  # set_params may reject random_state; ignore silently
        return self._imputer(data)


def _warn_if_deterministic(imputer, incomplete: pd.DataFrame, max_rows: int = 200):
    """Call imputer twice on a capped sample; identical output => warn."""
    sample = incomplete.head(max_rows)
    try:
        out1 = imputer(sample.copy())
        out2 = imputer(sample.copy())
    except Exception:
        return  # guard failed silently — not a hard error

    if not isinstance(out1, pd.DataFrame) or not isinstance(out2, pd.DataFrame):
        return

    try:
        identical = (
            out1.shape == out2.shape
            and (out1.fillna("__NA__") == out2.fillna("__NA__")).all().all()
        )
    except Exception:
        return

    if identical:
        warnings.warn(
            "The imputer appears to be deterministic (identical output on repeated "
            "calls to the same frame). Bootstrap-then-impute variance will be too "
            "narrow because no residual noise is drawn. Consider a stochastic imputer "
            "(e.g. sklearn.impute.IterativeImputer with sample_posterior=True).",
            UserWarning,
            stacklevel=3,
        )


def _warn_if_unseeded(imputer):
    """Warn if the imputer has random_state=None (non-reproducible draws)."""
    obj = imputer
    # If it's a bound method, inspect the underlying object
    if hasattr(imputer, "__self__"):
        obj = imputer.__self__
    if hasattr(obj, "random_state"):
        rs = getattr(obj, "random_state", None)
        if rs is None:
            warnings.warn(
                "The imputer has random_state=None. Bootstrap replicate draws will not "
                "be reproducible even with a fixed session rng_seed. Set random_state "
                "to an integer on the imputer for deterministic MI results.",
                UserWarning,
                stacklevel=3,
            )
