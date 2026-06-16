from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurveyDesign:
    """Complex-survey design specification.

    Declares the sampling design so pymargins can compute design-based
    standard errors (Taylor linearization) and design-correct bootstrap
    resampling. Attach to a session via ``GComputation(..., survey_design=...)``.

    Parameters
    ----------
    weights : array-like of shape (n_obs,)
        Sampling weights w_i (inverse inclusion probability). Required.
    psu : array-like of shape (n_obs,), optional
        Primary sampling unit (cluster) identifiers. If None, each row is its
        own PSU (no clustering).
    strata : array-like of shape (n_obs,), optional
        Stratum identifiers. If None, a single stratum is assumed.
    fpc : array-like of shape (n_obs,), optional
        Finite-population correction. Either the stratum population size N_h
        (per row, constant within stratum) or the sampling fraction n_h/N_h.
        See ``fpc_is_fraction``. If None, no FPC (with-replacement variance).
    fpc_is_fraction : bool, default False
        If True, ``fpc`` holds sampling fractions f_h = n_h/N_h directly.
        If False, ``fpc`` holds population counts N_h and fractions are
        derived as n_h / N_h where n_h is the number of PSUs sampled in
        stratum h.
    nest : bool, default True
        If True, PSU ids are treated as nested within strata (a PSU id is only
        compared within its stratum). This matches R survey's ``nest=TRUE``
        and is the safe default; set False only if PSU ids are unique
        across strata.

    Notes
    -----
    The three weight concepts in pymargins are distinct and must not be
    confused:
      * ``SurveyDesign.weights`` — design weights; drive the linearization
        variance and may be passed explicitly to ``GComputation(weights=...)``
        to obtain a population-weighted point estimate.
      * ``GComputation(weights=...)`` — aggregation weights for AME averaging.
        When ``survey_design`` is given, the user must explicitly set this
        to the design weights if a weighted point estimate is desired.
      * model-fit ``freq_weights``/``var_weights`` — passed to the estimator
        at fit time; already handled by the GLM adapter and out of scope here.
    """

    weights: np.ndarray
    psu: np.ndarray | None = None
    strata: np.ndarray | None = None
    fpc: np.ndarray | None = None
    fpc_is_fraction: bool = False
    nest: bool = True

    def __post_init__(self):
        # dataclass is frozen → use object.__setattr__ to normalize to arrays
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("survey weights must be 1-D")
        if not np.all(np.isfinite(w)):
            raise ValueError("survey weights must be finite")
        if np.any(w < 0):
            raise ValueError("survey weights must be non-negative")
        object.__setattr__(self, "weights", w)
        n = w.shape[0]
        for name in ("psu", "strata", "fpc"):
            val = getattr(self, name)
            if val is not None:
                arr = np.asarray(val)
                if arr.shape[0] != n:
                    raise ValueError(
                        f"survey_design.{name} length {arr.shape[0]} "
                        f"!= weights length {n}"
                    )
                object.__setattr__(self, name, arr)

    def hash_key(self) -> str:
        """Stable hash of the design, for bootstrap-bank cache keys."""
        import hashlib

        h = hashlib.sha256()
        for arr in (self.weights, self.psu, self.strata, self.fpc):
            h.update(b"none" if arr is None else np.asarray(arr).tobytes())
        h.update(str((self.fpc_is_fraction, self.nest)).encode())
        return h.hexdigest()[:16]
