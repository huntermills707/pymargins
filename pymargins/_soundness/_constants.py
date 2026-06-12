"""Calibration constants for the soundness layer.

Every constant in this module carries a docstring containing the citation
verbatim from the design note (§6.7).  Overrides are recorded in the plan hash.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# κ curvature (tier-1 driver)
# ---------------------------------------------------------------------------

KAPPA_RELIABLE = 0.1
"""κ ≤ 0.1 is considered reliable.

Basis: Bates–Watts / Skovgaard curvature calibration (shipped).
"""

KAPPA_BORDERLINE = 0.3
"""κ > 0.3 is considered borderline / unreliable.

Basis: Bates–Watts / Skovgaard curvature calibration (shipped).
"""

# ---------------------------------------------------------------------------
# Delta–simulation disagreement (tier-2 driver)
# ---------------------------------------------------------------------------

DISAGREEMENT_WARN = 0.05
"""Relative CI-endpoint disagreement > 5 % between delta and simulation triggers
a warning.

Basis: shipped convention (``_kappa.py`` docstring).
"""

# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------

ESS_NOTE_FRACTION = 0.5
"""ESS / n < 0.5 triggers a note.

Basis: Kish (1965) effective sample size; "half the information lost"
reading — convention.
"""

# ---------------------------------------------------------------------------
# Stabilized weight diagnostics
# ---------------------------------------------------------------------------

STABILIZED_WEIGHT_MAX_WARN = 20.0
"""Maximum normalized stabilized weight > 20 triggers a warn (positivity
violation).

Basis: Cole–Hernán (2008).
"""

STABILIZED_WEIGHT_MEAN_BAND = (0.9, 1.1)
"""Mean normalized stabilized weight outside [0.9, 1.1] triggers a note
(misspecification).

Basis: Cole–Hernán (2008).
"""

# ---------------------------------------------------------------------------
# Propensity-score trimming
# ---------------------------------------------------------------------------

PS_TRIM_DEFAULT = (0.1, 0.9)
"""Default PS trimming bounds when opted in.

Basis: Crump–Hotz–Imbens–Mitnik (2009) rule-of-thumb.
"""

# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------

FEW_CLUSTERS_WARN = 30
"""G < 30 triggers a warning for cluster bootstrap.

Basis: Cameron–Miller (2015): "maybe less than 30, maybe less than 50" —
take 30.
"""

# ---------------------------------------------------------------------------
# Bootstrap tail counts
# ---------------------------------------------------------------------------

TAIL_COUNT_NOTE = 25
"""B·(1−level)/2 < 25 triggers a note.

Basis: Efron–Tibshirani (B ≥ 1000 for 95 % CIs ⇒ ~25/tail);
Davison–Hinkley (B+1)α-integer convention (999, 1999, …).
"""

TAIL_COUNT_WARN = 10
"""B·(1−level)/2 < 10 triggers a warning.

Basis: Efron–Tibshirani (B ≥ 1000 for 95 % CIs ⇒ ~25/tail);
Davison–Hinkley (B+1)α-integer convention (999, 1999, …).
"""

SE_ONLY_MIN_B = 200
"""B ≥ 200 is sufficient for bootstrap SEs only (ci="se").

Basis: Efron–Tibshirani (SEs converge far faster than tails).
"""

BCA_MIN_B = 1999
"""B < 1999 triggers a note for BCa intervals.

Basis: Davison–Hinkley; Carpenter–Bithell (2000).
"""

# ---------------------------------------------------------------------------
# MI mixing gate
# ---------------------------------------------------------------------------

MIXING_MIN_M = 50
"""M ≥ 50 required for ``pooling="mix"`` (Zhou–Reiter mixture).

Basis: mixture omits (1+1/M)B ⇒ ≤ 2 % variance error at M = 50, below
MC noise; Zhou–Reiter "large M".
"""

# ---------------------------------------------------------------------------
# m-out-of-n subsampling
# ---------------------------------------------------------------------------


def m_out_of_n(n: int) -> int:
    """Default m-out-of-n subsample size.

    Basis: Politis–Romano–Wolf subsampling convention; Bickel–Sakov adaptive
    *(future)*.
    """
    return math.ceil(n ** (2 / 3))


# ---------------------------------------------------------------------------
# Block bootstrap fallback
# ---------------------------------------------------------------------------


def block_length_fallback(n: int) -> int:
    """Default block length when auto-selection is unavailable.

    Basis: Hall–Horowitz–Jing (1995); Politis–White (2004) primary.
    """
    return math.ceil(n ** (1 / 3))


# ---------------------------------------------------------------------------
# Replicate failure rate
# ---------------------------------------------------------------------------

REPLICATE_FAILURE_NOTE = 0.01
"""> 1 % replicate failures triggers a note.

Basis: convention — least literature-anchored row, flagged as such.
"""

REPLICATE_FAILURE_WARN = 0.05
"""> 5 % replicate failures triggers a warning.

Basis: convention — least literature-anchored row, flagged as such.
"""
