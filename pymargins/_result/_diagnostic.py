"""pymargins._result._diagnostic

Diagnostic result type.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ._text import SummaryString


# ---------------------------------------------------------------------------
# Diagnostic result (from session_kappa)
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    """Output of session-level kappa diagnostic.

    Returned by Margins.diagnose() to summarize delta-method validity
    across the design space before any specific estimand is computed.

    Attributes
    ----------
    kappa_min, kappa_median, kappa_max : float
        Summary stats of κ across sampled covariate vectors.

    kappa_distribution : array
        All sampled κ values, for inspection.

    verdict : str
        Classification: "delta_reliable", "delta_borderline", or
        "delta_unreliable", driven by max κ vs configured thresholds.

    n_samples : int
        How many design points were sampled.

    recommendation : str
        Human-readable advice based on verdict.

    session_summary : str
        One-line summary of the session's analytical posture (scale, vcov,
        method) for context in audit logs.
    """
    kappa_min: float
    kappa_median: float
    kappa_max: float
    kappa_distribution: np.ndarray
    verdict: str
    n_samples: int
    recommendation: str
    session_summary: str = ""

    def summary(self) -> str:
        return SummaryString(
            f"Session diagnostic ({self.n_samples} design points)\n"
            f"  Session: {self.session_summary}\n"
            f"  κ min:    {self.kappa_min:.3f}\n"
            f"  κ median: {self.kappa_median:.3f}\n"
            f"  κ max:    {self.kappa_max:.3f}\n"
            f"  Verdict:  {self.verdict}\n"
            f"  {self.recommendation}"
        )
