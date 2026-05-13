"""pymargins._result._test

Hypothesis test result type.
"""

from __future__ import annotations
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hypothesis test result
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Output of a hypothesis test on a MarginsResult.

    Attributes
    ----------
    statistic : array
        Test statistic (z for Wald, χ² for joint Wald). Per-component for
        vector estimands.

    pvalue : array
        P-value(s).

    df : int, optional
        Degrees of freedom (for χ² tests).

    null_value : array
        The hypothesized value(s) being tested against.

    alternative : str
        "two-sided", "greater", or "less".

    method : str
        Test type ("wald", "joint_wald", "empirical").

    estimand_metadata : dict
        Carried over from the source MarginsResult for output formatting.
    """
    statistic: np.ndarray
    pvalue: np.ndarray
    df: Optional[int] = None
    null_value: Union[np.ndarray, float] = 0.0
    alternative: str = "two-sided"
    method: str = "wald"
    estimand_metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the test."""
        lines = [
            f"Hypothesis test ({self.method})",
            f"  H0: estimand = {self.null_value}",
            f"  Alternative: {self.alternative}",
        ]
        stat = np.atleast_1d(self.statistic)
        p = np.atleast_1d(self.pvalue)
        if stat.size == 1:
            lines.append(f"  Statistic: {float(stat[0]):.4f}")
            lines.append(f"  P-value:   {float(p[0]):.4g}")
        else:
            for i, (s, pv) in enumerate(zip(stat, p)):
                lines.append(f"  [{i}] stat={s:.4f}, p={pv:.4g}")
        if self.df is not None:
            lines.append(f"  df: {self.df}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame, one row per estimand component."""
        stat = np.atleast_1d(self.statistic)
        p = np.atleast_1d(self.pvalue)
        return pd.DataFrame({
            "statistic": stat,
            "p_value": p,
        })
