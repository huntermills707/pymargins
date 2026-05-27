"""pymargins._result._test

Hypothesis test result type.
"""

from __future__ import annotations
from typing import Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._text import SummaryString


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
        return SummaryString("\n".join(lines))

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame, one row per estimand component."""
        stat = np.atleast_1d(self.statistic)
        p = np.atleast_1d(self.pvalue)
        return pd.DataFrame({
            "statistic": stat,
            "p_value": p,
        })


# ---------------------------------------------------------------------------
# Multiple-comparison adjustment
# ---------------------------------------------------------------------------

@dataclass
class AdjustedResults:
    """Container for multiple-comparison corrected p-values.

    Attributes
    ----------
    results : MarginsResult or list/dict thereof
        The original result(s) passed to ``adjust()``.
    p_raw : array
        Raw p-values concatenated across all input results.
    p_adj : array
        Adjusted p-values.
    reject : array
        Boolean rejection decisions at level ``alpha``.
    method : str
        Correction method name.
    alpha : float
        Significance level used.
    """
    results: Any
    p_raw: np.ndarray
    p_adj: np.ndarray
    reject: np.ndarray
    method: str
    alpha: float

    def summary(self) -> str:
        """Human-readable summary table."""
        lines = [
            f"Multiple-comparison adjustment ({self.method}, alpha={self.alpha})",
            "-" * 50,
            f"{'Index':>6}  {'Raw p':>10}  {'Adj. p':>10}  {'Reject':>8}",
            "-" * 50,
        ]
        for i, (pr, pa, rej) in enumerate(zip(self.p_raw, self.p_adj, self.reject)):
            lines.append(
                f"{i:>6}  {pr:>10.4g}  {pa:>10.4g}  {str(rej):>8}"
            )
        lines.append("-" * 50)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame."""
        return pd.DataFrame({
            "p_raw": self.p_raw,
            "p_adj": self.p_adj,
            "reject": self.reject,
            "method": self.method,
            "alpha": self.alpha,
        })


def adjust(
    results,
    method: str = "holm",
    *,
    alpha: float = 0.05,
):
    """Apply a multiple-comparison correction to a collection of results.

    Parameters
    ----------
    results : MarginsResult, dict, or list
        Result object(s) carrying test statistics. ``pvalue`` arrays are
        extracted via ``result.test()``.
    method : str, default "holm"
        One of ``"bonferroni"``, ``"holm"``, ``"sidak"``, ``"fdr_bh"``, or
        any ``statsmodels.stats.multitest`` method name.
    alpha : float, default 0.05
        Family-wise error rate or FDR level.

    Returns
    -------
    AdjustedResults
    """
    from statsmodels.stats.multitest import multipletests

    if hasattr(results, "test") and callable(results.test):
        rs = [results]
    elif isinstance(results, dict):
        rs = list(results.values())
    else:
        rs = list(results)

    pvals = np.concatenate([np.atleast_1d(r.test().pvalue) for r in rs])
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
    return AdjustedResults(
        results=results,
        p_raw=pvals,
        p_adj=p_adj,
        reject=reject,
        method=method,
        alpha=alpha,
    )
