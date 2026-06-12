"""GraphResult — doctrine-surface result object.

Implements the result contract from design §7.1 and req. §6.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from ._margins import MarginsResult


class GraphResult:
    """Result object for the new estimator surface.

    Wraps a :class:`MarginsResult` internally (the numbers are identical)
    but exposes the doctrine API: no ``level=`` in ``conf_int``,
    plan metadata in ``summary``, and self-contained ``influence()``.
    """

    def __init__(self, margins_result: MarginsResult, plan: Any):
        self._result = margins_result
        self._plan = plan

    @classmethod
    def _from_margins_result(cls, result: MarginsResult, plan: Any) -> GraphResult:
        return cls(result, plan)

    # ------------------------------------------------------------------
    # Delegate read-only attributes
    # ------------------------------------------------------------------

    @property
    def estimate(self):
        return self._result.estimate

    @property
    def std_error(self):
        return self._result.std_error

    @property
    def conf_int_lower(self):
        return self._result.conf_int_lower

    @property
    def conf_int_upper(self):
        return self._result.conf_int_upper

    @property
    def method(self):
        return self._result.method

    @property
    def level(self):
        return self._result.level

    @property
    def n_obs(self):
        return self._result.n_obs

    @property
    def kappa(self):
        return self._result.kappa

    # ------------------------------------------------------------------
    # Doctrine surface methods
    # ------------------------------------------------------------------

    def conf_int(self, correction: str | None = None):
        """Confidence intervals with optional family correction.

        Parameters
        ----------
        correction : {None, "bonferroni", "sidak", "sup-t"}, optional
            Family-wise correction applied to the *declared* level.
            No ``level=`` parameter — the level is locked at construction.
        """

        if correction is None:
            return self._result.conf_int()

        est = np.asarray(self._result.estimate)
        k = est.size if est.ndim > 0 else 1
        level = self._result.level

        if correction == "sup-t":
            return self._result.conf_int(simultaneous=True)

        if correction == "bonferroni":
            # Per-component level = 1 - (1 - overall_level) / k
            adjusted_level = 1.0 - (1.0 - level) / k
            return self._result.conf_int(level=adjusted_level)

        if correction == "sidak":
            # Per-component level = (overall_level)^(1/k)
            adjusted_level = level ** (1.0 / k)
            return self._result.conf_int(level=adjusted_level)

        raise ValueError(
            f"correction={correction!r} is not supported. "
            f"Supported: None, 'bonferroni', 'sidak', 'sup-t'."
        )

    def test(self, value=0, *, null_scale="reporting"):
        return self._result.test(value=value, null_scale=null_scale)

    def joint_test(self, value=0, *, kind="wald"):
        return self._result.joint_test(value=value, kind=kind)

    def summary(self):
        """Human-readable summary with plan footer."""
        base = self._result.summary()
        # Append plan hash to the summary string
        footer = f"\nplan {self._plan.hash}"
        if self._plan.population_note:
            footer += f" | population: {self._plan.population_note}"
        if self._result.kappa is not None:
            footer += f" | κ = {self._result.kappa:.3f}"
        return base + footer

    def to_frame(self):
        return self._result.to_frame()

    def to_latex(self):
        return self._result.to_latex()

    def to_html(self):
        return self._result.to_html()

    def outcome(self):
        return self._result.outcome()

    def scaled(self, by: float, units: str = ""):
        return GraphResult(self._result.scaled(by=by, units=units), self._plan)

    def contrast(self, C, labels=None):
        return GraphResult(self._result.contrast(C, labels=labels), self._plan)

    def pairwise_contrasts(self):
        return GraphResult(self._result.pairwise_contrasts(), self._plan)

    def influence(self):
        return self._result.influence()

    def to_disk(self, path: str, format: str = "pickle"):
        """Serialize result + plan to disk."""
        with open(path, "wb") as f:
            pickle.dump({"result": self._result, "plan": self._plan}, f)

    @classmethod
    def from_disk(cls, path: str):
        """Deserialize result + plan from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["result"], data["plan"])
