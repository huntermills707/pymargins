"""
pymargins._result

Result types: MarginsResult, TestResult, DiagnosticResult.

Result objects are user-facing. They carry the numerical outputs of inference
(estimates, SEs, CIs, p-values), the diagnostic information that drove method
choice (κ, fallback triggers), and enough underlying machinery (gradients,
draws) to support inter-call composability via arithmetic operators.

Composability is restricted to results from the same Margins session, since
joint inference requires a shared inference scale and Σ̂. Results from
different sessions cannot be composed without explicit scale conversion.
"""

from __future__ import annotations
from typing import Optional, Union, Literal, Any
from dataclasses import dataclass, field
import weakref
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from ._inference import run_test
from ._delta import delta_confint_from_se, joint_wald_test


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
        return (
            f"Session diagnostic ({self.n_samples} design points)\n"
            f"  Session: {self.session_summary}\n"
            f"  κ min:    {self.kappa_min:.3f}\n"
            f"  κ median: {self.kappa_median:.3f}\n"
            f"  κ max:    {self.kappa_max:.3f}\n"
            f"  Verdict:  {self.verdict}\n"
            f"  {self.recommendation}"
        )


# ---------------------------------------------------------------------------
# MarginsResult
# ---------------------------------------------------------------------------

@dataclass
class MarginsResult:
    """Container for marginal-effects estimates with inference and diagnostics.

    Carries:
      - The numerical outputs (estimate, SE, CI, p-value)
      - Diagnostics (κ, simulation disagreement, fallback flag)
      - Underlying machinery (gradient, draws) for composition with other
        results from the same session

    Composability via arithmetic operators (+, -, *, /) supports building
    derived quantities from already-computed results, with proper joint
    inference using the shared Σ̂. Cross-session composition is forbidden;
    raises ValueError.

    Attributes
    ----------
    estimate : array
        Point estimate(s) on the reporting scale (after phi).

    std_error : array
        Standard error(s) on the inference scale (before phi).

    conf_int_lower, conf_int_upper : array
        CI bounds on the reporting scale.

    method : str
        Inference method used: "delta", "simulation", "bootstrap".

    level : float
        Confidence level.

    n_obs : int
        Number of observations contributing to the estimand.

    kappa : array, optional
        Curvature diagnostic per estimand component.

    delta_sim_disagreement : float, optional
        Maximum relative disagreement between delta and simulation CIs,
        when both were computed.

    fallback_triggered : bool
        Whether the requested method (typically delta) was replaced by a
        fallback (typically simulation).

    fallback_reason : str, optional
        Why fallback occurred (e.g., "kappa=0.5>threshold=0.3").

    estimand_metadata : dict
        Variable names, scenario labels, contrast specifications,
        `at` setting, etc., for output formatting.

    gradient : array, optional
        ∇h at β̂ on the inference scale. Present for delta-method results;
        used for inter-call composition.

    draws : array, optional
        Estimand evaluations at simulated/bootstrapped β. Present for
        simulation/bootstrap results; used for inter-call composition with
        other simulation/bootstrap results.

    cov_params : array, optional
        Σ̂ frozen at the time the result was produced. Used by hypothesis
        tests and inter-call composition so that downstream operations are
        not affected by later mutation or re-fitting of the model wrapped
        by ``session``.

    phi : callable, optional
        Back-transform from inference scale to reporting scale. Captured at
        construction so reporting works even if the session is garbage
        collected.

    phi_inv : callable, optional
        Forward transform from reporting scale to inference scale. Captured
        at construction for the same reason as ``phi``.

    session : Margins
        Reference to the originating session. Used to validate composability
        (same-session check). Σ̂ is read from ``cov_params`` rather than
        re-fetched from ``session.adapter`` to make results robust to model
        mutation and to make ``materialize()`` semantically clean.
    """
    estimate: np.ndarray
    std_error: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    method: str
    level: float
    n_obs: int = 0
    kappa: Optional[np.ndarray] = None
    delta_sim_disagreement: Optional[float] = None
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    estimand_metadata: dict = field(default_factory=dict)
    gradient: Optional[np.ndarray] = None
    draws: Optional[np.ndarray] = None
    cov_params: Optional[np.ndarray] = None
    phi: Optional[Callable] = None
    phi_inv: Optional[Callable] = None
    session: Optional[Any] = None

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def _session_obj(self):
        """Dereference the session, handling weak references."""
        if self.session is None:
            return None
        if isinstance(self.session, weakref.ref):
            return self.session()
        return self.session

    def summary(self) -> str:
        """Human-readable summary including diagnostics."""
        lines = ["MarginsResult"]
        lines.append(f"  Method: {self.method}, level={self.level}")
        if self.fallback_triggered:
            lines.append(f"  WARNING — Fallback triggered: {self.fallback_reason}")
        est = np.atleast_1d(self.estimate)
        se = np.atleast_1d(self.std_error)
        lo = np.atleast_1d(self.conf_int_lower)
        hi = np.atleast_1d(self.conf_int_upper)
        labels = self.estimand_metadata.get("labels")

        for i in range(est.size):
            label = (labels[i] if labels and i < len(labels)
                     else f"estimand[{i}]")
            lines.append(
                f"  {label}: estimate={est[i]:.4f}  "
                f"SE={se[i]:.4f}  "
                f"CI=[{lo[i]:.4f}, {hi[i]:.4f}]"
            )
        # Mention scale asymmetry when inference and reporting scales differ
        if self.phi is not None:
            lines.append(
                "  Note: SE is on the inference scale; estimate and CI are on the reporting scale."
            )
        if self.kappa is not None:
            kappa_str = (f"{float(self.kappa):.3f}"
                         if np.ndim(self.kappa) == 0
                         else f"max={float(np.max(self.kappa)):.3f}")
            lines.append(f"  κ: {kappa_str}")
        if self.delta_sim_disagreement is not None:
            lines.append(
                f"  Delta-vs-sim disagreement: {self.delta_sim_disagreement:.3%}"
            )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame.

        One row per estimand component. Columns include estimate, SE, CI
        bounds, κ (if available), and any labels from estimand_metadata.
        """
        est = np.atleast_1d(self.estimate)
        se = np.atleast_1d(self.std_error)
        lo = np.atleast_1d(self.conf_int_lower)
        hi = np.atleast_1d(self.conf_int_upper)

        data = {
            "estimate": est,
            "estimate_scale": np.repeat("reporting", est.size),
            "std_error": se,
            "se_scale": np.repeat("inference", se.size),
            "ci_lower": lo,
            "ci_upper": hi,
            "ci_scale": np.repeat("reporting", est.size),
        }
        if self.kappa is not None:
            kvals = np.atleast_1d(self.kappa)
            if kvals.size == est.size:
                data["kappa"] = kvals

        labels = self.estimand_metadata.get("labels")
        if labels and len(labels) == est.size:
            data["label"] = labels

        return pd.DataFrame(data)

    def conf_int(self, level: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Recompute CI at a different confidence level.

        For delta-method results, uses the stored gradient and Σ̂ to
        construct a CI at the new level without re-running inference. For
        simulation/bootstrap results, recomputes quantiles of the stored
        draws.

        Parameters
        ----------
        level : float, optional
            New confidence level. If None, returns the stored CI.

        Returns
        -------
        (lower, upper) : tuple of arrays
        """
        if level is None or level == self.level:
            return self.conf_int_lower, self.conf_int_upper

        if self.gradient is not None:
            # Recompute via delta
            est_inf = self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
            lower, upper = delta_confint_from_se(
                est_inf, self.std_error, level=level, phi=self.phi,
            )
            return np.asarray(lower), np.asarray(upper)
        elif self.draws is not None:
            alpha = (1.0 - level) / 2.0
            lower = np.quantile(self.draws, alpha, axis=0)
            upper = np.quantile(self.draws, 1.0 - alpha, axis=0)
            return lower, upper
        else:
            raise ValueError(
                "Cannot recompute CI: result has neither gradient nor draws."
            )

    # -----------------------------------------------------------------------
    # Hypothesis tests
    # -----------------------------------------------------------------------

    def test(
        self,
        value: float = 0.0,
        kind: str = "wald",
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
        null_scale: Literal["reporting", "inference"] = "reporting",
    ) -> TestResult:
        """Test H₀: estimand = value (per-component).

        For multi-row results, returns per-component tests. For joint testing,
        use joint_test.

        Parameters
        ----------
        value : float, default 0.0
            Hypothesized value. By default interpreted on the reporting
            scale (what ``estimate`` is reported on, after ``phi``). For
            common nulls under non-identity scales, this is what users
            typically want — under ``log_scale`` the natural "no effect"
            null is RR=1, written ``value=1.0``; the test internally
            converts via ``phi_inv`` to the inference-scale null log(1)=0.

        kind : str, default "wald"
            Test type.

        alternative : str

        null_scale : {"reporting", "inference"}, default "reporting"
            Which scale ``value`` is supplied on.
              "reporting"  : interpret on the reporting scale (after ``phi``);
                             the test applies ``phi_inv`` to obtain the
                             inference-scale null. Use ``value=1.0`` for the
                             natural null under ``log_scale``, ``value=0.5``
                             under ``logit_scale`` (when phi=expit), etc.
              "inference"  : interpret directly on the inference scale; no
                             transformation is applied. Use ``value=0.0``
                             for the natural null on any session whose
                             inference scale represents "no effect" at zero
                             (log, logit, fisher_z, identity).

            For identity-scale sessions (``phi=None``) the two are
            equivalent.

        Returns
        -------
        result : TestResult
        """
        if null_scale == "inference" or self.phi_inv is None:
            null_inf = value
        elif null_scale == "reporting":
            null_inf = self.phi_inv(value)
        else:
            raise ValueError(
                f"null_scale must be 'reporting' or 'inference', got {null_scale!r}"
            )

        # Get inference-scale estimate and draws
        est_inf = self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
        draws_inf = self.phi_inv(self.draws) if self.phi_inv is not None and self.draws is not None else self.draws

        cov = self.cov_params if self.gradient is not None else None

        statistic, pvalue = run_test(
            estimate=np.asarray(est_inf),
            grad=self.gradient,
            cov_params=cov,
            draws=draws_inf,
            null_value=null_inf,
            alternative=alternative,
            method=kind,
        )
        return TestResult(
            statistic=statistic,
            pvalue=pvalue,
            null_value=value,
            alternative=alternative,
            method=kind,
            estimand_metadata=self.estimand_metadata,
        )

    def joint_test(
        self,
        value: Optional[np.ndarray] = None,
        kind: str = "wald",
        null_scale: Literal["reporting", "inference"] = "reporting",
    ) -> TestResult:
        """Joint test H₀: all estimand components equal (vector-valued) value.

        Tests the joint null using χ² with df = number of components.

        Parameters
        ----------
        value : array, optional
            Hypothesized vector. By default interpreted on the reporting
            scale (see ``test`` for full discussion). Defaults to a zero
            vector, which under ``null_scale="reporting"`` is the natural
            null only on identity scale; under log/logit/fisher_z the
            natural "no effect" null on the reporting scale is 1, 0.5,
            and 0 respectively. To pass an inference-scale zero directly
            (the universal "no effect" point on the inference scale), use
            ``null_scale="inference"``.

        kind : str, default "wald"

        null_scale : {"reporting", "inference"}, default "reporting"
            Which scale ``value`` is supplied on. See ``test`` for details.

        Returns
        -------
        result : TestResult
        """
        if self.gradient is None:
            raise NotImplementedError(
                "Joint test currently requires a delta-method result with "
                "gradients. For simulation/bootstrap, use the empirical "
                "joint distribution from result.draws."
            )
        if self.cov_params is None:
            raise ValueError(
                "Joint test requires Σ̂ on the result; this result has no "
                "frozen cov_params (likely materialized without it)."
            )

        if value is None:
            # Default null = zero on the inference scale (the universal
            # "no effect" point regardless of phi).
            value_inf = jnp.zeros_like(jnp.asarray(self.estimate))
        elif null_scale == "inference" or self.phi_inv is None:
            value_inf = jnp.asarray(value)
        elif null_scale == "reporting":
            value_inf = self.phi_inv(jnp.asarray(value))
        else:
            raise ValueError(
                f"null_scale must be 'reporting' or 'inference', got {null_scale!r}"
            )

        est_inf = self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
        cov = jnp.asarray(self.cov_params)

        chi2, p, df = joint_wald_test(
            jnp.asarray(est_inf),
            jnp.asarray(self.gradient),
            cov,
            null_value=value_inf,
        )
        return TestResult(
            statistic=np.asarray(chi2),
            pvalue=np.asarray(p),
            df=df,
            null_value=value if value is not None else 0.0,
            alternative="two-sided",
            method=f"joint_{kind}",
            estimand_metadata=self.estimand_metadata,
        )

    # -----------------------------------------------------------------------
    # Composability across calls (same session only)
    # -----------------------------------------------------------------------

    def _check_compatible(self, other: "MarginsResult") -> None:
        """Verify two results came from the same session and are composable."""
        self_sess = self._session_obj()
        other_sess = other._session_obj()
        if self_sess is None or other_sess is None:
            raise ValueError(
                "Composition requires both results to carry a session reference."
            )
        if self_sess is not other_sess:
            raise ValueError(
                "Cannot compose results from different Margins sessions. "
                "Different sessions may have different inference scales and "
                "covariances; composition is not well-defined."
            )

    def __sub__(self, other: "MarginsResult") -> "MarginsResult":
        """Difference of two estimands with proper joint inference.

        Computes the delta-method variance of the difference using the joint
        gradient and the shared Σ̂ from the session. Available only when
        both results carry gradients (delta-method results).
        """
        self._check_compatible(other)
        return _combine_results(
            self, other, lambda a, b: a - b,
            grad_combine=lambda g1, g2: g1 - g2,
            label_combine=lambda l1, l2: f"({l1}) - ({l2})",
        )

    def __add__(self, other: "MarginsResult") -> "MarginsResult":
        self._check_compatible(other)
        return _combine_results(
            self, other, lambda a, b: a + b,
            grad_combine=lambda g1, g2: g1 + g2,
            label_combine=lambda l1, l2: f"({l1}) + ({l2})",
        )

    def __mul__(self, other) -> "MarginsResult":
        if isinstance(other, MarginsResult):
            raise NotImplementedError(
                "Product of two MarginsResults is nonlinear; use evaluate() "
                "with a custom compose function instead."
            )
        scalar = float(other)

        # Scale-aware: SE/gradient/draws are on the inference scale;
        # estimate and CI bounds are on the reporting scale.
        if self.phi is not None and self.phi_inv is not None:
            new_est = np.asarray(self.phi(scalar * self.phi_inv(self.estimate)))
            lo = np.asarray(self.phi(scalar * self.phi_inv(self.conf_int_lower)))
            hi = np.asarray(self.phi(scalar * self.phi_inv(self.conf_int_upper)))
            if scalar < 0:
                lo, hi = hi, lo
            new_lo, new_hi = lo, hi
            new_draws = (
                np.asarray(self.phi(scalar * self.phi_inv(self.draws)))
                if self.draws is not None else None
            )
        else:
            new_est = self.estimate * scalar
            new_lo = (self.conf_int_lower * scalar
                      if scalar > 0 else self.conf_int_upper * scalar)
            new_hi = (self.conf_int_upper * scalar
                      if scalar > 0 else self.conf_int_lower * scalar)
            new_draws = self.draws * scalar if self.draws is not None else None

        return MarginsResult(
            estimate=new_est,
            std_error=self.std_error * abs(scalar),
            conf_int_lower=new_lo,
            conf_int_upper=new_hi,
            method=self.method,
            level=self.level,
            kappa=self.kappa,
            estimand_metadata={**self.estimand_metadata,
                               "labels": [f"({l})*{scalar}"
                                          for l in self.estimand_metadata.get("labels", [])]},
            gradient=(self.gradient * scalar
                      if self.gradient is not None else None),
            draws=new_draws,
            cov_params=self.cov_params,
            phi=self.phi,
            phi_inv=self.phi_inv,
            session=self.session,
        )

    def __truediv__(self, other) -> "MarginsResult":
        if isinstance(other, MarginsResult):
            raise NotImplementedError(
                "Ratio of two MarginsResults is nonlinear; use evaluate() "
                "with a custom compose function (e.g., compose=lambda p: p[0]/p[1]) "
                "for proper inference."
            )
        return self.__mul__(1.0 / float(other))

    # -----------------------------------------------------------------------
    # Cosmetic transformations (don't affect inference)
    # -----------------------------------------------------------------------

    def scaled(self, by: float, units: Optional[str] = None) -> "MarginsResult":
        """Cosmetic rescaling of the estimate and CI for reporting.

        Multiplies the estimate, SE, and CI bounds by `by`. Useful for unit
        conversion or percentage display. Does not affect any inference
        properties — the underlying gradients/draws are also rescaled so
        composability is preserved.

        Parameters
        ----------
        by : float
            Scaling factor. May be negative (flips direction).

        units : str, optional
            Label for the new scale; recorded in metadata.

        Returns
        -------
        scaled : MarginsResult
        """
        new = self * by
        if units:
            new.estimand_metadata["units"] = units
        return new

    def materialize(self) -> "MarginsResult":
        """Drop underlying machinery (gradient, draws, session) to reduce
        memory.

        After materialize(), the result is no longer composable with other
        results. Useful for storing many results long-term where you don't
        need to combine them further. Reporting (summary, to_frame, conf_int
        at the stored level) still works.
        """
        return MarginsResult(
            estimate=self.estimate,
            std_error=self.std_error,
            conf_int_lower=self.conf_int_lower,
            conf_int_upper=self.conf_int_upper,
            method=self.method,
            level=self.level,
            n_obs=self.n_obs,
            kappa=self.kappa,
            delta_sim_disagreement=self.delta_sim_disagreement,
            fallback_triggered=self.fallback_triggered,
            fallback_reason=self.fallback_reason,
            estimand_metadata=self.estimand_metadata,
            gradient=None,
            draws=None,
            cov_params=None,
            phi=self.phi,
            phi_inv=self.phi_inv,
            session=None,
        )


# ---------------------------------------------------------------------------
# Internal: result combination helper
# ---------------------------------------------------------------------------

def _combine_results(
    a: MarginsResult,
    b: MarginsResult,
    estimate_combine,
    grad_combine,
    label_combine,
) -> MarginsResult:
    """Combine two results from the same session via a linear operation."""
    # Inference-scale estimates and combined gradient
    a_inf = a.phi_inv(a.estimate) if a.phi_inv is not None else a.estimate
    b_inf = b.phi_inv(b.estimate) if b.phi_inv is not None else b.estimate
    combined_inf = estimate_combine(a_inf, b_inf)

    if a.gradient is None or b.gradient is None:
        raise NotImplementedError(
            "Composition currently requires delta-method results (with "
            "gradients). Simulation/bootstrap composition would require "
            "matched draws; not yet implemented."
        )

    new_grad = grad_combine(a.gradient, b.gradient)

    # Guard against vector-valued results — composition only defined for scalars
    if jnp.ndim(new_grad) != 1:
        raise NotImplementedError(
            "Composition is only supported for scalar estimands. "
            "For vector results, compose elementwise or use evaluate()."
        )

    # New SE and CI from delta on the combined gradient.
    # Both results are from the same session and were produced with the
    # same vcov_spec, so a.cov_params is the canonical Σ̂ for composition.
    if a.cov_params is None:
        raise ValueError(
            "Composition requires Σ̂ on the result (cov_params). The "
            "originating session should have populated it; if this result "
            "was constructed manually, supply cov_params."
        )
    cov = jnp.asarray(a.cov_params)
    var = jnp.dot(jnp.asarray(new_grad), cov @ jnp.asarray(new_grad))
    se = float(jnp.sqrt(var))

    z = stats.norm.ppf(0.5 + a.level / 2.0)
    lo_inf = combined_inf - z * se
    hi_inf = combined_inf + z * se

    if a.phi is not None:
        estimate_report = np.asarray(a.phi(combined_inf))
        lower_report = np.asarray(a.phi(lo_inf))
        upper_report = np.asarray(a.phi(hi_inf))
    else:
        estimate_report = combined_inf
        lower_report = lo_inf
        upper_report = hi_inf

    a_label = (a.estimand_metadata.get("labels", [""])[0]
               if a.estimand_metadata.get("labels") else "A")
    b_label = (b.estimand_metadata.get("labels", [""])[0]
               if b.estimand_metadata.get("labels") else "B")

    return MarginsResult(
        estimate=np.asarray(estimate_report),
        std_error=np.asarray(se),
        conf_int_lower=np.asarray(lower_report),
        conf_int_upper=np.asarray(upper_report),
        method=a.method,
        level=a.level,
        kappa=None,  # not recomputed for combined results
        estimand_metadata={"labels": [label_combine(a_label, b_label)]},
        gradient=new_grad,
        draws=None,
        cov_params=a.cov_params,
        session=a.session,
    )


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: Basic reporting
--------------------------

    result = m.predict(at={"treatment": 1})
    print(result.summary())
    df = result.to_frame()


Example 2: Hypothesis test
--------------------------

    result = m.contrasts(...)  # returns a MarginsResult
    test = result.test(value=0.0)
    print(test.summary())
    print(f"p = {float(test.pvalue):.4f}")


Example 3: Inter-call composability
-----------------------------------

    ame_overall = m.dydx("treatment")
    ame_old     = m.dydx("treatment", atexog={"age_group": "65+"})
    ame_young   = m.dydx("treatment", atexog={"age_group": "18-44"})

    # Joint inference computed using shared Σ̂
    deviation = ame_old - ame_overall
    test = deviation.test(value=0.0)


Example 4: Cosmetic rescaling for reporting
-------------------------------------------

    result = m.predict()                  # estimate in proportion units
    pct = result.scaled(by=100, units="%")
    print(pct.summary())                   # estimate now in percentage points


Example 5: Recompute CI at different level
------------------------------------------

    result = m.contrasts(...)              # default 95% CI
    lo90, hi90 = result.conf_int(level=0.90)
"""
