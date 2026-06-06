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

import dataclasses
import pickle
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from .._delta import delta_confint_from_se, joint_wald_test
from .._inference import run_test
from ._test import TestResult

# ---------------------------------------------------------------------------
# Hypothesis test result
# ---------------------------------------------------------------------------


@dataclass
class MarginsResult:
    """Container for marginal-effects estimates with inference and diagnostics.

    Carries:

    - The numerical outputs (estimate, SE, CI, p-value)
    - Diagnostics (κ, simulation disagreement, fallback flag)
    - Underlying machinery (gradient, draws) for composition with other
      results from the same session

    Composability via arithmetic operators (``+``, ``-``, ``*``, ``/``)
    supports building derived quantities from already-computed results,
    with proper joint inference using the shared Σ̂. Cross-session
    composition is forbidden; raises ValueError.

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
        Estimand evaluations at simulated/bootstrapped β on the reporting
        scale (after ``phi``). Present for simulation/bootstrap results.

    draws_inf : array, optional
        Estimand evaluations on the inference scale (before ``phi``).
        Present for simulation/bootstrap results; used for recomputing CIs
        with alternative bootstrap methods.

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

    ci_method : str, optional
        CI method used for bootstrap results: "percentile", "basic",
        "bca", or "studentized".

    bootstrap_extras : dict, optional
        Method-specific data for recomputing CIs (e.g., BCa z0/a,
        studentized t-star draws).

    n_boot_effective : int, optional
        Number of successful bootstrap replicates (may be less than
        ``n_boot`` if some replicates failed to refit).

    n_boot_failed : int, optional
        Number of bootstrap replicates that failed to refit.
    """

    estimate: np.ndarray
    std_error: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    method: str
    level: float
    n_obs: int = 0
    kappa: np.ndarray | None = None
    delta_sim_disagreement: float | None = None
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    estimand_metadata: dict = field(default_factory=dict)
    gradient: np.ndarray | None = None
    draws: np.ndarray | None = None
    draws_inf: np.ndarray | None = None
    cov_params: np.ndarray | None = None
    phi: Callable | None = None
    phi_inv: Callable | None = None
    session: Any | None = None
    ci_method: str | None = None
    bootstrap_extras: dict | None = None
    resample_bank_id: str | None = None
    n_boot_effective: int | None = None
    n_boot_failed: int | None = None
    imputation_diagnostic: Any | None = None
    """Rubin pooling diagnostic; present only on pool_imputations() output."""

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

    def _summary_rows(self):
        """Build per-row summary data as a list of dicts.

        Per-row p-values test H0: estimate = 0 on the *inference scale*.
        For logit-scale predictions this means testing against p = 0.5,
        which may not be the intended null.
        """
        est = np.atleast_1d(self.estimate).ravel()
        se = np.atleast_1d(self.std_error).ravel()
        lo = np.atleast_1d(self.conf_int_lower).ravel()
        hi = np.atleast_1d(self.conf_int_upper).ravel()
        labels = self.estimand_metadata.get("labels")
        if labels is None:
            labels = [f"[{i}]" for i in range(est.size)]

        z_vals = []
        p_vals = []
        stat_label = "z" if self.gradient is not None else "statistic"
        # Note: null=0 is on the inference scale. For logit-scale predictions,
        # this tests H0: logit(p)=0 i.e. p=0.5, which is rarely the intended null.
        try:
            tr = self.test(value=0.0, null_scale="inference")
            z_vals = np.atleast_1d(tr.statistic).ravel()
            p_vals = np.atleast_1d(tr.pvalue).ravel()
            if self.imputation_diagnostic is not None:
                stat_label = "t"
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"Test statistics omitted from summary: {exc}", stacklevel=2)

        rows = []
        for i in range(est.size):
            row = {
                "label": labels[i] if i < len(labels) else f"[{i}]",
                "estimate": float(est[i]),
                "std_error": float(se[i]),
                "ci_lower": float(lo[i]),
                "ci_upper": float(hi[i]),
            }
            if i < len(z_vals):
                row["statistic"] = float(z_vals[i])
                row["pvalue"] = float(p_vals[i])
                row["stat_label"] = stat_label
            rows.append(row)
        return rows

    @staticmethod
    def _star_notation(p: float, levels: tuple[float, float, float]) -> str:
        """Return significance stars for a p-value."""
        if p < levels[0]:
            return "***"
        elif p < levels[1]:
            return "**"
        elif p < levels[2]:
            return "*"
        return ""

    def summary(
        self,
        stars: bool = False,
        star_levels: tuple[float, float, float] = (0.01, 0.05, 0.10),
        float_fmt: str = ".4f",
        pvalue_fmt: str = ".3f",
        max_rows: int | None = None,
    ) -> str:
        """Human-readable summary including diagnostics.

        Mimics the tabular style of statsmodels ``summary()`` tables: aligned
        columns, separator lines, and optional significance stars.

        Parameters
        ----------
        stars : bool, default False
            Append significance stars to estimates.
        star_levels : tuple of float, default (0.01, 0.05, 0.10)
            Thresholds for ***, **, * respectively.
        float_fmt : str, default ".4f"
            Format string for floating-point columns.
        pvalue_fmt : str, default ".3f"
            Format string for p-values.
        max_rows : int, optional
            Truncate table to this many rows (showing ``...`` for remainder).

        Notes
        -----
        Per-row p-values test H0: estimate = 0 on the inference scale. For
        logit-scale predictions this corresponds to p = 0.5, which may not
        be the intended null hypothesis.

        Returns
        -------
        str
        """
        rows = self._summary_rows()
        has_stat = any("statistic" in r for r in rows)
        n_total = len(rows)
        truncated = False
        if max_rows is not None and n_total > max_rows:
            rows = rows[:max_rows]
            truncated = True

        # Column specification: (data_key, header_text)
        data_keys = [("estimate", "estimate"), ("std_error", "std err")]
        if has_stat:
            stat_header = rows[0].get("stat_label", "z") if rows else "z"
            p_header = "P>|t|" if stat_header == "t" else "P>|z|"
            data_keys.extend([("statistic", stat_header), ("pvalue", p_header)])
        data_keys.append(("ci", f"[{self.level * 100:.0f}% Conf. Int.]"))

        def _fmt(key, row):
            if key == "ci":
                return f"{row['ci_lower']:{float_fmt}}, {row['ci_upper']:{float_fmt}}"
            if key == "pvalue":
                s = f"{row['pvalue']:{pvalue_fmt}}"
                if s.startswith("-0.000"):
                    s = "0.000"
                return s
            val = row[key]
            if key == "estimate" and stars and "pvalue" in row:
                star = self._star_notation(row["pvalue"], star_levels)
                return f"{val:{float_fmt}}{star}"
            return f"{val:{float_fmt}}"

        # Build formatted rows
        fmt_rows = []
        for r in rows:
            fmt_rows.append([r["label"]] + [_fmt(k, r) for k, _ in data_keys])

        all_display = [[""] + [h for _, h in data_keys]] + fmt_rows
        widths = [
            max(len(r[i]) for r in all_display) for i in range(len(data_keys) + 1)
        ]

        def _line(cells, aligns):
            parts = []
            for i, c in enumerate(cells):
                a = aligns[i]
                w = widths[i]
                if a == "r":
                    parts.append(f"{c:>{w}}")
                elif a == "c":
                    parts.append(f"{c:^{w}}")
                else:
                    parts.append(f"{c:<{w}}")
                if i < len(cells) - 1:
                    parts.append("  ")
            return "".join(parts)

        aligns = ["l"] + ["r"] * len(data_keys)
        total_width = sum(widths) + 2 * (len(widths) - 1)
        sep = "=" * total_width
        thin = "-" * total_width

        out_lines = [sep]
        title = f"Margins Result ({self.method}, level={self.level})"
        out_lines.append(title.center(total_width))
        out_lines.append(sep)
        out_lines.append(_line([""] + [h for _, h in data_keys], aligns))
        out_lines.append(thin)
        for r in fmt_rows:
            out_lines.append(_line(r, aligns))
        if truncated:
            out_lines.append(_line(["..."] + [""] * len(data_keys), aligns))
        out_lines.append(sep)

        # Footer
        footers = []
        if self.n_obs:
            footers.append(f"n = {self.n_obs}")
        if self.fallback_triggered:
            footers.append(f"WARNING — Fallback triggered: {self.fallback_reason}")
        if self.phi is not None:
            footers.append(
                "Note: std err is on the inference scale; estimate and CI are on the reporting scale."
            )
        if self.kappa is not None:
            k = np.asarray(self.kappa)
            if not np.all(np.isnan(k)):
                kappa_str = (
                    f"{float(k):.3f}"
                    if k.ndim == 0
                    else f"max={float(np.nanmax(k)):.3f}"
                )
                footers.append(f"κ: {kappa_str}")
        if self.delta_sim_disagreement is not None:
            footers.append(
                f"Delta-vs-sim disagreement: {self.delta_sim_disagreement:.3%}"
            )
        if self.imputation_diagnostic is not None:
            footers.append(self.imputation_diagnostic.footer())
        if footers:
            out_lines.extend([""] + footers)

        return "\n".join(out_lines)

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame.

        One row per estimand component. Columns include estimate, SE, CI
        bounds, κ (if available), and any labels from estimand_metadata.

        Scenario columns (e.g. ``x1``, ``group``) are unpacked from
        ``estimand_metadata["scenarios"]`` when available, making the
        DataFrame ready for plotting without string parsing.
        """
        est = np.atleast_1d(self.estimate)
        se = np.atleast_1d(self.std_error)
        lo = np.atleast_1d(self.conf_int_lower)
        hi = np.atleast_1d(self.conf_int_upper)

        # G7a: structured multi-dimensional output — flatten if _outcome_shape
        # is recorded, producing an "outcome" column.
        outcome_shape = self.estimand_metadata.get("_outcome_shape")
        if outcome_shape is not None and est.ndim == 2:
            n_atoms = outcome_shape["n_atoms"]
            n_outcomes = outcome_shape["n_outcomes"]
            outcome_labels = outcome_shape["outcome_labels"]
            # Flatten column-major (atom-major) to match expanded labels
            est = est.ravel(order="C")
            se = se.ravel(order="C")
            lo = lo.ravel(order="C")
            hi = hi.ravel(order="C")
            n = est.size
            outcome_col = np.tile(outcome_labels, n_atoms)
        else:
            n = est.size
            outcome_col = None

        # Core estimate columns
        data: dict[str, Any] = {
            "estimate": est,
            "std_error": se,
            "ci_lower": lo,
            "ci_upper": hi,
            "conf_level": np.repeat(self.level, n),
            "n_obs": np.repeat(self.n_obs, n),
            "method": np.repeat(self.method, n),
        }

        if outcome_col is not None:
            data["outcome"] = outcome_col

        # Term: what this estimand is about
        kind = self.estimand_metadata.get("kind", "")
        variables = self.estimand_metadata.get("variables")
        labels = self.estimand_metadata.get("labels")
        if variables is not None:
            data["term"] = [list(variables) for _ in range(n)]
        elif labels is not None and len(labels) == n:
            data["term"] = labels
        else:
            data["term"] = np.repeat("", n)

        data["kind"] = np.repeat(kind, n)

        # Labels
        if labels is not None and len(labels) == n:
            data["label"] = labels

        # Statistics and p-values (try to compute if not cached)
        try:
            tr = self.test(value=0.0, null_scale="inference")
            z_vals = np.atleast_1d(tr.statistic)
            p_vals = np.atleast_1d(tr.pvalue)
            if outcome_shape is not None and z_vals.ndim == 2:
                z_vals = z_vals.ravel(order="C")
                p_vals = p_vals.ravel(order="C")
            if z_vals.size == n:
                data["statistic"] = z_vals
                data["p_value"] = p_vals
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            warnings.warn(f"Test statistics omitted from to_frame: {exc}", stacklevel=2)

        # Over info
        over = self.estimand_metadata.get("over")
        if over is not None:
            data["over"] = np.repeat(",".join(over), n)
            # Try to read over values from explicit metadata first
            over_values_meta = self.estimand_metadata.get("_over_values")
            if over_values_meta is not None and len(over_values_meta) == n:
                data["over_value"] = [
                    ", ".join(str(ov[o]) for o in over) for ov in over_values_meta
                ]
            elif labels is not None and len(labels) == n:
                over_values = []
                for lab in labels:
                    vals = []
                    for o in over:
                        import re

                        m = re.search(rf"{o}=([^,]+)", lab)
                        if m:
                            vals.append(m.group(1).strip())
                    over_values.append(", ".join(vals) if vals else "")
                data["over_value"] = over_values

        # Diagnostics
        if self.kappa is not None:
            kvals = np.atleast_1d(self.kappa)
            if outcome_shape is not None and kvals.ndim == 2:
                kvals = kvals.ravel(order="C")
            if kvals.size == n:
                data["kappa"] = kvals
        data["fallback_triggered"] = np.repeat(self.fallback_triggered, n)
        if self.fallback_reason:
            data["fallback_reason"] = np.repeat(self.fallback_reason, n)

        # Scenario columns
        scenarios = self.estimand_metadata.get("scenarios")
        kind = self.estimand_metadata.get("kind")
        if scenarios is not None and kind in ("prediction", "slope", None):
            if len(scenarios) == n:
                # 1-to-1 match: unpack directly
                all_keys = sorted(set().union(*(s.keys() for s in scenarios)))
                for key in all_keys:
                    col_values = [s.get(key, np.nan) for s in scenarios]
                    data[key] = col_values
            elif (
                outcome_shape is not None and len(scenarios) == outcome_shape["n_atoms"]
            ):
                # Multi-scenario × multi-outcome: tile each scenario once per
                # outcome to match the atom-major / outcome-minor ravel order.
                n_outcomes = outcome_shape["n_outcomes"]
                tiled = []
                for s in scenarios:
                    tiled.extend([s] * n_outcomes)
                all_keys = sorted(set().union(*(s.keys() for s in scenarios)))
                for key in all_keys:
                    col_values = [s.get(key, np.nan) for s in tiled]
                    data[key] = col_values
            elif len(scenarios) > 1 and outcome_shape is not None:
                # Cannot determine tiling: raise rather than silently drop.
                raise ValueError(
                    "to_frame() cannot unpack scenario columns for this "
                    "multi-outcome result. Use outcome=... to slice to a "
                    "single outcome first, or call outcome().to_frame()."
                )
            # Single scenario or non-multi-outcome vector estimand: skip silently

        return pd.DataFrame(data)

    def to_latex(
        self,
        stars: bool = False,
        star_levels: tuple[float, float, float] = (0.01, 0.05, 0.10),
        float_fmt: str = ".4f",
        pvalue_fmt: str = ".3f",
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        """LaTeX tabular representation of the result.

        Parameters
        ----------
        stars : bool, default False
            Append significance stars to estimates.
        star_levels : tuple of float, default (0.01, 0.05, 0.10)
            Thresholds for ***, **, * respectively.
        float_fmt : str, default ".4f"
            Format string for floating-point columns.
        pvalue_fmt : str, default ".3f"
            Format string for p-values.
        caption : str, optional
            Table caption (wraps in table environment if provided).
        label : str, optional
            Table label (requires caption).

        Returns
        -------
        str
        """
        rows = self._summary_rows()
        has_stat = any("statistic" in r for r in rows)

        data_keys = [("estimate", "estimate"), ("std_error", "std err")]
        if has_stat:
            stat_header = rows[0].get("stat_label", "z") if rows else "z"
            p_header = "P>|t|" if stat_header == "t" else "P>|z|"
            data_keys.extend([("statistic", stat_header), ("pvalue", p_header)])
        data_keys.append(("ci", f"[{self.level * 100:.0f}\\% Conf. Int.]"))

        def _fmt(key, row):
            if key == "ci":
                return f"{row['ci_lower']:{float_fmt}}, {row['ci_upper']:{float_fmt}}"
            if key == "pvalue":
                s = f"{row['pvalue']:{pvalue_fmt}}"
                if s.startswith("-0.000"):
                    s = "0.000"
                return s
            val = row[key]
            if key == "estimate" and stars and "pvalue" in row:
                star = self._star_notation(row["pvalue"], star_levels)
                return f"{val:{float_fmt}}{star}"
            return f"{val:{float_fmt}}"

        lines = []
        lines.append("\\begin{tabular}{l" + "r" * (len(data_keys) - 1) + "}")
        lines.append("\\hline")
        lines.append("\\hline")
        lines.append(" & ".join([""] + [h for _, h in data_keys]) + " \\\\")
        lines.append("\\hline")
        for r in rows:
            cells = [r["label"]] + [_fmt(k, r) for k, _ in data_keys]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
        lines.append("\\hline")
        lines.append("\\end{tabular}")

        tabular = "\n".join(lines)
        if caption:
            out = "\\begin{table}[htbp]\n\\centering\n"
            out += f"\\caption{{{caption}}}\n"
            if label:
                out += f"\\label{{{label}}}\n"
            out += tabular + "\n\\end{table}"
            return out
        return tabular

    def to_html(
        self,
        stars: bool = False,
        star_levels: tuple[float, float, float] = (0.01, 0.05, 0.10),
        float_fmt: str = ".4f",
        pvalue_fmt: str = ".3f",
        caption: str | None = None,
    ) -> str:
        """HTML table representation of the result.

        Parameters
        ----------
        stars : bool, default False
            Append significance stars to estimates.
        star_levels : tuple of float, default (0.01, 0.05, 0.10)
            Thresholds for ***, **, * respectively.
        float_fmt : str, default ".4f"
            Format string for floating-point columns.
        pvalue_fmt : str, default ".3f"
            Format string for p-values.
        caption : str, optional
            Table caption.

        Returns
        -------
        str
        """
        rows = self._summary_rows()
        has_stat = any("statistic" in r for r in rows)

        data_keys = [("estimate", "estimate"), ("std_error", "std err")]
        if has_stat:
            stat_header = rows[0].get("stat_label", "z") if rows else "z"
            p_header = "P>|t|" if stat_header == "t" else "P>|z|"
            data_keys.extend([("statistic", stat_header), ("pvalue", p_header)])
        data_keys.append(("ci", f"[{self.level * 100:.0f}% Conf. Int.]"))

        def _fmt(key, row):
            if key == "ci":
                return f"{row['ci_lower']:{float_fmt}}, {row['ci_upper']:{float_fmt}}"
            if key == "pvalue":
                s = f"{row['pvalue']:{pvalue_fmt}}"
                if s.startswith("-0.000"):
                    s = "0.000"
                return s
            val = row[key]
            if key == "estimate" and stars and "pvalue" in row:
                star = self._star_notation(row["pvalue"], star_levels)
                return f"{val:{float_fmt}}{star}"
            return f"{val:{float_fmt}}"

        lines = []
        if caption:
            lines.append(f"<caption>{caption}</caption>")
        lines.append("<thead>")
        lines.append(
            "<tr>"
            + "".join(f"<th>{h}</th>" for h in ([""] + [h for _, h in data_keys]))
            + "</tr>"
        )
        lines.append("</thead>")
        lines.append("<tbody>")
        for r in rows:
            cells = [r["label"]] + [_fmt(k, r) for k, _ in data_keys]
            lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        lines.append("</tbody>")

        return '<table class="pymargins-result">\n' + "\n".join(lines) + "\n</table>"

    def conf_int(
        self,
        level: float | None = None,
        simultaneous: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recompute CI at a different confidence level.

        For delta-method results, uses the stored gradient and Σ̂ to
        construct a CI at the new level without re-running inference. For
        simulation/bootstrap results, recomputes quantiles of the stored
        draws.

        Parameters
        ----------
        level : float, optional
            New confidence level. If None, returns the stored CI.

        simultaneous : bool, default False
            If True, return a **simultaneous** (family-wise) confidence band
            rather than per-component intervals. For simulation/bootstrap
            results this uses the sup-t method: the critical value is the
            (1−α) quantile of ``maxⱼ |θ_{b,j} − θ̂ⱼ| / seⱼ`` over draws. For
            delta-method results this uses the multivariate-normal
            equicoordinate quantile (Bonferroni-style adjustment via the
            correlation matrix). This is the appropriate interval when
            reporting a vector estimand as a set.

        Returns
        -------
        (lower, upper) : tuple of arrays
        """
        if level is None:
            level = self.level

        if not simultaneous:
            if level == self.level:
                return self.conf_int_lower, self.conf_int_upper

        est_inf = (
            self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
        )

        if self.imputation_diagnostic is not None:
            if simultaneous:
                raise NotImplementedError(
                    "Simultaneous CIs are not yet supported for pooled results."
                )
            se = np.asarray(self.std_error)
            df = np.asarray(self.imputation_diagnostic.df)
            tcrit = stats.t.ppf(0.5 + level / 2.0, df)
            lo_inf = est_inf - tcrit * se
            hi_inf = est_inf + tcrit * se
            if self.phi is not None:
                return np.asarray(self.phi(lo_inf)), np.asarray(self.phi(hi_inf))
            return np.asarray(lo_inf), np.asarray(hi_inf)

        if self.gradient is not None and self.cov_params is not None:
            # Delta method
            if simultaneous:
                # sup-t band from correlation structure of J Σ̂ Jᵀ.
                from .._delta import joint_covariance_of_results

                grad = jnp.asarray(self.gradient)
                if grad.ndim == 1:
                    grad = grad[None, :]
                cov_joint = joint_covariance_of_results(
                    [grad[i] for i in range(grad.shape[0])],
                    jnp.asarray(self.cov_params),
                )
                se_vec = jnp.sqrt(jnp.diag(cov_joint))
                n_comp = int(se_vec.shape[0])

                if n_comp == 1:
                    # Scalar: sup-t equals pointwise z
                    crit = float(stats.norm.ppf(0.5 + level / 2.0))
                else:
                    # Monte-Carlo equicoordinate quantile: draw from N(0,R)
                    # and take the (1−α) quantile of max_j |Z_j|.
                    cov_np = np.asarray(cov_joint)
                    # Force symmetry (numerical noise from JAX→numpy)
                    cov_np = (cov_np + cov_np.T) / 2.0
                    # Regularise for positive definiteness
                    eigvals = np.linalg.eigvalsh(cov_np)
                    min_eig = float(np.min(eigvals))
                    ridge = max(0.0, -min_eig + 1e-6)
                    if ridge > 0:
                        cov_np = cov_np + np.eye(n_comp) * ridge
                    R = cov_np / np.outer(se_vec, se_vec)
                    # Force symmetry again after scaling
                    R = (R + R.T) / 2.0
                    # Clamp to valid correlation bounds
                    R = np.clip(R, -1.0, 1.0)
                    np.fill_diagonal(R, 1.0)
                    rng = np.random.default_rng(42)
                    n_mc = 10000
                    z_draws = rng.multivariate_normal(
                        mean=np.zeros(n_comp), cov=R, size=n_mc
                    )
                    max_abs = np.max(np.abs(z_draws), axis=1)
                    crit = float(np.quantile(max_abs, level))

                lower_inf = est_inf - crit * se_vec
                upper_inf = est_inf + crit * se_vec
            else:
                lower_inf, upper_inf = delta_confint_from_se(
                    est_inf,
                    self.std_error,
                    level=level,
                    phi=None,
                )

            if self.phi is not None:
                return np.asarray(self.phi(lower_inf)), np.asarray(self.phi(upper_inf))
            return np.asarray(lower_inf), np.asarray(upper_inf)

        elif self.draws_inf is not None:
            draws = np.asarray(self.draws_inf)
            est_arr = np.asarray(est_inf)

            if simultaneous:
                # sup-t band: c = (1−α) quantile of max_j |draw_{b,j} − est_j| / se_j
                se = np.asarray(self.std_error)
                if se.ndim == 0:
                    se = np.array([se])
                # Ensure draws and est have compatible shapes
                if draws.ndim == 1:
                    draws = draws[:, None]
                if est_arr.ndim == 0:
                    est_arr = np.array([est_arr])
                # Broadcast est to (n_draws, n_components)
                est_bc = est_arr[None, :] if est_arr.ndim == 1 else est_arr
                # Compute standardized deviations
                with np.errstate(divide="ignore", invalid="ignore"):
                    std_dev = np.abs(draws - est_bc) / se[None, :]
                # max over components per draw
                max_dev = np.nanmax(std_dev, axis=1)
                crit = float(np.quantile(max_dev, level))
                lower_inf = est_arr - crit * se
                upper_inf = est_arr + crit * se
                if self.phi is not None:
                    return (
                        np.asarray(self.phi(lower_inf)),
                        np.asarray(self.phi(upper_inf)),
                    )
                return np.asarray(lower_inf), np.asarray(upper_inf)

            alpha = (1.0 - level) / 2.0

            if self.ci_method == "basic":
                lower_inf = 2 * np.asarray(est_inf) - np.quantile(
                    self.draws_inf, 1.0 - alpha, axis=0
                )
                upper_inf = 2 * np.asarray(est_inf) - np.quantile(
                    self.draws_inf, alpha, axis=0
                )
                if self.phi is not None:
                    return np.asarray(self.phi(lower_inf)), np.asarray(
                        self.phi(upper_inf)
                    )
                return lower_inf, upper_inf

            elif self.ci_method == "bca" and self.bootstrap_extras is not None:
                z0 = self.bootstrap_extras.get("z0")
                a = self.bootstrap_extras.get("a")
                if z0 is not None:
                    from .._inference import _bca_confint

                    lower, upper = _bca_confint(
                        self.draws_inf,
                        est_inf,
                        level,
                        z0,
                        a,
                        self.phi,
                    )
                    return np.asarray(lower), np.asarray(upper)

            elif self.ci_method == "studentized" and self.bootstrap_extras is not None:
                t_stats = self.bootstrap_extras.get("t_star")
                se_hat = self.bootstrap_extras.get("se_hat")
                if t_stats is not None and se_hat is not None:
                    t_lower = np.quantile(t_stats, alpha, axis=0)
                    t_upper = np.quantile(t_stats, 1.0 - alpha, axis=0)
                    lower_inf = np.asarray(est_inf) - t_upper * se_hat
                    upper_inf = np.asarray(est_inf) - t_lower * se_hat
                    if self.phi is not None:
                        return np.asarray(self.phi(lower_inf)), np.asarray(
                            self.phi(upper_inf)
                        )
                    return lower_inf, upper_inf

            # Default to percentile
            lower = np.quantile(self.draws_inf, alpha, axis=0)
            upper = np.quantile(self.draws_inf, 1.0 - alpha, axis=0)
            if self.phi is not None:
                return np.asarray(self.phi(lower)), np.asarray(self.phi(upper))
            return lower, upper

        elif self.draws is not None:
            draws = np.asarray(self.draws)
            est_arr = np.asarray(self.estimate)

            if simultaneous:
                se = np.asarray(self.std_error)
                if se.ndim == 0:
                    se = np.array([se])
                if draws.ndim == 1:
                    draws = draws[:, None]
                if est_arr.ndim == 0:
                    est_arr = np.array([est_arr])
                est_bc = est_arr[None, :] if est_arr.ndim == 1 else est_arr
                with np.errstate(divide="ignore", invalid="ignore"):
                    std_dev = np.abs(draws - est_bc) / se[None, :]
                max_dev = np.nanmax(std_dev, axis=1)
                crit = float(np.quantile(max_dev, level))
                lower = est_arr - crit * se
                upper = est_arr + crit * se
                return np.asarray(lower), np.asarray(upper)

            alpha = (1.0 - level) / 2.0
            lower = np.quantile(draws, alpha, axis=0)
            upper = np.quantile(draws, 1.0 - alpha, axis=0)
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
        if not np.isfinite(value):
            raise ValueError(f"test value must be finite, got {value}")
        if null_scale == "inference" or self.phi_inv is None:
            null_inf = value
        elif null_scale == "reporting":
            null_inf = self.phi_inv(value)
        else:
            raise ValueError(
                f"null_scale must be 'reporting' or 'inference', got {null_scale!r}"
            )

        # Get inference-scale estimate and draws
        est_inf = (
            self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
        )
        draws_inf = (
            self.draws_inf
            if self.draws_inf is not None
            else (
                self.phi_inv(self.draws)
                if self.phi_inv is not None and self.draws is not None
                else self.draws
            )
        )

        if self.imputation_diagnostic is not None:
            # Pooled result: t-test against Rubin df. Same alternative
            # conventions as delta_wald_test, with Student-t in place of normal.
            se = np.asarray(self.std_error)
            df = np.asarray(self.imputation_diagnostic.df)
            t_stat = (np.asarray(est_inf) - null_inf) / se
            if alternative == "two-sided":
                pvalue = 2.0 * stats.t.sf(np.abs(t_stat), df)
            elif alternative == "greater":
                pvalue = stats.t.sf(t_stat, df)
            elif alternative == "less":
                pvalue = stats.t.cdf(t_stat, df)
            else:
                raise ValueError(f"Unknown alternative: {alternative!r}")
            return TestResult(
                statistic=t_stat,
                pvalue=pvalue,
                null_value=value,
                alternative=alternative,
                method="wald",
                estimand_metadata=self.estimand_metadata,
            )

        if self.gradient is not None and self.cov_params is None:
            raise ValueError(
                "Cannot run test: result has a gradient but no frozen cov_params. "
                "This typically happens when the result was materialized without "
                "covariance information."
            )

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
        value: np.ndarray | None = None,
        kind: str = "wald",
        null_scale: Literal["reporting", "inference"] = "reporting",
    ) -> TestResult:
        """Joint test H₀: all estimand components equal (vector-valued) value.

        Tests the joint null using either:

        - ``kind="wald"`` (default): χ² with df = number of components. For
          delta-method results this is the analytical Wald test; for
          simulation/bootstrap draws it is a normal-theory plug-in using the
          empirical covariance of the draws.

        - ``kind="empirical"``: bootstrap-faithful quadratic-form test. The
          distribution of the Mahalanobis statistic
          ``Q_b = (θ_b−θ̂)ᵀ Σ̂_emp⁻¹ (θ_b−θ̂)`` over the draws is used as its own
          reference distribution. The p-value is the proportion of draws with
          ``Q_b ≥ Q_obs``. This avoids the Gaussian approximation that
          ``kind="wald"`` imposes on bootstrap/simulation results and is
          recommended for skewed or heavy-tailed sampling distributions.

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

        kind : {"wald", "empirical"}, default "wald"

        null_scale : {"reporting", "inference"}, default "reporting"
            Which scale ``value`` is supplied on. See ``test`` for details.

        Returns
        -------
        result : TestResult
        """
        if kind not in ("wald", "empirical"):
            raise ValueError(f"kind must be 'wald' or 'empirical', got {kind!r}")

        if value is None:
            # Default null = zero on the inference scale (the universal
            # "no effect" point regardless of phi).
            if self.phi_inv is not None:
                value_inf = self.phi_inv(jnp.zeros_like(jnp.asarray(self.estimate)))
            else:
                value_inf = jnp.zeros_like(jnp.asarray(self.estimate))
        elif null_scale == "inference" or self.phi_inv is None:
            value_inf = jnp.asarray(value)
        elif null_scale == "reporting":
            value_inf = self.phi_inv(jnp.asarray(value))
        else:
            raise ValueError(
                f"null_scale must be 'reporting' or 'inference', got {null_scale!r}"
            )

        estimate = np.asarray(self.estimate)
        value_arr = np.asarray(value_inf)
        if value_arr.shape != estimate.shape:
            raise ValueError(
                f"value shape {value_arr.shape} does not match estimate shape {estimate.shape}"
            )
        if not np.all(np.isfinite(value_arr)):
            raise ValueError("value must be finite (no NaN or Inf)")

        est_inf = (
            self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
        )

        if self.gradient is not None and self.cov_params is not None:
            # Delta-method joint Wald test
            cov = jnp.asarray(self.cov_params)
            chi2, p, df = joint_wald_test(
                jnp.asarray(est_inf),
                jnp.asarray(self.gradient),
                cov,
                null_value=value_inf,
            )
        elif self.draws_inf is not None:
            # Empirical joint test from simulation/bootstrap draws
            draws = np.asarray(self.draws_inf)
            if draws.ndim == 1:
                draws = draws[:, None]
            est_arr = np.asarray(est_inf)
            if est_arr.ndim == 0:
                est_arr = np.array([est_arr])
            diff_arr = est_arr - np.asarray(value_inf)
            # Center draws and compute empirical covariance
            centered = draws - est_arr
            emp_cov = np.cov(centered, rowvar=False)
            if emp_cov.ndim == 0:
                emp_cov = np.array([[emp_cov]])
            # Regularize if singular
            try:
                emp_cov_reg = emp_cov
                solved = np.linalg.solve(emp_cov_reg, diff_arr)
            except np.linalg.LinAlgError:
                ridge = 1e-12 * float(np.trace(emp_cov)) / emp_cov.shape[0]
                ridge = max(ridge, float(np.finfo(emp_cov.dtype).eps))
                emp_cov_reg = emp_cov + ridge * np.eye(emp_cov.shape[0])
                solved = np.linalg.solve(emp_cov_reg, diff_arr)

            if kind == "empirical":
                # Empirical quadratic-form: use the bootstrap distribution of
                # Q_b as its own reference distribution.
                # Q_b = (draw_b - est)^T Sigma^{-1} (draw_b - est)
                # Compute for every draw
                Q = np.sum((centered @ np.linalg.inv(emp_cov_reg)) * centered, axis=1)
                Q_obs = float(diff_arr @ solved)
                p = float(np.mean(Q >= Q_obs))
                chi2 = Q_obs
            else:
                # Wald plug-in: chi2 reference distribution
                chi2 = float(diff_arr @ solved)
                p = float(1.0 - stats.chi2.cdf(chi2, int(diff_arr.shape[0])))
            df = int(diff_arr.shape[0])
        else:
            raise ValueError(
                "Joint test requires either (a) a delta-method result with "
                "gradients and cov_params, or (b) simulation/bootstrap draws. "
                "This result has neither."
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

    def _check_compatible(self, other: MarginsResult) -> None:
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

    def __sub__(self, other: MarginsResult) -> MarginsResult:
        """Difference of two estimands with proper joint inference.

        Computes the delta-method variance of the difference using the joint
        gradient and the shared Σ̂ from the session. Available only when
        both results carry gradients (delta-method results).
        """
        self._check_compatible(other)
        return _combine_results(
            self,
            other,
            lambda a, b: a - b,
            grad_combine=lambda g1, g2: g1 - g2,
            label_combine=lambda l1, l2: f"({l1}) - ({l2})",
        )

    def __add__(self, other: MarginsResult) -> MarginsResult:
        """Add two estimands with proper joint inference via the delta method."""
        self._check_compatible(other)
        return _combine_results(
            self,
            other,
            lambda a, b: a + b,
            grad_combine=lambda g1, g2: g1 + g2,
            label_combine=lambda l1, l2: f"({l1}) + ({l2})",
        )

    def __mul__(self, other) -> MarginsResult:
        """Scale the estimand by a scalar, with inference-aware transforms."""
        if isinstance(other, MarginsResult):
            raise ValueError(
                "Product of two MarginsResults is nonlinear; use evaluate() "
                "with a custom compose function instead."
            )
        try:
            scalar = float(other)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"MarginsResult multiplication requires a scalar, got {type(other).__name__}"
            ) from exc

        # Scale the reported estimate, CI bounds, and inference-scale
        # quantities (SE, gradient, draws) linearly by the scalar.
        new_est = self.estimate * scalar
        if scalar >= 0:
            new_lo = self.conf_int_lower * scalar
            new_hi = self.conf_int_upper * scalar
        else:
            new_lo = self.conf_int_upper * scalar
            new_hi = self.conf_int_lower * scalar
        new_draws = self.draws * scalar if self.draws is not None else None

        return MarginsResult(
            estimate=new_est,
            std_error=self.std_error * abs(scalar),
            conf_int_lower=new_lo,
            conf_int_upper=new_hi,
            method=self.method,
            level=self.level,
            n_obs=self.n_obs,
            kappa=self.kappa,
            delta_sim_disagreement=self.delta_sim_disagreement,
            fallback_triggered=self.fallback_triggered,
            fallback_reason=self.fallback_reason,
            estimand_metadata={
                **self.estimand_metadata,
                "labels": [
                    f"({lbl})*{scalar}"
                    for lbl in self.estimand_metadata.get("labels", [])
                ],
            },
            gradient=(self.gradient * scalar if self.gradient is not None else None),
            draws=new_draws,
            draws_inf=(self.draws_inf * scalar if self.draws_inf is not None else None),
            cov_params=self.cov_params,
            phi=self.phi,
            phi_inv=self.phi_inv,
            session=self.session,
            ci_method=self.ci_method,
            bootstrap_extras=self.bootstrap_extras,
            resample_bank_id=self.resample_bank_id,
            n_boot_effective=self.n_boot_effective,
            n_boot_failed=self.n_boot_failed,
        )

    def __truediv__(self, other) -> MarginsResult:
        """Scale the estimand by the reciprocal of a scalar."""
        if isinstance(other, MarginsResult):
            raise ValueError(
                "Ratio of two MarginsResults is nonlinear; use evaluate() "
                "with a custom compose function (e.g., compose=lambda p: p[0]/p[1]) "
                "for proper inference."
            )
        return self.__mul__(1.0 / float(other))

    # -----------------------------------------------------------------------
    # Cosmetic transformations (don't affect inference)
    # -----------------------------------------------------------------------

    def scaled(self, by: float, units: str | None = None) -> MarginsResult:
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

    def outcome(self, index: int | str) -> MarginsResult:
        """Slice a multi-outcome result to a single outcome.

        Parameters
        ----------
        index : int or str
            Outcome index (0-based) or label.

        Returns
        -------
        MarginsResult
            A new result with only the requested outcome.
        """
        outcome_shape = self.estimand_metadata.get("_outcome_shape")
        labels = self.estimand_metadata.get("labels", [])
        est = np.atleast_1d(self.estimate)

        # G7a: prefer structured metadata over label heuristics
        if outcome_shape is not None and est.ndim == 2:
            n_atoms = outcome_shape["n_atoms"]
            n_outcomes = outcome_shape["n_outcomes"]
            outcome_labels = outcome_shape["outcome_labels"]

            if isinstance(index, str):
                if index not in outcome_labels:
                    raise ValueError(
                        f"Outcome label {index!r} not found. "
                        f"Available: {outcome_labels}"
                    )
                outcome_idx = outcome_labels.index(index)
            else:
                outcome_idx = int(index)
                if not (0 <= outcome_idx < n_outcomes):
                    raise ValueError(
                        f"Outcome index {outcome_idx} out of range "
                        f"(0..{n_outcomes - 1})."
                    )

            # Slice along the outcome axis
            def _slice(arr):
                if arr is None:
                    return None
                a = np.asarray(arr)
                if a.ndim == 2 and a.shape == (n_atoms, n_outcomes):
                    return a[:, outcome_idx]
                elif a.ndim == 3 and a.shape[:2] == (n_atoms, n_outcomes):
                    # gradient: (n_atoms, n_outcomes, n_params)
                    return a[:, outcome_idx, :]
                elif a.ndim == 3 and a.shape[1:] == (n_atoms, n_outcomes):
                    # draws: (n_draws, n_atoms, n_outcomes)
                    return a[:, :, outcome_idx]
                elif a.ndim == 1 and a.size == n_atoms * n_outcomes:
                    # Expanded labels — select every n_outcomes-th
                    mask = np.arange(a.size) % n_outcomes == outcome_idx
                    return a[mask]
                return arr

            new_labels = (
                [labels[i] for i in range(len(labels)) if i % n_outcomes == outcome_idx]
                if labels
                else None
            )
            new_meta = dict(self.estimand_metadata)
            new_meta["labels"] = new_labels
            new_meta["outcome_sliced"] = True
            # Drop _outcome_shape after slicing to avoid confusion
            new_meta.pop("_outcome_shape", None)

            return MarginsResult(
                estimate=_slice(self.estimate),
                std_error=_slice(self.std_error),
                conf_int_lower=_slice(self.conf_int_lower),
                conf_int_upper=_slice(self.conf_int_upper),
                method=self.method,
                level=self.level,
                n_obs=self.n_obs,
                kappa=_slice(self.kappa),
                delta_sim_disagreement=self.delta_sim_disagreement,
                fallback_triggered=self.fallback_triggered,
                fallback_reason=self.fallback_reason,
                estimand_metadata=new_meta,
                gradient=_slice(self.gradient),
                draws=_slice(self.draws),
                draws_inf=_slice(self.draws_inf),
                cov_params=self.cov_params,
                phi=self.phi,
                phi_inv=self.phi_inv,
                session=self.session,
                ci_method=self.ci_method,
                bootstrap_extras=self.bootstrap_extras,
                resample_bank_id=self.resample_bank_id,
                n_boot_effective=self.n_boot_effective,
                n_boot_failed=self.n_boot_failed,
            )

        # Fallback to label-heuristic path for legacy results without _outcome_shape
        n_components = est.size
        n_labels = len(labels)

        if n_labels == 0 or n_labels != n_components:
            raise ValueError(
                "outcome() requires expanded outcome labels; "
                "this result may not be from a multi-outcome model."
            )

        # Each original label was expanded into K suffixed labels.
        # Find which positions correspond to the requested outcome.
        # Heuristic: labels are "lab (suffix)"; group by suffix.
        suffixes = []
        for lab in labels:
            if " (" in lab and lab.endswith(")"):
                suffix = lab[lab.rfind(" (") + 2 : -1]
            else:
                suffix = lab
            suffixes.append(suffix)

        unique_suffixes = []
        seen = set()
        for s in suffixes:
            if s not in seen:
                unique_suffixes.append(s)
                seen.add(s)

        K = len(unique_suffixes)
        if K == 1:
            raise ValueError("outcome() called on a single-outcome result.")

        if isinstance(index, str):
            if index not in unique_suffixes:
                raise ValueError(
                    f"Outcome label {index!r} not found. Available: {unique_suffixes}"
                )
            outcome_idx = unique_suffixes.index(index)
        else:
            outcome_idx = int(index)
            if not (0 <= outcome_idx < K):
                raise ValueError(
                    f"Outcome index {outcome_idx} out of range (0..{K - 1})."
                )

        # Select every K-th entry starting at outcome_idx
        mask = np.arange(n_components) % K == outcome_idx
        if not np.any(mask):
            raise ValueError(f"No components found for outcome {index!r}.")

        def _slice_legacy(arr):
            if arr is None:
                return None
            a = np.asarray(arr)
            if a.ndim == 1:
                return a[mask]
            elif a.ndim == 2:
                if a.shape[0] == n_components:
                    return a[mask]
                elif a.shape[1] == n_components:
                    return a[:, mask]
                else:
                    return a
            elif a.ndim == 3:
                if a.shape[1] == n_components:
                    return a[:, mask]
                elif a.shape[2] == n_components:
                    return a[:, :, mask]
                else:
                    return a
            return a

        new_labels = [labels[i] for i in np.where(mask)[0]]
        new_meta = dict(self.estimand_metadata)
        new_meta["labels"] = new_labels

        return MarginsResult(
            estimate=_slice_legacy(self.estimate),
            std_error=_slice_legacy(self.std_error),
            conf_int_lower=_slice_legacy(self.conf_int_lower),
            conf_int_upper=_slice_legacy(self.conf_int_upper),
            method=self.method,
            level=self.level,
            n_obs=self.n_obs,
            kappa=_slice_legacy(self.kappa),
            delta_sim_disagreement=self.delta_sim_disagreement,
            fallback_triggered=self.fallback_triggered,
            fallback_reason=self.fallback_reason,
            estimand_metadata=new_meta,
            gradient=_slice_legacy(self.gradient),
            draws=_slice_legacy(self.draws),
            draws_inf=_slice_legacy(self.draws_inf),
            cov_params=self.cov_params,
            phi=self.phi,
            phi_inv=self.phi_inv,
            session=self.session,
            ci_method=self.ci_method,
            bootstrap_extras=self.bootstrap_extras,
            resample_bank_id=self.resample_bank_id,
            n_boot_effective=self.n_boot_effective,
            n_boot_failed=self.n_boot_failed,
        )

    def materialize(self) -> MarginsResult:
        """Drop underlying machinery (gradient, draws, session) to reduce
        memory.

        After materialize(), the result is no longer composable with other
        results. Useful for storing many results long-term where you don't
        need to combine them further. Reporting (summary, to_frame, conf_int
        at the stored level) still works.

        Warnings
        --------
        After calling ``materialize()``, ``test()`` and ``conf_int(level=...)``
        will raise ``ValueError`` because ``cov_params`` and ``gradient`` are
        cleared. Only the confidence level stored at construction remains
        available without error.
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
            resample_bank_id=self.resample_bank_id,
        )

    # -----------------------------------------------------------------------
    # Linear hypothesis on an existing vector result
    # -----------------------------------------------------------------------

    def contrast(
        self,
        C: np.ndarray,
        labels: list[str] | None = None,
    ) -> MarginsResult:
        """Apply a contrast matrix to a vector result.

        Parameters
        ----------
        C : ndarray of shape (m, k)
            Contrast matrix. Each row is a linear combination of the
            current result's components.
        labels : list of str, optional
            Labels for the new m contrasts.

        Returns
        -------
        MarginsResult
            A new length-m vector result with proper joint inference.
        """
        if self.gradient is None:
            raise ValueError(
                "contrast() requires a delta-method result. For sim/boot, "
                "apply C to draws_inf manually or use compose_results."
            )
        C = jnp.asarray(C)
        est_inf = (
            self.phi_inv(self.estimate) if self.phi_inv is not None else self.estimate
        )
        new_est_inf = C @ jnp.asarray(est_inf)
        new_grad = C @ jnp.asarray(self.gradient)  # (m, p)
        cov = jnp.asarray(self.cov_params)
        var = jnp.einsum("ij,jk,ik->i", new_grad, cov, new_grad)
        se = np.asarray(jnp.sqrt(var))
        z = stats.norm.ppf(0.5 + self.level / 2.0)
        lo_inf = new_est_inf - z * se
        hi_inf = new_est_inf + z * se
        phi = self.phi
        return MarginsResult(
            estimate=np.asarray(phi(new_est_inf)) if phi else np.asarray(new_est_inf),
            std_error=se,
            conf_int_lower=np.asarray(phi(lo_inf)) if phi else np.asarray(lo_inf),
            conf_int_upper=np.asarray(phi(hi_inf)) if phi else np.asarray(hi_inf),
            method=self.method,
            level=self.level,
            n_obs=self.n_obs,
            kappa=self.kappa,
            estimand_metadata={
                "labels": labels or [f"contrast[{i}]" for i in range(C.shape[0])]
            },
            gradient=np.asarray(new_grad),
            cov_params=self.cov_params,
            phi=self.phi,
            phi_inv=self.phi_inv,
            session=self.session,
        )

    def pairwise_contrasts(
        self,
        labels: list[str] | None = None,
    ) -> MarginsResult:
        """All pairwise differences between components of a vector result.

        Parameters
        ----------
        labels : list of str, optional
            Labels for the original k components. If omitted, uses
            ``estimand_metadata["labels"]`` or generic ``[0], [1], ...``.

        Returns
        -------
        MarginsResult
            A vector result of length ``k*(k-1)/2`` with joint inference.
            Labels are formatted as ``"j - i"``.
        """
        if self.gradient is None:
            raise ValueError("pairwise_contrasts() requires a delta-method result.")
        est = np.atleast_1d(self.estimate)
        k = int(est.size)
        if k < 2:
            raise ValueError("pairwise_contrasts() requires at least 2 components")
        if labels is None:
            labels = self.estimand_metadata.get("labels")
        if labels is None or len(labels) != k:
            labels = [f"[{i}]" for i in range(k)]
        from ..scenarios import diff_matrix

        C = diff_matrix(k, kind="pairwise")
        new_labels = []
        row = 0
        for i in range(k):
            for j in range(i + 1, k):
                new_labels.append(f"{labels[j]} - {labels[i]}")
                row += 1
        return self.contrast(C, labels=new_labels)

    # -----------------------------------------------------------------------
    # Per-observation influence
    # -----------------------------------------------------------------------

    def influence(self) -> np.ndarray:
        """Per-observation influence on the estimand (DFBETA sign).

        Returns, for each training observation ``i``, the approximate amount
        observation ``i`` contributes to the estimate — i.e. ``θ̂ − θ_{(-i)}``,
        the leave-one-out deletion influence. Positive values mean dropping
        the observation would pull the estimate down.

        Two routes, returning the same first-order quantity on the
        **inference scale**:

        * **Delta-method** results use the analytical empirical influence
          function ``∇h(β̂)ᵀ Σ̂ score_i(β̂)``, where ``score_i`` is the
          per-observation score from ``adapter.score_obs()`` and ``Σ̂`` the
          frozen covariance. ``Σ̂ score_i`` is the one-step (infinitesimal
          jackknife) approximation to ``β̂ − β̂_{(-i)}``.
        * **BCa bootstrap** results reuse the exact leave-one-out
          refits cached during acceleration: ``θ̂_inf − θ_{(-i)}``.

        Returns
        -------
        ndarray
            Shape ``(n_obs,)`` for a scalar estimand, ``(n_obs, k)`` for a
            length-``k`` vector estimand.

        Notes
        -----
        Summing the squared influence reconstructs the delta-method variance:
        ``Σ_i (influence_i)² ≈ ∇hᵀ Σ̂ ∇h`` under a correctly specified model
        (default ``Σ̂``). With a robust/cluster ``vcov`` the value is the
        corresponding sandwich influence.
        """
        if self.ci_method == "bca" and self.bootstrap_extras:
            jack = self.bootstrap_extras.get("influence_jackknife")
            if jack is not None:
                theta_minus = np.asarray(jack)  # (n,) or (n, k), inference scale
                est_inf = (
                    np.asarray(self.phi_inv(self.estimate))
                    if self.phi_inv is not None
                    else np.asarray(self.estimate)
                )
                return est_inf - theta_minus
        if self.gradient is not None and self.cov_params is not None:
            sess = self._session_obj()
            adapter = getattr(sess, "adapter", None) if sess is not None else None
            if adapter is not None and hasattr(adapter, "score_obs"):
                S = np.asarray(adapter.score_obs())  # (n, p)
                cov = np.asarray(self.cov_params)  # (p, p)
                g = np.asarray(self.gradient)  # (p,) or (k, p)
                beta_infl = S @ cov  # (n, p) ≈ β̂ − β̂_(−i)
                if g.ndim == 1:
                    return beta_infl @ g  # (n,)
                return beta_infl @ g.T  # (n, k)
        raise NotImplementedError(
            "Influence requires either a BCa bootstrap result or a "
            "delta-method result whose adapter exposes score_obs(). The "
            "statsmodels GLM, OLS, Logit/Probit, and Poisson/NegBin adapters "
            "support score_obs(); others can add a one-line wrapper."
        )

    # -----------------------------------------------------------------------
    # Disk persistence helpers
    # -----------------------------------------------------------------------

    def to_disk(self, path: str | Path, *, format: str = "pickle") -> None:
        """Persist a materialized result to disk. Auto-materializes."""
        if format != "pickle":
            raise ValueError(
                f"Unsupported format: {format!r}. Only 'pickle' is supported."
            )
        obj = (
            self.materialize()
            if (
                self.gradient is not None
                or self.draws is not None
                or self.session is not None
            )
            else self
        )
        from .. import __version__

        phi_name = _phi_to_name(obj.phi)
        phi_inv_name = _phi_to_name(obj.phi_inv)
        if phi_name is None and obj.phi is not None:
            warnings.warn(
                "MarginsResult.phi is a custom function and cannot be serialized; "
                "it will be set to None on load.",
                UserWarning,
                stacklevel=2,
            )
        if phi_inv_name is None and obj.phi_inv is not None:
            warnings.warn(
                "MarginsResult.phi_inv is a custom function and cannot be serialized; "
                "it will be set to None on load.",
                UserWarning,
                stacklevel=2,
            )
        obj = dataclasses.replace(obj, session=None, phi=None, phi_inv=None)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "version": __version__,
                    "result": obj,
                    "phi_name": phi_name,
                    "phi_inv_name": phi_inv_name,
                },
                f,
            )

    @classmethod
    def from_disk(cls, path: str | Path) -> MarginsResult:
        """Load a MarginsResult from disk."""
        with open(path, "rb") as f:
            blob = pickle.load(f)
        from .. import __version__

        if blob.get("version") != __version__:
            warnings.warn(
                f"Result was saved with pymargins {blob.get('version')}; "
                f"current version is {__version__}. Schema may differ.",
                UserWarning,
                stacklevel=2,
            )
        result = blob["result"]
        phi = _name_to_phi(blob.get("phi_name"))
        phi_inv = _name_to_phi(blob.get("phi_inv_name"))
        return dataclasses.replace(result, phi=phi, phi_inv=phi_inv)


# ---------------------------------------------------------------------------
# Internal: phi serialization helpers for disk persistence
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Internal: result combination helper
# ---------------------------------------------------------------------------


def _join_fallback_reasons(a_reason, b_reason):
    """Combine fallback reasons from two results."""
    if not a_reason and not b_reason:
        return None
    if a_reason and not b_reason:
        return a_reason
    if b_reason and not a_reason:
        return b_reason
    return f"{a_reason}; {b_reason}"


def _conservative_kappa(a_kappa, b_kappa):
    """Propagate κ through composition using a conservative bound.

    The combined estimand h = f(h_A, h_B) may have a different Lipschitz
    constant than either component. Without access to the outer function f
    we take the maximum of the component κ values as a safe upper bound.
    """
    if a_kappa is None and b_kappa is None:
        return None
    if a_kappa is None:
        return b_kappa
    if b_kappa is None:
        return a_kappa
    # Element-wise maximum for array κ, scalar max for scalar κ
    a_arr = np.asarray(a_kappa)
    b_arr = np.asarray(b_kappa)
    result = np.maximum(a_arr, b_arr)
    # Preserve scalar when both inputs were scalar
    if a_arr.ndim == 0 and b_arr.ndim == 0:
        return float(result)
    return result


def _check_draws_match(a: MarginsResult, b: MarginsResult) -> None:
    """Verify that two simulation/bootstrap results carry matched draws.

    Simulation draws are matched when they share the same session, rng_seed,
    n_sim, and Σ̂ (same ``cov_params`` identity). Bootstrap draws are matched
    when they share the same session-level resample bank (checked via
    ``resample_bank_id``).
    """
    # Bootstrap composition: matched resample bank
    if a.method == "bootstrap" or b.method == "bootstrap":
        if a.resample_bank_id is None or b.resample_bank_id is None:
            raise ValueError(
                "Bootstrap composition requires both results to carry a "
                "resample_bank_id. Results produced before the session bank "
                "was initialized are not composable."
            )
        if a.resample_bank_id != b.resample_bank_id:
            raise ValueError(
                "Bootstrap composition requires both results to share the same "
                "session-level resample bank. The resample indices differ, "
                "so the draws are not jointly distributed."
            )
        # Fall through to Σ̂ check below

    # Same session already checked by _check_compatible
    a_sess = a._session_obj()
    b_sess = b._session_obj()
    if a_sess is not None and b_sess is not None:
        if getattr(a_sess, "rng_seed", None) != getattr(b_sess, "rng_seed", None):
            raise ValueError(
                "Simulation composition requires both results to use the same "
                "rng_seed. The draws were generated from different random streams."
            )
        if getattr(a_sess, "n_sim", None) != getattr(b_sess, "n_sim", None):
            raise ValueError(
                "Simulation composition requires both results to use the same "
                "n_sim. The draws have different lengths."
            )

    # Σ̂ identity — same covariance matrix object or equal values
    if a.cov_params is not None and b.cov_params is not None:
        if not np.allclose(np.asarray(a.cov_params), np.asarray(b.cov_params)):
            raise ValueError(
                "Simulation composition requires both results to share the same "
                "Σ̂ (cov_params). The covariance matrices differ."
            )


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

    if a.gradient is not None and b.gradient is not None:
        # Delta-method composition path (G2 vector-aware)
        new_grad = grad_combine(a.gradient, b.gradient)

        if a.cov_params is None:
            raise ValueError(
                "Composition requires Σ̂ on the result (cov_params). The "
                "originating session should have populated it; if this result "
                "was constructed manually, supply cov_params."
            )
        cov = jnp.asarray(a.cov_params)
        g = jnp.asarray(new_grad)

        # Compute SE: scalar or vector
        if g.ndim == 1:
            var = jnp.dot(g, cov @ g)
            se = float(jnp.sqrt(var))
        else:
            # g has shape (n_components, n_params)
            # var[i] = g[i] @ cov @ g[i]
            var = jnp.einsum("ij,jk,ik->i", g, cov, g)
            se = np.asarray(jnp.sqrt(var))

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

        a_label = (
            a.estimand_metadata.get("labels", [""])[0]
            if a.estimand_metadata.get("labels")
            else "A"
        )
        b_label = (
            b.estimand_metadata.get("labels", [""])[0]
            if b.estimand_metadata.get("labels")
            else "B"
        )

        # G4: conservative κ propagation
        kappa = _conservative_kappa(a.kappa, b.kappa)

        return MarginsResult(
            estimate=np.asarray(estimate_report),
            std_error=np.asarray(se),
            conf_int_lower=np.asarray(lower_report),
            conf_int_upper=np.asarray(upper_report),
            method=a.method,
            level=a.level,
            n_obs=max(a.n_obs, b.n_obs),
            kappa=kappa,
            delta_sim_disagreement=None,
            fallback_triggered=a.fallback_triggered or b.fallback_triggered,
            fallback_reason=_join_fallback_reasons(
                a.fallback_reason, b.fallback_reason
            ),
            estimand_metadata={"labels": [label_combine(a_label, b_label)]},
            gradient=new_grad,
            draws=None,
            draws_inf=None,
            cov_params=a.cov_params,
            phi=a.phi,
            phi_inv=a.phi_inv,
            session=a.session,
            ci_method=None,
            bootstrap_extras=None,
            resample_bank_id=None,
            n_boot_effective=None,
            n_boot_failed=None,
        )

    elif a.draws_inf is not None and b.draws_inf is not None:
        # Simulation/bootstrap draws composition path (G1)
        _check_draws_match(a, b)

        a_draws = np.asarray(a.draws_inf)
        b_draws = np.asarray(b.draws_inf)
        # Ensure both are at least 2-D for stacking
        if a_draws.ndim == 1:
            a_draws = a_draws[:, None]
        if b_draws.ndim == 1:
            b_draws = b_draws[:, None]

        # Combine elementwise on the inference scale
        combined_draws_inf = estimate_combine(a_draws, b_draws)

        # SE = std dev of combined draws
        se_arr = np.std(combined_draws_inf, axis=0, ddof=1)
        se = float(se_arr) if se_arr.ndim == 0 else np.asarray(se_arr)

        # CI via percentile of combined draws
        alpha = (1.0 - a.level) / 2.0
        lo_inf = np.quantile(combined_draws_inf, alpha, axis=0)
        hi_inf = np.quantile(combined_draws_inf, 1.0 - alpha, axis=0)

        # If the original results had scalar estimates, squeeze back to scalar
        a_est = np.asarray(a.estimate)
        if a_est.ndim == 0:
            lo_inf = float(lo_inf.flat[0]) if np.ndim(lo_inf) > 0 else float(lo_inf)
            hi_inf = float(hi_inf.flat[0]) if np.ndim(hi_inf) > 0 else float(hi_inf)
            se = float(se.flat[0]) if np.ndim(se) > 0 else float(se)
            combined_draws_inf = np.asarray(combined_draws_inf).ravel()

        if a.phi is not None:
            estimate_report = np.asarray(a.phi(combined_inf))
            lower_report = np.asarray(a.phi(lo_inf))
            upper_report = np.asarray(a.phi(hi_inf))
            combined_draws = np.asarray(a.phi(combined_draws_inf))
        else:
            estimate_report = combined_inf
            lower_report = lo_inf
            upper_report = hi_inf
            combined_draws = combined_draws_inf

        a_label = (
            a.estimand_metadata.get("labels", [""])[0]
            if a.estimand_metadata.get("labels")
            else "A"
        )
        b_label = (
            b.estimand_metadata.get("labels", [""])[0]
            if b.estimand_metadata.get("labels")
            else "B"
        )

        kappa = _conservative_kappa(a.kappa, b.kappa)

        return MarginsResult(
            estimate=np.asarray(estimate_report),
            std_error=np.asarray(se),
            conf_int_lower=np.asarray(lower_report),
            conf_int_upper=np.asarray(upper_report),
            method=a.method,
            level=a.level,
            n_obs=max(a.n_obs, b.n_obs),
            kappa=kappa,
            delta_sim_disagreement=None,
            fallback_triggered=a.fallback_triggered or b.fallback_triggered,
            fallback_reason=_join_fallback_reasons(
                a.fallback_reason, b.fallback_reason
            ),
            estimand_metadata={"labels": [label_combine(a_label, b_label)]},
            gradient=None,
            draws=combined_draws,
            draws_inf=combined_draws_inf,
            cov_params=a.cov_params,
            phi=a.phi,
            phi_inv=a.phi_inv,
            session=a.session,
            ci_method="percentile (composed)",
            bootstrap_extras=None,
            resample_bank_id=None,
        )

    else:
        raise ValueError(
            "Composition requires either (a) delta-method results with "
            "gradients and cov_params, or (b) simulation/bootstrap results "
            "with matched draws. One result has gradients, the other has "
            "draws — mixing methods is not supported."
        )


# ---------------------------------------------------------------------------
# Nonlinear multi-result composition (G3)
# ---------------------------------------------------------------------------


def compose_results(
    results: list[MarginsResult],
    fn: Callable,
    label: str | None = None,
) -> MarginsResult:
    """Compose multiple results nonlinearly via an explicit function.

    This is the supported way to build ratios, products, and other nonlinear
    combinations of estimands from separate calls (e.g. an AME and a
    prediction).  The arithmetic operators ``+`` and ``-`` only handle linear
    combinations; for anything else, use ``compose_results``.

    Parameters
    ----------
    results : list of MarginsResult
        Results to compose.  All must come from the same ``Margins`` session
        and use the same inference method (delta, simulation, or bootstrap).

    fn : callable
        A function ``fn(theta) -> y`` where ``theta`` is a 1-D array of
        inference-scale estimates (one per input result) and ``y`` is the
        composed estimand (scalar or vector).  For the delta path, ``fn``
        must be JAX-differentiable.

    label : str, optional
        Label for the composed estimand.  Defaults to ``"composed"``.

    Returns
    -------
    MarginsResult
        A new result carrying the composed estimate, SE, CI, and (for delta)
        gradient or (for simulation/bootstrap) draws.

    Raises
    ------
    ValueError
        If results are from different sessions, use different methods, or
        lack the required machinery (gradients for delta, matched draws for
        simulation/bootstrap).
    """
    if len(results) < 2:
        raise ValueError("compose_results requires at least two results.")

    # Validate same session
    sessions = [r._session_obj() for r in results]
    if any(s is None for s in sessions):
        raise ValueError("All results must carry a session reference for composition.")
    first_sess = sessions[0]
    if not all(s is first_sess for s in sessions[1:]):
        raise ValueError(
            "compose_results requires all results to come from the same "
            "Margins session."
        )

    # Validate same method
    methods = [r.method for r in results]
    if not all(m == methods[0] for m in methods):
        raise ValueError(
            "compose_results requires all results to use the same inference "
            f"method. Got: {methods}"
        )

    method = methods[0]
    level = results[0].level
    phi = results[0].phi
    phi_inv = results[0].phi_inv
    cov_params = results[0].cov_params
    n_obs = max(r.n_obs for r in results)

    # Inference-scale estimates
    theta_inf = jnp.stack(
        [
            jnp.asarray(phi_inv(r.estimate) if phi_inv is not None else r.estimate)
            for r in results
        ]
    )

    # Evaluate fn at the point estimate
    composed_inf = jnp.asarray(fn(theta_inf))

    if method == "delta":
        # Delta path: chain rule through fn
        for r in results:
            if r.gradient is None:
                raise ValueError(
                    "Delta composition requires all results to carry gradients."
                )
        if cov_params is None:
            raise ValueError(
                "Delta composition requires Σ̂ (cov_params) on the results."
            )

        # Jacobian of fn w.r.t. theta
        jac_fn = jax.jacfwd(fn)
        is_vector = theta_inf.ndim == 2
        if is_vector:
            composed_inf = jnp.asarray(jax.vmap(fn, in_axes=1)(theta_inf))
            J = jnp.asarray(jax.vmap(jac_fn, in_axes=1)(theta_inf))  # (k, n_results)
        else:
            composed_inf = jnp.asarray(fn(theta_inf))
            J = jnp.asarray(jac_fn(theta_inf))

        # Component gradients w.r.t. beta
        grads = [jnp.asarray(r.gradient) for r in results]

        # combined_grad = J @ stacked_grads
        # Handle scalar and vector fn output
        if J.ndim == 0:
            # fn returns scalar, J is scalar (unlikely but handle)
            J = J.reshape(1, 1)
        elif J.ndim == 1 and composed_inf.ndim == 0:
            # fn returns scalar, J is (n_results,)
            J = J[None, :]

        if all(g.ndim == 1 for g in grads):
            # All scalar components
            stacked = jnp.stack(grads, axis=0)  # (n_results, n_params)
            combined_grad = J @ stacked  # (n_out, n_params)
        elif all(g.ndim == 2 for g in grads):
            if not is_vector:
                raise NotImplementedError(
                    "Vector gradients with scalar theta_inf not supported. "
                    "For non-elementwise composition of vector results, use evaluate()."
                )
            stacked = jnp.stack(grads, axis=0)  # (n_results, k, p)
            if J.ndim != 2:
                raise NotImplementedError(
                    f"Expected J.ndim==2 for vector composition, got {J.ndim}"
                )
            combined_grad = jnp.einsum("ij,ijk->jk", J, stacked)  # (k, p)
        else:
            raise NotImplementedError(
                "Mixed scalar/vector gradients not supported in compose_results."
            )

        # Squeeze scalar-output shapes to 0-d for consistency with ordinary
        # scalar results.  Only squeeze when fn genuinely returns a scalar,
        # not a length-1 vector.
        if combined_grad.shape[0] == 1 and composed_inf.ndim == 0:
            combined_grad = combined_grad[0]
            composed_inf = (
                composed_inf.item()
                if hasattr(composed_inf, "item")
                else float(composed_inf)
            )

        # SE from delta method
        cov = jnp.asarray(cov_params)
        if combined_grad.ndim == 1:
            var = float(jnp.dot(combined_grad, cov @ combined_grad))
            se = float(jnp.sqrt(var))
        else:
            var = jnp.einsum("ij,jk,ik->i", combined_grad, cov, combined_grad)
            se = np.asarray(jnp.sqrt(var))

        z = stats.norm.ppf(0.5 + level / 2.0)
        lo_inf = composed_inf - z * se
        hi_inf = composed_inf + z * se

        if phi is not None:
            estimate_report = np.asarray(phi(composed_inf))
            lower_report = np.asarray(phi(lo_inf))
            upper_report = np.asarray(phi(hi_inf))
        else:
            estimate_report = composed_inf
            lower_report = lo_inf
            upper_report = hi_inf

        # Reduce κ over all results
        from functools import reduce

        kappa = reduce(_conservative_kappa, [r.kappa for r in results])

        return MarginsResult(
            estimate=np.asarray(estimate_report),
            std_error=np.asarray(se),
            conf_int_lower=np.asarray(lower_report),
            conf_int_upper=np.asarray(upper_report),
            method=method,
            level=level,
            n_obs=n_obs,
            kappa=kappa,
            delta_sim_disagreement=None,
            fallback_triggered=any(r.fallback_triggered for r in results),
            fallback_reason="; ".join(
                r.fallback_reason for r in results if r.fallback_reason
            )
            or None,
            estimand_metadata={"labels": [label or "composed"]},
            gradient=combined_grad,
            draws=None,
            draws_inf=None,
            cov_params=cov_params,
            phi=phi,
            phi_inv=phi_inv,
            session=results[0].session,
            ci_method=None,
            bootstrap_extras=None,
            resample_bank_id=None,
        )

    else:  # simulation or bootstrap
        # Draws path: apply fn elementwise to matched draws
        for r in results:
            if r.draws_inf is None:
                raise ValueError(
                    f"{method} composition requires all results to carry draws."
                )

        # Validate matched draws
        for i in range(1, len(results)):
            _check_draws_match(results[0], results[i])

        # Warn if bootstrap results used BCa/studentized — composed
        # percentile CIs are not valid for those methods without
        # recomputing z0/a/t* on the derived quantity.
        ci_methods = {r.ci_method for r in results if r.ci_method}
        invalid_for_compose = {"bca", "studentized"}
        bad = ci_methods & invalid_for_compose
        if bad:
            warnings.warn(
                f"Bootstrap composition uses percentile CIs, but input results "
                f"carry {sorted(bad)}.  BCa/studentized intervals are not "
                f"generally valid on a derived quantity without recomputing "
                f"acceleration/studentization on the composition.  Use "
                f"evaluate() if you need those CI methods.",
                UserWarning,
                stacklevel=2,
            )

        draws_list = [np.asarray(r.draws_inf) for r in results]
        # Ensure all are 2-D (n_draws, n_components)
        for i, d in enumerate(draws_list):
            if d.ndim == 1:
                draws_list[i] = d[:, None]

        n_draws = draws_list[0].shape[0]

        # Stack along axis 1: (n_draws, n_results, n_components_per_result)
        # For scalar results, each is (n_draws, 1)
        stacked = np.stack(draws_list, axis=1)  # (n_draws, n_results, ...)

        # Apply fn to each draw
        is_vector = theta_inf.ndim == 2
        composed_draws_inf = []
        if is_vector:
            for b in range(n_draws):
                theta_b = jnp.asarray(stacked[b, :, :])  # (n_results, k)
                val = jax.vmap(fn, in_axes=1)(theta_b)  # (k,)
                composed_draws_inf.append(np.asarray(val))
        else:
            for b in range(n_draws):
                theta_b = jnp.asarray(stacked[b, :, 0])  # (n_results,)
                val = fn(theta_b)
                composed_draws_inf.append(np.asarray(val))
        composed_draws_inf = np.stack(composed_draws_inf, axis=0)
        # composed_draws_inf shape: (n_draws,) for scalar fn, (n_draws, m) for vector fn

        if composed_draws_inf.ndim == 1:
            se = float(np.std(composed_draws_inf, ddof=1))
            alpha = (1.0 - level) / 2.0
            lo_inf = float(np.quantile(composed_draws_inf, alpha))
            hi_inf = float(np.quantile(composed_draws_inf, 1.0 - alpha))
        else:
            se_arr = np.std(composed_draws_inf, axis=0, ddof=1)
            se = float(se_arr) if se_arr.ndim == 0 else np.asarray(se_arr)
            alpha = (1.0 - level) / 2.0
            lo_inf = np.quantile(composed_draws_inf, alpha, axis=0)
            hi_inf = np.quantile(composed_draws_inf, 1.0 - alpha, axis=0)

        if phi is not None:
            estimate_report = np.asarray(phi(composed_inf))
            lower_report = np.asarray(phi(lo_inf))
            upper_report = np.asarray(phi(hi_inf))
            composed_draws = np.asarray(phi(composed_draws_inf))
        else:
            estimate_report = np.asarray(composed_inf)
            lower_report = np.asarray(lo_inf)
            upper_report = np.asarray(hi_inf)
            composed_draws = composed_draws_inf

        from functools import reduce

        kappa = reduce(_conservative_kappa, [r.kappa for r in results])

        return MarginsResult(
            estimate=np.asarray(estimate_report),
            std_error=np.asarray(se),
            conf_int_lower=np.asarray(lower_report),
            conf_int_upper=np.asarray(upper_report),
            method=method,
            level=level,
            n_obs=n_obs,
            kappa=kappa,
            delta_sim_disagreement=None,
            fallback_triggered=any(r.fallback_triggered for r in results),
            fallback_reason="; ".join(
                r.fallback_reason for r in results if r.fallback_reason
            )
            or None,
            estimand_metadata={"labels": [label or "composed"]},
            gradient=None,
            draws=np.asarray(composed_draws),
            draws_inf=np.asarray(composed_draws_inf),
            cov_params=cov_params,
            phi=phi,
            phi_inv=phi_inv,
            session=results[0].session,
            ci_method="percentile (composed)",
            bootstrap_extras=None,
            resample_bank_id=results[0].resample_bank_id,
            n_boot_effective=None,
            n_boot_failed=None,
        )


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
