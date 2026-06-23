"""GraphResult — self-contained doctrine result object.

Implements the result contract from design §7.1 and req. §6.
Added in 0.4.0 (R4).
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from .._delta import delta_se
from ..scenarios import diff_matrix
from ._intervals import (
    bonferroni_level,
    delta_interval,
    draws_interval,
    draws_test,
    joint_wald,
    sidak_level,
    supt_interval_delta,
    supt_interval_draws,
    wald_test,
)
from ._scales import _name_to_phi, _phi_to_name
from ._test import TestResult

LEVEL_LOCKED_MSG = (
    "conf_int() takes no level=. The confidence level is declared at the "
    "estimator constructor (level=<x> in this plan) and is part of the "
    "pre-registered analysis. To report at a different level, declare a new "
    "estimator (the recompute is cheap; the new plan hash is the point). "
    'Family corrections — conf_int(correction="bonferroni"|"sidak"|"sup-t") '
    "— allocate the declared budget and only widen."
)


@dataclass(eq=False)
class GraphResult:
    """Self-contained result object. Req §6, design §7.1. Added in 0.4.0 (R4).

    GraphResult stores everything needed for reporting, hypothesis tests,
    simultaneous intervals, disk round-trips, and downstream composability.
    It holds no reference to the originating estimator or session.
    """

    # Core reporting --------------------------------------------------------
    estimate: np.ndarray            # reporting scale
    std_error: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    labels: list[str] | None
    method: str                     # resolved inference method
    level: float                    # declared confidence level (locked)
    ci: str
    scale: str
    at: str
    plan: Any                       # immutable Plan/copy
    population_note: str | None
    n_obs: int
    estimand_metadata: dict

    # Diagnostics -----------------------------------------------------------
    kappa: np.ndarray | float | None = None
    delta_sim_disagreement: float | None = None
    n_boot_effective: int | None = None
    n_boot_failed: int | None = None
    imputation_diagnostic: Any | None = None

    # Per-method payload ----------------------------------------------------
    gradient: np.ndarray | None = None          # delta (inference scale)
    cov_params: np.ndarray | None = None        # delta (frozen Σ̂)
    draws: np.ndarray | None = None             # sim/boot (reporting scale)
    draws_inf: np.ndarray | None = None         # sim/boot (inference scale)
    psi_h: np.ndarray | None = None             # tier-1 influence ψ^h
    ci_method: str | None = None
    bootstrap_extras: dict | None = None
    phi: Callable | None = None                 # reporting-scale transform
    phi_inv: Callable | None = None             # inverse transform

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_engine(
        cls,
        result_data: dict,
        *,
        plan: Any,
        labels: list[str] | None = None,
        population_note: str | None = None,
        n_obs: int = 0,
        psi_h: np.ndarray | None = None,
        phi: Callable | None = None,
        phi_inv: Callable | None = None,
    ) -> GraphResult:
        """Build a GraphResult from the executor's result dict (G1.3)."""
        meta = dict(result_data.get("estimand_metadata", {}))
        if labels is not None:
            meta["labels"] = labels

        return cls(
            estimate=np.asarray(result_data["estimate"]),
            std_error=np.asarray(result_data["std_error"]),
            conf_int_lower=np.asarray(result_data["conf_int_lower"]),
            conf_int_upper=np.asarray(result_data["conf_int_upper"]),
            labels=labels if labels is not None else meta.get("labels"),
            method=result_data["method"],
            level=float(result_data.get("level", plan.level)),
            ci=plan.ci if plan.ci is not None else cls._default_ci_method(result_data["method"]),
            scale=plan.scale,
            at=getattr(plan, "at", "overall"),
            plan=plan,
            population_note=population_note,
            n_obs=int(n_obs),
            estimand_metadata=meta,
            kappa=result_data.get("kappa"),
            delta_sim_disagreement=result_data.get("delta_sim_disagreement"),
            n_boot_effective=result_data.get("n_boot_effective"),
            n_boot_failed=result_data.get("n_boot_failed"),
            gradient=result_data.get("gradient"),
            cov_params=result_data.get("cov_params"),
            draws=result_data.get("draws"),
            draws_inf=result_data.get("draws_inf"),
            psi_h=psi_h,
            ci_method=result_data.get("ci_method"),
            bootstrap_extras=result_data.get("bootstrap_extras"),
            phi=phi,
            phi_inv=phi_inv,
        )

    @staticmethod
    def _default_ci_method(method: str) -> str:
        return "percentile" if method == "bootstrap" else "wald"

    # ------------------------------------------------------------------
    # Scale helpers
    # ------------------------------------------------------------------

    def _to_inference(self, value_reporting: np.ndarray | float) -> np.ndarray:
        if self.phi_inv is None:
            return np.asarray(value_reporting)
        return np.asarray(self.phi_inv(np.asarray(value_reporting)))

    def _to_reporting(self, value_inference: np.ndarray) -> np.ndarray:
        if self.phi is None:
            return np.asarray(value_inference)
        return np.asarray(self.phi(np.asarray(value_inference)))

    def _inference_estimate(self) -> np.ndarray:
        return self._to_inference(self.estimate)

    def _inference_draws(self) -> np.ndarray | None:
        if self.draws_inf is not None:
            return self.draws_inf
        if self.draws is not None and self.phi_inv is not None:
            return np.asarray(self.phi_inv(self.draws))
        return self.draws

    # ------------------------------------------------------------------
    # Intervals
    # ------------------------------------------------------------------

    def conf_int(self, correction: str | None = None, **dead):
        """Confidence intervals with optional family correction.

        The confidence ``level`` is locked at construction; pass
        ``correction="bonferroni"|"sidak"|"sup-t"`` to allocate the declared
        level across multiple comparisons.
        """
        if "level" in dead:
            raise TypeError(LEVEL_LOCKED_MSG)
        if dead:
            raise TypeError(f"conf_int() got unexpected keyword argument {next(iter(dead))!r}")

        if correction is None:
            return self.conf_int_lower, self.conf_int_upper

        est = np.asarray(self.estimate)
        k = int(est.size if est.ndim > 0 else 1)

        if correction == "bonferroni":
            adj_level = bonferroni_level(self.level, k)
        elif correction == "sidak":
            adj_level = sidak_level(self.level, k)
        elif correction == "sup-t":
            return self._supt_interval()
        else:
            raise ValueError(
                f"correction={correction!r} is not supported. "
                f"Supported: None, 'bonferroni', 'sidak', 'sup-t'."
            )

        return self._pointwise_interval_at_level(adj_level)

    def _pointwise_interval_at_level(
        self, level: float
    ) -> tuple[np.ndarray, np.ndarray]:
        est_inf = self._inference_estimate()
        if self.gradient is not None and self.cov_params is not None:
            return delta_interval(
                est_inf, self.gradient, self.cov_params, level, phi=self.phi
            )
        draws_inf = self._inference_draws()
        if draws_inf is not None:
            return draws_interval(
                draws_inf,
                level,
                phi=self.phi,
                ci_method=self.ci_method or "percentile",
                bootstrap_extras=self.bootstrap_extras,
            )
        raise ValueError(
            "Cannot recompute CI: result has neither gradient nor draws."
        )

    def _supt_interval(self) -> tuple[np.ndarray, np.ndarray]:
        est_inf = self._inference_estimate()
        se = np.asarray(self.std_error)
        if self.gradient is not None and self.cov_params is not None:
            return supt_interval_delta(
                est_inf, self.gradient, self.cov_params, self.level, phi=self.phi
            )
        draws_inf = self._inference_draws()
        if draws_inf is not None:
            return supt_interval_draws(
                draws_inf, est_inf, se, self.level, phi=self.phi
            )
        raise ValueError(
            "Cannot compute sup-t interval: result has neither gradient nor draws."
        )

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test(self, value: float = 0.0, *, null_scale: str = "reporting") -> TestResult:
        """Per-component test H₀: estimand = value."""
        if not np.isfinite(value):
            raise ValueError(f"test value must be finite, got {value}")
        null_inf = self._null_to_inference(value, null_scale)
        est_inf = self._inference_estimate()

        if self.gradient is not None and self.cov_params is not None:
            statistic, pvalue = wald_test(
                est_inf,
                self.gradient,
                self.cov_params,
                null_value=null_inf,
                alternative="two-sided",
            )
        else:
            draws_inf = self._inference_draws()
            if draws_inf is None:
                raise ValueError(
                    "Cannot run test: result has neither gradient nor draws."
                )
            statistic, pvalue = draws_test(
                est_inf, draws_inf, null_value=null_inf, alternative="two-sided"
            )

        return TestResult(
            statistic=np.asarray(statistic),
            pvalue=np.asarray(pvalue),
            null_value=value,
            alternative="two-sided",
            method="wald",
            estimand_metadata=self.estimand_metadata,
        )

    def joint_test(
        self, value: np.ndarray | None = None, *, kind: str = "wald"
    ) -> TestResult:
        """Joint test H₀: all estimand components equal value."""
        if kind not in ("wald", "empirical"):
            raise ValueError(f"kind must be 'wald' or 'empirical', got {kind!r}")

        if value is None:
            value_arr = self._null_to_inference(
                np.zeros_like(np.asarray(self.estimate)), "inference"
            )
        else:
            value_arr = self._null_to_inference(np.asarray(value), "reporting")

        est_inf = self._inference_estimate()
        est_arr = np.atleast_1d(np.asarray(est_inf))
        value_arr = np.atleast_1d(np.asarray(value_arr))
        if value_arr.shape != est_arr.shape:
            raise ValueError(
                f"value shape {value_arr.shape} does not match estimate shape {est_arr.shape}"
            )

        if self.gradient is not None and self.cov_params is not None:
            chi2, p, df = joint_wald(
                est_arr,
                self.gradient,
                self.cov_params,
                null_value=value_arr,
            )
            method = "joint_wald"
        else:
            draws_inf = self._inference_draws()
            if draws_inf is None:
                raise ValueError(
                    "Joint test requires either gradients/cov_params or draws."
                )
            draws = np.asarray(draws_inf)
            if draws.ndim == 1:
                draws = draws[:, None]
            centered = draws - est_arr
            emp_cov = np.cov(centered, rowvar=False)
            if emp_cov.ndim == 0:
                emp_cov = np.array([[emp_cov]])
            diff = est_arr - value_arr
            emp_cov_reg = emp_cov
            try:
                solved = np.linalg.solve(emp_cov_reg, diff)
            except np.linalg.LinAlgError:
                ridge = 1e-12 * float(np.trace(emp_cov)) / emp_cov.shape[0]
                ridge = max(ridge, float(np.finfo(emp_cov.dtype).eps))
                emp_cov_reg = emp_cov + ridge * np.eye(emp_cov.shape[0])
                solved = np.linalg.solve(emp_cov_reg, diff)

            if kind == "empirical":
                Q = np.sum(
                    (centered @ np.linalg.inv(emp_cov_reg)) * centered, axis=1
                )
                Q_obs = float(diff @ solved)
                p = float(np.mean(Q >= Q_obs))
                chi2 = Q_obs
            else:
                chi2 = float(diff @ solved)
                p = float(1.0 - stats.chi2.cdf(chi2, int(diff.shape[0])))
            df = int(diff.shape[0])
            method = "joint_empirical" if kind == "empirical" else "joint_wald"

        return TestResult(
            statistic=np.asarray(chi2),
            pvalue=np.asarray(p),
            df=df,
            null_value=value if value is not None else 0.0,
            alternative="two-sided",
            method=method,
            estimand_metadata=self.estimand_metadata,
        )

    def _null_to_inference(
        self, value: np.ndarray | float, null_scale: str
    ) -> np.ndarray:
        if null_scale == "inference" or self.phi_inv is None:
            return np.asarray(value)
        if null_scale == "reporting":
            return np.asarray(self.phi_inv(np.asarray(value)))
        raise ValueError(
            f"null_scale must be 'reporting' or 'inference', got {null_scale!r}"
        )

    # ------------------------------------------------------------------
    # Summary / formatting
    # ------------------------------------------------------------------

    def _summary_rows(self) -> list[dict]:
        """Build per-row summary data."""
        est = np.atleast_1d(self.estimate).ravel()
        se = np.atleast_1d(self.std_error).ravel()
        lo = np.atleast_1d(self.conf_int_lower).ravel()
        hi = np.atleast_1d(self.conf_int_upper).ravel()
        labels = self.labels
        if labels is None:
            labels = [f"[{i}]" for i in range(est.size)]

        z_vals = []
        p_vals = []
        stat_label = "z" if self.gradient is not None else "statistic"
        try:
            tr = self.test(value=0.0, null_scale="inference")
            z_vals = np.atleast_1d(tr.statistic).ravel()
            p_vals = np.atleast_1d(tr.pvalue).ravel()
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
        if p < levels[0]:
            return "***"
        if p < levels[1]:
            return "**"
        if p < levels[2]:
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
        """Human-readable summary with plan footer."""
        rows = self._summary_rows()
        has_stat = any("statistic" in r for r in rows)
        n_total = len(rows)
        truncated = False
        if max_rows is not None and n_total > max_rows:
            rows = rows[:max_rows]
            truncated = True

        data_keys = [("estimate", "estimate"), ("std_error", "std err")]
        if has_stat:
            stat_header = rows[0].get("stat_label", "z") if rows else "z"
            p_header = "P>|t|" if stat_header == "t" else "P>|z|"
            data_keys.extend([("statistic", stat_header), ("pvalue", p_header)])
        data_keys.append(("ci", f"[{self.level * 100:.0f}% Conf. Int.]"))

        def _fmt(key: str, row: dict) -> str:
            if key == "ci":
                return (
                    f"{row['ci_lower']:{float_fmt}}, {row['ci_upper']:{float_fmt}}"
                )
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

        fmt_rows = []
        for r in rows:
            fmt_rows.append([r["label"]] + [_fmt(k, r) for k, _ in data_keys])

        all_display = [[""] + [h for _, h in data_keys]] + fmt_rows
        widths = [
            max(len(r[i]) for r in all_display) for i in range(len(data_keys) + 1)
        ]

        def _line(cells: list[str], aligns: list[str]) -> str:
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
        title = f"Graph Result ({self.method}, level={self.level})"
        out_lines.append(title.center(total_width))
        out_lines.append(sep)
        out_lines.append(_line([""] + [h for _, h in data_keys], aligns))
        out_lines.append(thin)
        for r in fmt_rows:
            out_lines.append(_line(r, aligns))
        if truncated:
            out_lines.append(_line(["..."] + [""] * len(data_keys), aligns))
        out_lines.append(sep)

        footers = []
        if self.n_obs:
            footers.append(f"n = {self.n_obs}")
        if self.phi is not None:
            footers.append(
                "Note: std err is on the inference scale; estimate and CI are on the reporting scale."
            )
        if self.delta_sim_disagreement is not None:
            footers.append(
                f"Delta-vs-sim disagreement: {self.delta_sim_disagreement:.3%}"
            )

        plan_hash = getattr(self.plan, "hash", getattr(self.plan, "plan_hash", "?"))
        footer = f"plan {plan_hash}"
        if self.population_note:
            footer += f" | population: {self.population_note}"
        if self.kappa is not None:
            k = np.asarray(self.kappa)
            if not np.all(np.isnan(k)):
                footer += f" | κ = {float(k):.3f}" if k.ndim == 0 else f" | κ = max {float(np.nanmax(k)):.3f}"
        footers.append(footer)

        if self.imputation_diagnostic is not None:
            footers.append(self.imputation_diagnostic.footer())

        if footers:
            out_lines.extend([""] + footers)

        return "\n".join(out_lines)

    def to_frame(self) -> pd.DataFrame:
        """Return as a tidy DataFrame."""
        est = np.atleast_1d(self.estimate)
        se = np.atleast_1d(self.std_error)
        lo = np.atleast_1d(self.conf_int_lower)
        hi = np.atleast_1d(self.conf_int_upper)

        outcome_shape = self.estimand_metadata.get("_outcome_shape")
        if outcome_shape is not None and est.ndim == 2:
            n_atoms = outcome_shape["n_atoms"]
            n_outcomes = outcome_shape["n_outcomes"]
            outcome_labels = outcome_shape["outcome_labels"]
            est = est.ravel(order="C")
            se = se.ravel(order="C")
            lo = lo.ravel(order="C")
            hi = hi.ravel(order="C")
            n = est.size
            outcome_col = np.tile(outcome_labels, n_atoms)
        else:
            n = est.size
            outcome_col = None

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

        kind = self.estimand_metadata.get("kind", "")
        variables = self.estimand_metadata.get("variables")
        labels = self.labels
        if variables is not None:
            data["term"] = [list(variables) for _ in range(n)]
        elif labels is not None and len(labels) == n:
            data["term"] = labels
        else:
            data["term"] = np.repeat("", n)

        data["kind"] = np.repeat(kind, n)
        if labels is not None and len(labels) == n:
            data["label"] = labels

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

        over = self.estimand_metadata.get("over")
        if over is not None:
            data["over"] = np.repeat(",".join(over), n)
            over_values_meta = self.estimand_metadata.get("_over_values")
            if over_values_meta is not None and len(over_values_meta) == n:
                data["over_value"] = [
                    ", ".join(str(ov[o]) for o in over) for ov in over_values_meta
                ]
            elif labels is not None and len(labels) == n:
                import re

                over_values = []
                for lab in labels:
                    vals = []
                    for o in over:
                        m = re.search(rf"{o}=([^,]+)", lab)
                        if m:
                            vals.append(m.group(1).strip())
                    over_values.append(", ".join(vals) if vals else "")
                data["over_value"] = over_values

        if self.kappa is not None:
            kvals = np.atleast_1d(self.kappa)
            if outcome_shape is not None and kvals.ndim == 2:
                kvals = kvals.ravel(order="C")
            if kvals.size == n:
                data["kappa"] = kvals

        scenarios = self.estimand_metadata.get("scenarios")
        kind = self.estimand_metadata.get("kind")
        if scenarios is not None and kind in ("prediction", "slope", None):
            if len(scenarios) == n:
                all_keys = sorted(set().union(*(s.keys() for s in scenarios)))
                for key in all_keys:
                    data[key] = [s.get(key, np.nan) for s in scenarios]
            elif (
                outcome_shape is not None
                and len(scenarios) == outcome_shape["n_atoms"]
            ):
                n_outcomes = outcome_shape["n_outcomes"]
                tiled = []
                for s in scenarios:
                    tiled.extend([s] * n_outcomes)
                all_keys = sorted(set().union(*(s.keys() for s in scenarios)))
                for key in all_keys:
                    data[key] = [s.get(key, np.nan) for s in tiled]
            elif len(scenarios) > 1 and outcome_shape is not None:
                raise ValueError(
                    "to_frame() cannot unpack scenario columns for this "
                    "multi-outcome result. Use outcome=... to slice to a "
                    "single outcome first, or call outcome().to_frame()."
                )

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
        """LaTeX tabular representation."""
        rows = self._summary_rows()
        has_stat = any("statistic" in r for r in rows)

        data_keys = [("estimate", "estimate"), ("std_error", "std err")]
        if has_stat:
            stat_header = rows[0].get("stat_label", "z") if rows else "z"
            p_header = "P>|t|" if stat_header == "t" else "P>|z|"
            data_keys.extend([("statistic", stat_header), ("pvalue", p_header)])
        data_keys.append(("ci", f"[{self.level * 100:.0f}\\% Conf. Int.]"))

        def _fmt(key: str, row: dict) -> str:
            if key == "ci":
                return (
                    f"{row['ci_lower']:{float_fmt}}, {row['ci_upper']:{float_fmt}}"
                )
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
        """HTML table representation."""
        rows = self._summary_rows()
        has_stat = any("statistic" in r for r in rows)

        data_keys = [("estimate", "estimate"), ("std_error", "std err")]
        if has_stat:
            stat_header = rows[0].get("stat_label", "z") if rows else "z"
            p_header = "P>|t|" if stat_header == "t" else "P>|z|"
            data_keys.extend([("statistic", stat_header), ("pvalue", p_header)])
        data_keys.append(("ci", f"[{self.level * 100:.0f}% Conf. Int.]"))

        def _fmt(key: str, row: dict) -> str:
            if key == "ci":
                return (
                    f"{row['ci_lower']:{float_fmt}}, {row['ci_upper']:{float_fmt}}"
                )
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

        html_lines = []
        if caption:
            html_lines.append(f"<caption>{caption}</caption>")
        html_lines.append("<thead>")
        html_lines.append(
            "<tr>"
            + "".join(f"<th>{h}</th>" for h in ([""] + [h for _, h in data_keys]))
            + "</tr>"
        )
        html_lines.append("</thead>")
        html_lines.append("<tbody>")
        for r in rows:
            cells = [r["label"]] + [_fmt(k, r) for k, _ in data_keys]
            html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        html_lines.append("</tbody>")

        return '<table class="pymargins-result">\n' + "\n".join(html_lines) + "\n</table>"

    # ------------------------------------------------------------------
    # Outcome / composability
    # ------------------------------------------------------------------

    def outcome(
        self, index: int | str | list[int] | tuple[int, ...]
    ) -> GraphResult:
        """Slice a multi-outcome result to one or more outcomes."""
        outcome_shape = self.estimand_metadata.get("_outcome_shape")
        labels = self.labels
        est = np.atleast_1d(self.estimate)

        if outcome_shape is not None and est.ndim in (1, 2):
            n_atoms = outcome_shape["n_atoms"]
            n_outcomes = outcome_shape["n_outcomes"]
            outcome_labels = outcome_shape["outcome_labels"]

            # Normalize to a list of integer outcome indices.
            if isinstance(index, (list, tuple)):
                raw_indices = list(index)
            else:
                raw_indices = [index]

            resolved: list[int] = []
            for idx in raw_indices:
                if isinstance(idx, str):
                    if idx not in outcome_labels:
                        raise ValueError(
                            f"Outcome label {idx!r} not found. "
                            f"Available: {outcome_labels}"
                        )
                    resolved.append(outcome_labels.index(idx))
                else:
                    idx_int = int(idx)
                    if not (0 <= idx_int < n_outcomes):
                        raise ValueError(
                            f"Outcome index {idx_int} out of range "
                            f"(0..{n_outcomes - 1})."
                        )
                    resolved.append(idx_int)

            if len(resolved) == 1:
                outcome_idx = resolved[0]

                def _slice(arr):
                    if arr is None:
                        return None
                    a = np.asarray(arr)
                    if a.ndim == 2 and a.shape == (n_atoms, n_outcomes):
                        return a[:, outcome_idx]
                    if n_atoms == 1 and a.ndim == 2 and a.shape[0] == n_outcomes:
                        return a[outcome_idx : outcome_idx + 1]
                    if a.ndim == 3 and a.shape[:2] == (n_atoms, n_outcomes):
                        return a[:, outcome_idx, :]
                    if a.ndim == 3 and a.shape[1:] == (n_atoms, n_outcomes):
                        return a[:, :, outcome_idx]
                    if a.ndim == 1 and a.size == n_atoms * n_outcomes:
                        mask = np.arange(a.size) % n_outcomes == outcome_idx
                        return a[mask]
                    return arr

                new_labels = (
                    [labels[i] for i in range(len(labels)) if i % n_outcomes == outcome_idx]
                    if labels
                    else None
                )
            else:
                outcome_indices = resolved

                def _slice(arr):
                    if arr is None:
                        return None
                    a = np.asarray(arr)
                    if a.ndim == 2 and a.shape == (n_atoms, n_outcomes):
                        return a[:, outcome_indices]
                    if n_atoms == 1 and a.ndim == 2 and a.shape[0] == n_outcomes:
                        return a[outcome_indices]
                    if a.ndim == 3 and a.shape[:2] == (n_atoms, n_outcomes):
                        return a[:, outcome_indices, :]
                    if a.ndim == 3 and a.shape[1:] == (n_atoms, n_outcomes):
                        return a[:, :, outcome_indices]
                    if a.ndim == 1 and a.size == n_atoms * n_outcomes:
                        mask = np.isin(np.arange(a.size) % n_outcomes, outcome_indices)
                        return a[mask]
                    return arr

                new_labels = (
                    [labels[i] for i in range(len(labels)) if i % n_outcomes in outcome_indices]
                    if labels
                    else None
                )

            new_meta = dict(self.estimand_metadata)
            new_meta["labels"] = new_labels
            new_meta["outcome_sliced"] = True
            new_meta.pop("_outcome_shape", None)

            return self._replace(
                estimate=_slice(self.estimate),
                std_error=_slice(self.std_error),
                conf_int_lower=_slice(self.conf_int_lower),
                conf_int_upper=_slice(self.conf_int_upper),
                labels=new_labels,
                estimand_metadata=new_meta,
                kappa=_slice(self.kappa),
                gradient=_slice(self.gradient),
                draws=_slice(self.draws),
                draws_inf=_slice(self.draws_inf),
                psi_h=None,
            )

        # Legacy label-heuristic path
        n_components = est.size
        n_labels = len(labels) if labels else 0
        if n_labels == 0 or n_labels != n_components:
            raise ValueError(
                "outcome() requires expanded outcome labels; "
                "this result may not be from a multi-outcome model."
            )

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

        mask = np.arange(n_components) % K == outcome_idx
        if not np.any(mask):
            raise ValueError(f"No components found for outcome {index!r}.")

        def _slice_legacy(arr):
            if arr is None:
                return None
            a = np.asarray(arr)
            if a.ndim == 1:
                return a[mask]
            if a.ndim == 2:
                if a.shape[0] == n_components:
                    return a[mask]
                if a.shape[1] == n_components:
                    return a[:, mask]
                return a
            if a.ndim == 3:
                if a.shape[1] == n_components:
                    return a[:, mask]
                if a.shape[2] == n_components:
                    return a[:, :, mask]
                return a
            return a

        new_labels = [labels[i] for i in np.where(mask)[0]]
        new_meta = dict(self.estimand_metadata)
        new_meta["labels"] = new_labels

        return self._replace(
            estimate=_slice_legacy(self.estimate),
            std_error=_slice_legacy(self.std_error),
            conf_int_lower=_slice_legacy(self.conf_int_lower),
            conf_int_upper=_slice_legacy(self.conf_int_upper),
            labels=new_labels,
            estimand_metadata=new_meta,
            kappa=_slice_legacy(self.kappa),
            gradient=_slice_legacy(self.gradient),
            draws=_slice_legacy(self.draws),
            draws_inf=_slice_legacy(self.draws_inf),
            psi_h=None,
        )

    def scaled(self, by: float, units: str = "") -> GraphResult:
        """Cosmetic rescaling of the estimate and CI for reporting."""
        new = self._replace(
            estimate=self.estimate * by,
            std_error=self.std_error * abs(by),
            conf_int_lower=(
                self.conf_int_lower * by
                if by >= 0
                else self.conf_int_upper * by
            ),
            conf_int_upper=(
                self.conf_int_upper * by
                if by >= 0
                else self.conf_int_lower * by
            ),
            gradient=(self.gradient * by if self.gradient is not None else None),
            draws=(self.draws * by if self.draws is not None else None),
            draws_inf=(self.draws_inf * by if self.draws_inf is not None else None),
            psi_h=(self.psi_h * by if self.psi_h is not None else None),
        )
        meta = dict(self.estimand_metadata)
        meta["labels"] = [
            f"({lbl})*{by}" for lbl in self.labels or []
        ]
        if units:
            meta["units"] = units
        new.estimand_metadata = meta
        new.labels = meta["labels"]
        return new

    def contrast(self, C: np.ndarray, labels: list[str] | None = None) -> GraphResult:
        """Apply a contrast matrix to a vector result."""
        if self.gradient is None:
            raise ValueError(
                "contrast() requires a delta-method result. For sim/boot, "
                "apply C to draws_inf manually."
            )
        C = jnp.asarray(C)
        est_inf = self._inference_estimate()
        new_est_inf = C @ jnp.asarray(est_inf)
        new_grad = C @ jnp.asarray(self.gradient)  # (m, p)
        cov = jnp.asarray(self.cov_params)
        var = jnp.einsum("ij,jk,ik->i", new_grad, cov, new_grad)
        se = np.asarray(jnp.sqrt(var))
        z = stats.norm.ppf(0.5 + self.level / 2.0)
        lo_inf = new_est_inf - z * se
        hi_inf = new_est_inf + z * se

        new_est = np.asarray(self.phi(new_est_inf)) if self.phi else np.asarray(new_est_inf)
        new_lo = np.asarray(self.phi(lo_inf)) if self.phi else np.asarray(lo_inf)
        new_hi = np.asarray(self.phi(hi_inf)) if self.phi else np.asarray(hi_inf)

        return self._replace(
            estimate=new_est,
            std_error=se,
            conf_int_lower=new_lo,
            conf_int_upper=new_hi,
            labels=labels or [f"contrast[{i}]" for i in range(C.shape[0])],
            estimand_metadata={
                **self.estimand_metadata,
                "labels": labels or [f"contrast[{i}]" for i in range(C.shape[0])],
            },
            gradient=np.asarray(new_grad),
            draws=None,
            draws_inf=None,
            psi_h=None,
        )

    def pairwise_contrasts(self, labels: list[str] | None = None) -> GraphResult:
        """All pairwise differences between components of a vector result."""
        if self.gradient is None:
            raise ValueError("pairwise_contrasts() requires a delta-method result.")
        est = np.atleast_1d(self.estimate)
        k = int(est.size)
        if k < 2:
            raise ValueError("pairwise_contrasts() requires at least 2 components")
        use_labels = labels if labels is not None else self.labels
        if use_labels is None or len(use_labels) != k:
            use_labels = [f"[{i}]" for i in range(k)]
        C = diff_matrix(k, kind="pairwise")
        new_labels = []
        for i in range(k):
            for j in range(i + 1, k):
                new_labels.append(f"{use_labels[j]} - {use_labels[i]}")
        return self.contrast(C, labels=new_labels)

    # ------------------------------------------------------------------
    # Arithmetic composability
    # ------------------------------------------------------------------

    def __neg__(self) -> GraphResult:
        """Unary minus; cosmetic rescale that preserves covariance structure."""
        return self.scaled(-1.0)

    def __add__(self, other: Any) -> GraphResult:
        if isinstance(other, GraphResult):
            return self._binary_op(other, weights=(1.0, 1.0), op="+")
        return NotImplemented

    def __sub__(self, other: Any) -> GraphResult:
        if isinstance(other, GraphResult):
            return self._binary_op(other, weights=(1.0, -1.0), op="-")
        return NotImplemented

    def _binary_op(
        self,
        other: GraphResult,
        weights: tuple[float, float],
        op: str,
    ) -> GraphResult:
        """Combine two results from the same plan with propagated uncertainty.

        The combination is performed on the inference scale so that log- or
        other link-scale results remain coherent.  For simulation/bootstrap
        results the draws are assumed to be paired (same estimator/plan), which
        is the normal case when two queries reuse a compiled estimator.
        """
        self._assert_compatible_for_combination(other)
        w0, w1 = weights

        est_inf_self = np.atleast_1d(self._inference_estimate())
        est_inf_other = np.atleast_1d(other._inference_estimate())
        if est_inf_self.shape != est_inf_other.shape:
            raise ValueError(
                "Cannot combine results with different estimate shapes: "
                f"{est_inf_self.shape} vs {est_inf_other.shape}."
            )

        new_est_inf = w0 * est_inf_self + w1 * est_inf_other
        new_estimate = (
            np.asarray(self.phi(new_est_inf))
            if self.phi is not None
            else np.asarray(new_est_inf)
        )

        labels = self.labels or [f"[{i}]" for i in range(new_estimate.size)]
        other_labels = other.labels or [f"[{i}]" for i in range(new_estimate.size)]
        new_labels = [
            f"{s} {op} {o}" for s, o in zip(labels, other_labels, strict=False)
        ]

        # Delta-method combination ---------------------------------------
        if self.gradient is not None and other.gradient is not None:
            if self.cov_params is None or other.cov_params is None:
                raise ValueError("Delta-method results require cov_params to combine.")
            if self.cov_params.shape != other.cov_params.shape:
                raise ValueError(
                    "Cannot combine delta results with different cov_params shapes."
                )
            grad_self = np.atleast_2d(np.asarray(self.gradient))
            grad_other = np.atleast_2d(np.asarray(other.gradient))
            if grad_self.shape != grad_other.shape:
                raise ValueError(
                    "Cannot combine delta results with different gradient shapes."
                )
            new_gradient = w0 * grad_self + w1 * grad_other
            cov_params = np.asarray(self.cov_params)
            se_inf = np.asarray(delta_se(jnp.asarray(new_gradient), jnp.asarray(cov_params)))
            lo_inf, hi_inf = delta_interval(
                new_est_inf, new_gradient, cov_params, self.level, phi=self.phi
            )
            psi_h = None
            if self.psi_h is not None and other.psi_h is not None:
                psi_self = np.atleast_2d(np.asarray(self.psi_h))
                psi_other = np.atleast_2d(np.asarray(other.psi_h))
                if psi_self.shape == psi_other.shape:
                    psi_h = w0 * psi_self + w1 * psi_other
            draws = None
            draws_inf = None
            ci_method = self.ci_method or "wald"
        # Simulation/bootstrap combination --------------------------------
        elif self._inference_draws() is not None and other._inference_draws() is not None:
            draws_self = np.asarray(self._inference_draws())
            draws_other = np.asarray(other._inference_draws())
            # Scalar results store draws as (B,); reshape to (B, 1) for combining.
            if draws_self.ndim == 1:
                draws_self = draws_self[:, None]
            if draws_other.ndim == 1:
                draws_other = draws_other[:, None]
            if draws_self.shape != draws_other.shape:
                raise ValueError(
                    "Cannot combine draw-based results with different draw shapes: "
                    f"{draws_self.shape} vs {draws_other.shape}."
                )
            new_draws_inf = w0 * draws_self + w1 * draws_other
            se_inf = np.asarray(np.std(new_draws_inf, ddof=1, axis=0))
            lo_inf, hi_inf = draws_interval(
                new_draws_inf,
                self.level,
                phi=self.phi,
                ci_method="percentile",
                bootstrap_extras=None,
            )
            new_est_inf_from_draws = np.asarray(np.mean(new_draws_inf, axis=0))
            new_estimate = (
                np.asarray(self.phi(new_est_inf_from_draws))
                if self.phi is not None
                else np.asarray(new_est_inf_from_draws)
            )
            draws_inf = np.asarray(new_draws_inf)
            draws = (
                np.asarray(self.phi(new_draws_inf))
                if self.phi is not None
                else np.asarray(new_draws_inf)
            )
            new_gradient = None
            cov_params = None
            psi_h = None
            ci_method = "percentile"
        else:
            raise ValueError(
                "Cannot combine results: both must be delta-method results or both "
                "must have simulation/bootstrap draws."
            )

        if new_estimate.ndim == 0 and new_est_inf.size == 1:
            new_estimate = np.atleast_1d(new_estimate)
        se_inf = np.atleast_1d(se_inf)
        lo_inf = np.atleast_1d(lo_inf)
        hi_inf = np.atleast_1d(hi_inf)

        meta = {
            **self.estimand_metadata,
            "labels": new_labels,
            "kind": "combined",
            "operation": op,
        }

        return self._replace(
            estimate=new_estimate,
            std_error=se_inf,
            conf_int_lower=lo_inf,
            conf_int_upper=hi_inf,
            labels=new_labels,
            estimand_metadata=meta,
            gradient=new_gradient,
            cov_params=cov_params,
            draws=draws,
            draws_inf=draws_inf,
            psi_h=psi_h,
            ci_method=ci_method,
            bootstrap_extras=None,
            delta_sim_disagreement=None,
            n_boot_effective=None,
            n_boot_failed=None,
        )

    def _assert_compatible_for_combination(self, other: GraphResult) -> None:
        if not isinstance(other, GraphResult):
            raise TypeError(f"Cannot combine GraphResult with {type(other).__name__}.")

        self_hash = getattr(self.plan, "plan_hash", getattr(self.plan, "hash", None))
        other_hash = getattr(other.plan, "plan_hash", getattr(other.plan, "hash", None))
        if self_hash is not None and other_hash is not None and self_hash != other_hash:
            raise ValueError(
                "Cannot combine results from different plans. "
                "Declare a single estimator and issue both queries against it."
            )
        if self.method != other.method:
            raise ValueError(
                f"Cannot combine results with different methods: "
                f"{self.method!r} vs {other.method!r}."
            )
        if self.level != other.level:
            raise ValueError(
                f"Cannot combine results with different confidence levels: "
                f"{self.level} vs {other.level}."
            )
        self_phi_name = _phi_to_name(self.phi)
        other_phi_name = _phi_to_name(other.phi)
        if self_phi_name != other_phi_name:
            raise ValueError(
                "Cannot combine results with different reporting scales."
            )

    # ------------------------------------------------------------------
    # Influence
    # ------------------------------------------------------------------

    def influence(self) -> np.ndarray:
        """Per-observation influence on the estimand."""
        if self.psi_h is not None:
            return np.asarray(self.psi_h)
        if self.ci_method == "bca" and self.bootstrap_extras:
            jack = self.bootstrap_extras.get("influence_jackknife")
            if jack is not None:
                theta_minus = np.asarray(jack)
                est_inf = self._inference_estimate()
                return np.asarray(est_inf) - theta_minus
        raise ValueError(
            "Influence is not available for this result. "
            "Compute the estimand with method='delta' and an adapter that "
            "exposes score_obs(), or with bootstrap ci_method='bca'."
        )

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def to_disk(self, path: str | Path) -> None:
        """Serialize result to disk. Only pickle is supported."""
        phi_name = _phi_to_name(self.phi)
        phi_inv_name = _phi_to_name(self.phi_inv)
        if phi_name is None and self.phi is not None:
            raise ValueError(
                "Cannot serialize GraphResult: phi is a custom function and "
                "cannot be pickled. Use a named scale, or pickle the arrays "
                "manually."
            )
        if phi_inv_name is None and self.phi_inv is not None:
            raise ValueError(
                "Cannot serialize GraphResult: phi_inv is a custom function and "
                "cannot be pickled. Use a named scale, or pickle the arrays "
                "manually."
            )

        payload = {
            "format_version": 1,
            **self.__dict__,
        }
        payload["phi_name"] = phi_name
        payload["phi_inv_name"] = phi_inv_name
        payload.pop("phi", None)
        payload.pop("phi_inv", None)

        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def from_disk(cls, path: str | Path) -> GraphResult:
        """Deserialize result from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("format_version") != 1:
            raise ValueError(
                f"Unsupported GraphResult format version: {payload.get('format_version')!r}"
            )
        phi = _name_to_phi(payload.get("phi_name"))
        phi_inv = _name_to_phi(payload.get("phi_inv_name"))
        payload.pop("phi_name", None)
        payload.pop("phi_inv_name", None)
        payload.pop("format_version", None)
        payload["phi"] = phi
        payload["phi_inv"] = phi_inv
        return cls(**payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _result(self) -> GraphResult:
        """Compatibility shim for tests/anchor/ during the R4-R6 window."""
        return self

    def _replace(self, **changes) -> GraphResult:
        """Return a copy with replaced fields."""
        from dataclasses import replace

        return replace(self, **changes)
