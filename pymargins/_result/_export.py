"""
pymargins._result._export

Export, formatting, and reporting methods for MarginsResult.
"""

from __future__ import annotations
from typing import Any, Optional
import re
import warnings

import numpy as np
import pandas as pd

from ._margins import MarginsResult


# ---------------------------------------------------------------------------
# Summary / reporting helpers
# ---------------------------------------------------------------------------

def _summary_rows(self: MarginsResult):
    """Build per-row summary data as a list of dicts.

    Per-row p-values test H0: estimate = 0 on the *inference scale*.
    For logit-scale predictions this means testing against p = 0.5,
    which may not be the intended null.
    """
    est = np.atleast_1d(self.estimate)
    se = np.atleast_1d(self.std_error)
    lo = np.atleast_1d(self.conf_int_lower)
    hi = np.atleast_1d(self.conf_int_upper)
    labels = self.estimand_metadata.get("labels")
    if labels is None:
        labels = [f"[{i}]" for i in range(est.size)]

    z_vals = []
    p_vals = []
    is_delta = self.gradient is not None
    # Note: null=0 is on the inference scale. For logit-scale predictions,
    # this tests H0: logit(p)=0 i.e. p=0.5, which is rarely the intended null.
    try:
        tr = self.test(value=0.0, null_scale="inference")
        z_vals = np.atleast_1d(tr.statistic)
        p_vals = np.atleast_1d(tr.pvalue)
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
            row["stat_label"] = "z" if is_delta else "statistic"
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
    self: MarginsResult,
    stars: bool = False,
    star_levels: tuple[float, float, float] = (0.01, 0.05, 0.10),
    float_fmt: str = ".4f",
    pvalue_fmt: str = ".3f",
    max_rows: Optional[int] = None,
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
        data_keys.extend([("statistic", stat_header), ("pvalue", "P>|z|")])
    data_keys.append(("ci", f"[{self.level*100:.0f}% Conf. Int.]"))

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
    widths = [max(len(r[i]) for r in all_display) for i in range(len(data_keys) + 1)]

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
    out_lines.append(_line([h for _, h in data_keys], aligns))
    out_lines.append(thin)
    for r in fmt_rows:
        out_lines.append(_line(r, aligns))
    if truncated:
        out_lines.append(_line(["..."] + [""] * (len(data_keys) - 1), aligns))
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
        footers.append(f"Delta-vs-sim disagreement: {self.delta_sim_disagreement:.3%}")
    if footers:
        out_lines.extend([""] + footers)

    return "\n".join(out_lines)


def to_frame(self: MarginsResult) -> pd.DataFrame:
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
    n = est.size

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
                    m = re.search(rf"{o}=([^,]+)", lab)
                    if m:
                        vals.append(m.group(1).strip())
                over_values.append(", ".join(vals) if vals else "")
            data["over_value"] = over_values

    # Diagnostics
    if self.kappa is not None:
        kvals = np.atleast_1d(self.kappa)
        if kvals.size == n:
            data["kappa"] = kvals
    data["fallback_triggered"] = np.repeat(self.fallback_triggered, n)
    if self.fallback_reason:
        data["fallback_reason"] = np.repeat(self.fallback_reason, n)

    # Scenario columns
    scenarios = self.estimand_metadata.get("scenarios")
    kind = self.estimand_metadata.get("kind")
    if scenarios is not None and len(scenarios) == n and kind in ("prediction", "slope", None):
        all_keys = sorted(set().union(*(s.keys() for s in scenarios)))
        for key in all_keys:
            col_values = [s.get(key, np.nan) for s in scenarios]
            data[key] = col_values

    return pd.DataFrame(data)


def to_latex(
    self: MarginsResult,
    stars: bool = False,
    star_levels: tuple[float, float, float] = (0.01, 0.05, 0.10),
    float_fmt: str = ".4f",
    pvalue_fmt: str = ".3f",
    caption: Optional[str] = None,
    label: Optional[str] = None,
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
        data_keys.extend([("statistic", stat_header), ("pvalue", "P>|z|")])
    data_keys.append(("ci", f"[{self.level*100:.0f}\\% Conf. Int.]"))

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
    self: MarginsResult,
    stars: bool = False,
    star_levels: tuple[float, float, float] = (0.01, 0.05, 0.10),
    float_fmt: str = ".4f",
    pvalue_fmt: str = ".3f",
    caption: Optional[str] = None,
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
        data_keys.extend([("statistic", stat_header), ("pvalue", "P>|z|")])
    data_keys.append(("ci", f"[{self.level*100:.0f}% Conf. Int.]"))

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
    lines.append("<tr>" + "".join(f"<th>{h}</th>" for h in ([""] + [h for _, h in data_keys])) + "</tr>")
    lines.append("</thead>")
    lines.append("<tbody>")
    for r in rows:
        cells = [r["label"]] + [_fmt(k, r) for k, _ in data_keys]
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    lines.append("</tbody>")

    return '<table class="pymargins-result">\n' + "\n".join(lines) + "\n</table>"


# ---------------------------------------------------------------------------
# Patch methods onto MarginsResult
# ---------------------------------------------------------------------------

MarginsResult._summary_rows = _summary_rows
MarginsResult._star_notation = _star_notation
MarginsResult.summary = summary
MarginsResult.to_frame = to_frame
MarginsResult.to_latex = to_latex
MarginsResult.to_html = to_html
