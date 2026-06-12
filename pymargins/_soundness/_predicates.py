"""Severity-routed soundness predicates.

Implements the structural and statistical soundness checks from design §6.
Each predicate appends to a :class:`CompileReport`.  Phase-2 reachable
predicates only; the module grows in Phases 3 and 4.
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass
from typing import Any

from ._constants import (
    BCA_MIN_B,
    FEW_CLUSTERS_WARN,
    SE_ONLY_MIN_B,
    TAIL_COUNT_NOTE,
    TAIL_COUNT_WARN,
)


class Severity(enum.Enum):
    REFUSE = "refuse"
    WARN = "warn"
    NOTE = "note"


class CompileError(ValueError):
    """Raised when a soundness predicate refuses compilation."""


class SoundnessWarning(UserWarning):
    """Emitted when a soundness predicate warns."""


@dataclass(frozen=True)
class CompileReport:
    """Immutable collection of soundness findings."""

    entries: tuple[tuple[Severity, str, str], ...] = ()

    def append(self, severity: Severity, code: str, message: str) -> CompileReport:
        return CompileReport(entries=self.entries + ((severity, code, message),))

    def has(self, severity: Severity | None = None, code: str | None = None) -> bool:
        for s, c, _ in self.entries:
            if severity is not None and s != severity:
                continue
            if code is not None and c != code:
                continue
            return True
        return False

    def raise_for_refusals(self) -> None:
        for s, c, msg in self.entries:
            if s == Severity.REFUSE:
                raise CompileError(f"[{c}] {msg}")

    def emit_warnings(self) -> None:
        for s, c, msg in self.entries:
            if s == Severity.WARN:
                warnings.warn(f"[{c}] {msg}", SoundnessWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# Phase-2 predicates
# ---------------------------------------------------------------------------


def check_method_adapter_compatibility(
    method: str,
    supported: set[str],
    report: CompileReport,
) -> CompileReport:
    """Refuse if the requested method is not in the adapter's supported set."""
    if method not in supported:
        report = report.append(
            Severity.REFUSE,
            "method_unsupported",
            f'method="{method}" is not supported by this adapter. '
            f"Supported: {sorted(supported)}.",
        )
    return report


def check_ci_method_compatibility(
    ci: str,
    method: str,
    report: CompileReport,
) -> CompileReport:
    """Refuse incompatible ci × method pairings."""
    if ci in ("percentile", "bca", "basic") and method == "delta":
        report = report.append(
            Severity.REFUSE,
            "ci_method_incompatible",
            f'ci="{ci}" × method="delta" is incompatible. '
            f'Delta method produces Wald-type intervals; use ci="wald" or '
            f'switch method to "bootstrap".',
        )
    if ci == "studentized" and method != "bootstrap":
        report = report.append(
            Severity.REFUSE,
            "ci_method_incompatible",
            'ci="studentized" requires method="bootstrap". '
            'Per-replicate SEs are computed during the bootstrap run.',
        )
    return report


def check_tail_count_adequacy(
    B: int,
    level: float,
    ci: str,
    report: CompileReport,
) -> CompileReport:
    """Note/warn when bootstrap tail counts are inadequate."""
    if B <= 0:
        return report
    tail = B * (1 - level) / 2
    if ci == "bca" and B < BCA_MIN_B:
        report = report.append(
            Severity.NOTE,
            "bca_small_b",
            f"BCa with B={B} < {BCA_MIN_B}: acceleration/tail quantiles may be unstable.",
        )
    if ci == "se" and B < SE_ONLY_MIN_B:
        report = report.append(
            Severity.NOTE,
            "se_small_b",
            f"Bootstrap SEs with B={B} < {SE_ONLY_MIN_B}: SE estimates may be noisy.",
        )
    if tail < TAIL_COUNT_WARN:
        report = report.append(
            Severity.WARN,
            "tail_count_low",
            f"Bootstrap tail count ≈ {tail:.1f} (< {TAIL_COUNT_WARN}) at level={level}: "
            f"percentile tails are poorly estimated. Increase B or use ci=\"se\".",
        )
    elif tail < TAIL_COUNT_NOTE:
        report = report.append(
            Severity.NOTE,
            "tail_count_low",
            f"Bootstrap tail count ≈ {tail:.1f} (< {TAIL_COUNT_NOTE}) at level={level}: "
            f"percentile tails are moderately sparse.",
        )
    return report


def check_cluster_count(
    n_clusters: int | None,
    report: CompileReport,
) -> CompileReport:
    """Warn when the number of bootstrap clusters is small."""
    if n_clusters is not None and n_clusters < FEW_CLUSTERS_WARN:
        report = report.append(
            Severity.WARN,
            "few_clusters",
            f"Cluster bootstrap with G={n_clusters} (< {FEW_CLUSTERS_WARN}): "
            f"consider t with G−1 df or wild cluster bootstrap *(future)*.",
        )
    return report


def check_lonely_psu(
    design: Any | None,
    report: CompileReport,
) -> CompileReport:
    """Refuse if any stratum has only one PSU."""
    if design is None:
        return report
    psu = getattr(design, "psu", None)
    strata = getattr(design, "strata", None)
    if psu is None or strata is None:
        return report
    import numpy as np

    psu_arr = np.asarray(psu)
    strata_arr = np.asarray(strata)
    for h in np.unique(strata_arr):
        in_h = strata_arr == h
        n_h = len(np.unique(psu_arr[in_h]))
        if n_h < 2:
            report = report.append(
                Severity.REFUSE,
                "lonely_psu",
                f"Stratum {h!r} has only {n_h} PSU(s). "
                f"Collapse this stratum or declare a certainty PSU.",
            )
    return report
