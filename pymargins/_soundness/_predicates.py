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
    ESS_NOTE_FRACTION,
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
            "Per-replicate SEs are computed during the bootstrap run.",
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
            f'percentile tails are poorly estimated. Increase B or use ci="se".',
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
    """Warn when the number of clusters is small for cluster-robust inference."""
    if n_clusters is not None and n_clusters < FEW_CLUSTERS_WARN:
        report = report.append(
            Severity.WARN,
            "few_clusters",
            f"Cluster-robust inference with G={n_clusters} (< {FEW_CLUSTERS_WARN}): "
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


def check_ess(
    weights: Any | None,
    report: CompileReport,
) -> CompileReport:
    """Note when the effective sample size is low relative to n."""
    if weights is None:
        return report
    import numpy as np

    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return report
    ess = float(np.sum(w) ** 2 / np.sum(w**2))
    n = w.shape[0]
    if ess / n < ESS_NOTE_FRACTION:
        report = report.append(
            Severity.NOTE,
            "ess_low",
            f"ESS = {ess:.1f} / n = {n} ({ess / n:.1%}); "
            f"declared weights carry less than half the information of the nominal sample.",
        )
    return report


# ---------------------------------------------------------------------------
# Registry of design §6 soundness rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoundnessRow:
    """One row from the design §6 soundness tables.

    ``predicate`` is the qualname of the implementing check, or ``None`` for
    rows that are not yet implemented (they still carry the verbatim text).
    """

    id: str
    design_section: str
    severity: str
    text: str
    predicate: str | None = None


SOUNDNESS_ROWS: tuple[SoundnessRow, ...] = (
    # §6.1 Structural
    SoundnessRow(
        "6.1-method-unsupported",
        "§6.1",
        "refuse",
        'method="{method}" is not supported by this adapter. Supported: {supported}.',
        predicate="pymargins._soundness._predicates.check_method_adapter_compatibility",
    ),
    SoundnessRow(
        "6.1-nondiff-compose-delta",
        "§6.1",
        "refuse",
        "Non-differentiable estimand under method='delta'. "
        'Use method="simulation" (§4.8).',
        predicate="pymargins._engine._execute.execute_query",
    ),
    SoundnessRow(
        "6.1-ci-method-incompatible",
        "§6.1",
        "refuse",
        'ci="studentized" requires method="bootstrap". '
        "Per-replicate SEs are computed during the bootstrap run.",
        predicate="pymargins._soundness._predicates.check_ci_method_compatibility",
    ),
    SoundnessRow(
        "6.1-ci-method-incompatible-delta",
        "§6.1",
        "refuse",
        'ci="percentile"/"bca"/"basic" requires method!="delta". '
        "Delta method produces Wald-type intervals; use ci='wald' or method='bootstrap'.",
        predicate="pymargins._soundness._predicates.check_ci_method_compatibility",
    ),
    SoundnessRow(
        "6.1-studentized-no-per-replicate-se",
        "§6.1",
        "refuse",
        "studentized bootstrap requires a per-replicate SE (analytic SE per replicate or nested resampling).",
    ),
    SoundnessRow(
        "6.1-delta-sim-no-influence",
        "§6.1",
        "refuse",
        "delta/sim over a no-influence() stage (matching) is not supported by this adapter. "
        "Use a method supported by the adapter.",
    ),
    SoundnessRow(
        "6.1-delta-bootstrap-only-adapter",
        "§6.1",
        "refuse",
        "method='delta' on a no-score adapter (BootstrapOnlyAdapter) is not supported. "
        "Use method='bootstrap'.",
    ),
    SoundnessRow(
        "6.1-mixed-methods-across-branches",
        "§6.1",
        "unrepresentable",
        "Mixed inference methods across fan branches are unrepresentable; method resolves once (§5.2).",
    ),
    SoundnessRow(
        "6.1-bootmi-vs-miboot-ambiguity",
        "§6.1",
        "unrepresentable",
        "boot(mi) vs mi(boot) nesting ambiguity is unrepresentable; spell impute vs reimpute (§4.4).",
    ),
    SoundnessRow(
        "6.1-frozen-estimated-stage",
        "§6.1",
        "unrepresentable",
        "Frozen estimated stage under a transform is unrepresentable; stages linearize or re-execute, never freeze. "
        "Freezing errs in a functional-dependent direction: conservative for IPW-ATE (Hirano–Imbens–Ridder 2003; "
        "Lunceford–Davidian 2004), anti-conservative for outcome-side two-step nuisances — invalid either way.",
    ),
    SoundnessRow(
        "6.1-forgotten-dependence",
        "§6.1",
        "unrepresentable",
        "Forgotten dependence in the analytic Σ̂ is unrepresentable; dependence declares once at steps.input and "
        "flows to VarianceScheme and resampler alike (§4.1).",
    ),
    SoundnessRow(
        "6.1-match-filter-04-refusal",
        "§6.1",
        "refuse",
        "match + row-filter stages in one wiring lands with the fan engine in 0.5.0; "
        "today, apply filters before matching outside the wiring or use matching alone.",
    ),
    # §6.2 Resampling × non-smooth stages
    SoundnessRow(
        "6.2-bootstrap-nn-matching-with-replacement",
        "§6.2",
        "warn",
        "bootstrap ∘ NN matching with replacement is inconsistent (Abadie–Imbens 2008; failure can go either direction). "
        "Steers: analytic AI variance (Abadie–Imbens 2006), "
        "Otsu–Rai weighted bootstrap *(future — the modern fix: bootstraps the martingale representation, no re-matching, no tuning)*, "
        "m-out-of-n.",
    ),
    SoundnessRow(
        "6.2-bootstrap-ps-matching-without-replacement",
        "§6.2",
        "warn",
        "bootstrap ∘ PS matching without replacement (re-estimate PS + re-match per replicate — what the engine does) "
        "has no general validity theory; simulation evidence is mildly conservative (Austin & Small 2014). "
        "Alternatives: pair-cluster bootstrap (resample matched pairs — same paper), "
        "AI-2016 analytic variance for estimated-PS matching.",
    ),
    SoundnessRow(
        "6.2-m-out-of-n",
        "§6.2",
        "conditional",
        "m-out-of-n requires m→∞, m/n→0; default m = ⌈n^{2/3}⌉, CI rescaled by √(m/n); Bickel–Sakov adaptive choice *(future)*.",
    ),
    SoundnessRow(
        "6.2-matching-estimated-ps-analytic",
        "§6.2",
        "conditional",
        "matching × estimated PS analytic route requires the PS-estimation correction (Abadie–Imbens 2016); "
        "it is not covered by naive AI-2006 variance; surface as a distinct steer, don't conflate.",
    ),
    SoundnessRow(
        "6.2-long-format-row-resampling",
        "§6.2",
        "refuse",
        "long-format data (time-varying Cox, panel) × row resampling is invalid; resampling unit = subject/entity. "
        "Episode structure detected → demand the cluster declaration at steps.input.",
    ),
    # §6.3 Fans × views
    SoundnessRow(
        "6.3-impute-fan-est-se-rubin",
        "§6.3",
        "sound",
        "impute-fan × (est, SE) view → Rubin is sound (Rubin 1987; Barnard–Rubin 1999 df; "
        "W_m may be bootstrap-SE (Schomaker–Heumann 2018 'MI Boot pooled SE'), costs M×B — compile note points at reimpute).",
    ),
    SoundnessRow(
        "6.3-impute-fan-quantile-refuse",
        "§6.3",
        "refuse",
        "impute-fan × quantile view is refused; quantiles don't pool. Steer steps.reimpute or ci='se'.",
    ),
    SoundnessRow(
        "6.3-reimpute-percentile-sound",
        "§6.3",
        "sound",
        "reimpute (boot-MI) × percentile is sound (Schomaker–Heumann 2018); "
        "also the uncongeniality-robust choice (Bartlett–Hughes 2020); "
        "efficient pooled variants exist (von Hippel-style two-way decomposition) *(future)*.",
    ),
    SoundnessRow(
        "6.3-rubin-uncongenial",
        "§6.3",
        "note",
        "Rubin × uncongenial imputer/analyst pair: Rubin's T is biased when imputation and analysis models "
        "aren't congenial (Meng 1994): conservative when the imputer is richer, anti-conservative when poorer "
        "(Robins–Wang 2000). Heuristic predicate: imputer family ∉ analyst family (e.g., forest-imputer + GLM) "
        "→ note recommending reimpute.",
    ),
    SoundnessRow(
        "6.3-impute-fan-simulation-mix",
        "§6.3",
        "conditional",
        "impute-fan × simulation mixed draw clouds: default reduce per branch to (est, SE) → Rubin; "
        "pooling='mix' (Zhou–Reiter 2010) only at M ≥ 50 (§6.7) — the mixture misses Rubin's (1+1/M) inflation, "
        "~10% variance understatement at M=5; refused below the gate with the arithmetic shown.",
    ),
    SoundnessRow(
        "6.3-mi-match-sound",
        "§6.3",
        "sound",
        "mi ∘ match (re-match per imputation) is sound (Mitra–Reiter 2016; Leyrat et al. 2019); "
        "the 'across' variant (average PS, match once) is a different estimator, not offered.",
    ),
    SoundnessRow(
        "6.3-mi-survey",
        "§6.3",
        "conditional",
        "mi × survey design requires the imputation model to include design variables (Reiter–Raghunathan–Kinney 2006); "
        "else warn that strata/PSU/weights must be in the imputer features.",
    ),
    # §6.4 Estimated nuisances, weights, and overlap
    SoundnessRow(
        "6.4-ipw-aipw-stacked-psi",
        "§6.4",
        "sound",
        "IPW/AIPW × stacked-ψ delta (parametric nuisances) is sound: stacked M-estimation sandwich "
        "(Stefanski–Boos 2002; Lunceford–Davidian 2004 give AIPW's explicitly).",
    ),
    SoundnessRow(
        "6.4-aipw-bootstrap",
        "§6.4",
        "sound",
        "AIPW × bootstrap is sound; re-execution absorbs the coupling; the IF face is dormant (benign collision, §2.3).",
    ),
    SoundnessRow(
        "6.4-ipw-aipw-ml-nuisances-delta",
        "§6.4",
        "refuse",
        "IPW/AIPW with ML nuisances under delta is refused; orthogonal scores + cross-fitting required (Chernozhukov et al. 2018). "
        "steps.crossfit is L3 frontier *(future)*.",
    ),
    SoundnessRow(
        "6.4-ipw-aipw-ml-nuisances-bootstrap",
        "§6.4",
        "warn",
        "IPW/AIPW with ML nuisances under bootstrap warns; overfitting bias survives the bootstrap. "
        "Orthogonal scores + cross-fitting required (Chernozhukov et al. 2018). steps.crossfit is L3 frontier *(future)*.",
    ),
    SoundnessRow(
        "6.4-weak-overlap",
        "§6.4",
        "warn",
        "Weak overlap / positivity: ESS = (Σw)²/Σw², max normalized weight, and PS tail mass trip → warn. "
        "Steers: declared trimming (Crump et al. 2009 — re-executed per replicate/branch, estimand relabeled "
        "'trimmed population') or overlap weights/ATO (Li–Morgan–Zaslavsky 2018).",
    ),
    SoundnessRow(
        "6.4-population-without-support",
        "§6.4",
        "warn",
        "g-computation at= population without support warns for model extrapolation.",
    ),
    # §6.5 Dependence and view mechanics
    SoundnessRow(
        "6.5-survey-bootstrap",
        "§6.5",
        "sound",
        "survey × bootstrap is sound iff the design drives resampling (PSU-within-stratum with rescaling, Rao–Wu 1988) — "
        "already enforced (survey.py).",
    ),
    SoundnessRow(
        "6.5-lonely-psu",
        "§6.5",
        "refuse",
        "Stratum has only one PSU. Collapse this stratum or declare a certainty PSU.",
        predicate="pymargins._soundness._predicates.check_lonely_psu",
    ),
    SoundnessRow(
        "6.5-survey-matching",
        "§6.5",
        "warn",
        "survey × matching warns; conventions unsettled (DuGoff–Schuler–Stuart 2014; Austin et al. 2018): "
        "weighted PS model vs weights-as-covariate, survey-weighted matched analysis — "
        "require an explicit convention argument; open item.",
    ),
    SoundnessRow(
        "6.5-few-clusters",
        "§6.5",
        "warn",
        "Cluster-robust inference with G < 30: consider t with G−1 df or wild cluster bootstrap *(future)*.",
        predicate="pymargins._soundness._predicates.check_cluster_count",
    ),
    SoundnessRow(
        "6.5-block-bootstrap-no-length",
        "§6.5",
        "conditional",
        "Block bootstrap without a block length uses auto-selection (Politis–White 2004), record in the plan; "
        "~n^{1/3} fallback (Hall–Horowitz–Jing 1995).",
    ),
    SoundnessRow(
        "6.5-bca-cluster-block",
        "§6.5",
        "conditional",
        "BCa × cluster/block resampling requires the matching jackknife unit (cluster jackknife) for sound acceleration; "
        "else silently unsound.",
    ),
    SoundnessRow(
        "6.5-bca-small-b",
        "§6.5",
        "note",
        "BCa × small B (B < 1999): acceleration/tail quantiles unstable (Efron–Tibshirani; Davison–Hinkley; §6.7).",
        predicate="pymargins._soundness._predicates.check_tail_count_adequacy",
    ),
    SoundnessRow(
        "6.5-boundary-estimands",
        "§6.5",
        "warn",
        "Boundary estimands (p̂ ≈ 0/1, variance components ≈ 0) warn under all methods; κ's documented blind spot; "
        "delta is nonstandard, sim draws violate support, and the bootstrap is inconsistent at a boundary (Andrews 2000). "
        "No clean steer; report prominently.",
    ),
    SoundnessRow(
        "6.5-ratio-small-denominator",
        "§6.5",
        "warn",
        "Ratio estimands with small denominator |t| warn; κ trips (= 2/|t|, §5.2). Steer simulation; "
        "Fieller-type interval as a future ci= option *(future)*.",
    ),
    # §6.6 Quantitative predicates roster (no predicates here; implemented elsewhere)
    SoundnessRow(
        "6.6-kappa",
        "§6.6",
        "note",
        "κ (Skovgaard relative curvature) is reported per query; never steers silently.",
    ),
    SoundnessRow(
        "6.6-delta-sim-disagreement",
        "§6.6",
        "note",
        "delta_simulation_disagreement is reported as a cross-check on κ; > 5% relative CI-endpoint disagreement warns.",
    ),
    SoundnessRow(
        "6.6-ess",
        "§6.6",
        "note",
        "ESS / weight concentration is reported; ESS/n < 0.5 triggers a note (Kish 1965).",
        predicate="pymargins._soundness._predicates.check_ess",
    ),
    SoundnessRow(
        "6.6-fd-hessian-agreement",
        "§6.6",
        "note",
        "FD-Hessian agreement (tier-2 κ trustworthiness, §11.8).",
    ),
    # §6.7 Bootstrap calibration / tail checks
    SoundnessRow(
        "6.7-boundary-proximity",
        "§6.7",
        "warn",
        "Boundary proximity: estimand within z_level·SE of the parameter-space boundary → warn; "
        "any scenario p̂ within 0.01 of {0,1} → note.",
    ),
    SoundnessRow(
        "6.7-stabilized-weight",
        "§6.7",
        "warn",
        "Stabilized-weight diagnostics: max w̃ > 20 → warn (positivity); "
        "mean w̃ ∉ [0.9, 1.1] → note (misspecification) (Cole–Hernán 2008).",
    ),
    SoundnessRow(
        "6.7-tail-counts",
        "§6.7",
        "note",
        "Bootstrap tail counts are checked against TAIL_COUNT_NOTE/WARN thresholds; "
        "percentile/basic/BCa tails require enough replicates.",
        predicate="pymargins._soundness._predicates.check_tail_count_adequacy",
    ),
    SoundnessRow(
        "6.7-se-b",
        "§6.7",
        "note",
        "ci='se' (normal-approx SE-only interval) requires B >= SE_ONLY_MIN_B; "
        "otherwise the bootstrap SE estimate itself is too noisy.",
        predicate="pymargins._soundness._predicates.check_tail_count_adequacy",
    ),
    SoundnessRow(
        "6.7-replicate-failures",
        "§6.7",
        "note",
        "Replicate failure rate is always reported; > 1% note, > 5% warn.",
        predicate="pymargins._engine._execute.execute_query",
    ),
)
