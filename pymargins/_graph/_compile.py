"""Compile pipeline: C1 structural + C2 data compile.

Implements the compiler from design §3 and req. §3.
"""

from __future__ import annotations

from typing import Any

from pymargins._adapter import ModelAdapter
from pymargins._adapters import auto_detect_adapter
from pymargins._soundness._predicates import (
    CompileError,
    CompileReport,
    Severity,
    check_ci_method_compatibility,
    check_cluster_count,
    check_lonely_psu,
    check_method_adapter_compatibility,
    check_tail_count_adequacy,
)

from ._plan import Plan


class _DataFingerprintAdapter(ModelAdapter):
    """Minimal concrete adapter used only to fingerprint a DataFrame."""

    def __init__(self, data):
        self._data = data

    @property
    def training_data(self):
        return self._data

    def coefficients(self):
        raise NotImplementedError

    def covariance(self, vcov_spec=None):
        raise NotImplementedError

    def predict(self, beta, X, offset=None):
        raise NotImplementedError

    def design_matrix_from_df(self, df):
        raise NotImplementedError

    def column_index_of_variable(self, name):
        raise NotImplementedError

    def variable_metadata(self):
        raise NotImplementedError

    @property
    def supports_jax_autodiff(self):
        return False

    @property
    def supported_inference_methods(self):
        return set()

    @property
    def gradient_backend_recommendation(self):
        return "fd"


def compile(
    wiring,
    outcome: Any,
    *,
    at: str = "overall",
    scale: str = "response",
    method: str = "delta",
    vcov: Any | None = None,
    ci: str = "wald",
    level: float = 0.95,
    B: int = 0,
    n_sim: int = 0,
    seed: int | None = None,
    kappa_threshold: float | None = None,
    cluster: Any | None = None,
    block_size: int | None = None,
    mode: str = "doctrine",
    **extra,
) -> tuple[Plan, CompileReport]:
    """Compile a wiring graph + outcome into an immutable Plan.

    Parameters
    ----------
    wiring : Node
        The prepared-data graph.
    outcome : ModelAdapter | fitted model
        The outcome specification or a fitted model used as template.
    mode : {"doctrine", "legacy"}
        Doctrine mode applies refusals; legacy mode skips them.

    Returns
    -------
    plan : Plan
    report : CompileReport
    """
    report = CompileReport()

    # Walk graph
    node_kinds = []
    node_hashes = []
    population_notes = []
    design = None
    _walk_graph(wiring, node_kinds, node_hashes, population_notes)
    # Extract design from input node params
    for node in _flatten_graph(wiring):
        if node.kind == "input":
            for k, v in node.params:
                if k == "design" and v is not None and v is not True:
                    design = v
            break

    # Resolve outcome to adapter
    adapter, data_fp, unhashable, report = _resolve_outcome(wiring, outcome, report)

    # Auto-resolve method
    supported = adapter.supported_inference_methods
    method_resolved = method
    resolution_reason = "user-specified"
    if method == "auto":
        if "delta" in supported:
            method_resolved = "delta"
            resolution_reason = "auto: delta supported"
        elif "simulation" in supported:
            method_resolved = "simulation"
            resolution_reason = "auto: simulation fallback"
        elif "bootstrap" in supported:
            method_resolved = "bootstrap"
            resolution_reason = "auto: bootstrap fallback"
        else:
            report = report.append(
                Severity.REFUSE, "auto_method_fail",
                "method='auto' could not resolve: no supported methods.",
            )
            report.raise_for_refusals()

    # Method / ci compatibility checks (use resolved method)
    report = check_method_adapter_compatibility(method_resolved, supported, report)
    report = check_ci_method_compatibility(ci, method_resolved, report)
    if method_resolved == "bootstrap" and B > 0:
        report = check_tail_count_adequacy(B, level, ci, report)
    n_clusters = None
    if cluster is not None:
        import numpy as np
        n_clusters = len(np.unique(np.asarray(cluster)))
    report = check_cluster_count(n_clusters, report)
    report = check_lonely_psu(design, report)
    report.raise_for_refusals()

    plan = Plan(
        node_kinds=tuple(node_kinds),
        node_hashes=tuple(node_hashes),
        at=at,
        scale=scale,
        method_declared=method,
        method_resolved=method_resolved,
        method_resolution_reason=resolution_reason,
        vcov=vcov,
        ci=ci,
        level=level,
        B=B,
        n_sim=n_sim,
        seed=seed,
        data_fingerprint=data_fp,
        unhashable_callable=unhashable,
        population_note="; ".join(n for n in population_notes if n) or None,
    )
    report.emit_warnings()
    return plan, report


def _flatten_graph(node):
    """Yield all nodes in topological order."""
    seen = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        for inp in n.inputs:
            stack.append(inp)
        yield n


def _walk_graph(node, kinds, hashes, pop_notes):
    """Record node metadata in topological order."""
    for inp in node.inputs:
        _walk_graph(inp, kinds, hashes, pop_notes)
    kinds.append(node.kind)
    hashes.append(node.hash)
    if node.population_note:
        pop_notes.append(node.population_note)


def _resolve_outcome(wiring, outcome, report):
    """Return (adapter, data_fingerprint, unhashable_callable, report)."""
    if isinstance(outcome, ModelAdapter):
        adapter = outcome
        data_fp = adapter.data_fingerprint()
        # Template consistency check: wiring output fingerprint vs template data fingerprint
        try:
            wiring_data = wiring.collect()
            wiring_fp = _DataFingerprintAdapter(wiring_data).data_fingerprint()
        except NotImplementedError:
            wiring_fp = None
        if wiring_fp is not None and wiring_fp != data_fp:
            report = report.append(
                Severity.REFUSE,
                "template_mismatch",
                f"Template training data fingerprint ({data_fp[:16]}...) does not match "
                f"wiring output fingerprint ({wiring_fp[:16]}...).",
            )
        return adapter, data_fp, False, report

    # Assume fitted model object
    try:
        adapter = auto_detect_adapter(outcome)
    except Exception as exc:
        raise CompileError(f"Could not auto-detect adapter for outcome: {exc}") from exc

    data_fp = adapter.data_fingerprint()
    # Template consistency check
    try:
        wiring_data = wiring.collect()
        wiring_fp = _DataFingerprintAdapter(wiring_data).data_fingerprint()
    except NotImplementedError:
        wiring_fp = None
    if wiring_fp is not None and wiring_fp != data_fp:
        report = report.append(
            Severity.REFUSE,
            "template_mismatch",
            f"Template training data fingerprint ({data_fp[:16]}...) does not match "
            f"wiring output fingerprint ({wiring_fp[:16]}...).",
        )
    return adapter, data_fp, False, report
