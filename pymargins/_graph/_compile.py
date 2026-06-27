"""Compile pipeline: C1 structural + C2 data compile.

Design \u00a74.3/\u00a74.5/\u00a75.2, req \u00a73-\u00a74. Added in 0.4.0 (R5).
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from pymargins._adapter import ModelAdapter
from pymargins._adapters import auto_detect_adapter
from pymargins._engine._queries import (
    QueryContext,
    QuerySpec,
    WiringFacts,
    compile_query,
    resolve_scale,
)
from pymargins._estimands import is_jax_differentiable
from pymargins._gradients import gradient
from pymargins._graph._node import Node, _fingerprint
from pymargins._kappa import delta_simulation_disagreement, session_kappa
from pymargins._soundness._constants import (
    DISAGREEMENT_WARN,
    KAPPA_BORDERLINE,
)
from pymargins._soundness._predicates import (
    CompileError,
    CompileReport,
    Severity,
    check_ci_method_compatibility,
    check_cluster_count,
    check_ess,
    check_lonely_psu,
    check_method_adapter_compatibility,
    check_tail_count_adequacy,
)
from pymargins._tabular import fingerprint_frame, to_pandas_if_needed

from ._plan import Plan


@dataclass(frozen=True)
class Compiled:
    """Everything the noun needs beyond the Plan.

    This is an internal object returned by ``compile()`` so that the
    estimator/facade can construct ``QueryContext`` and call the engine
    without re-deriving the adapter, base data, frozen \u03a3\u0302, or scale
    callables.
    """

    adapter: ModelAdapter
    wiring_facts: WiringFacts
    base_data: Any
    frozen_cov: np.ndarray
    phi: Any | None
    phi_inv: Any | None
    weights: np.ndarray | None = None
    at: Any = "overall"


_KNOWN_NODE_KINDS = frozenset({"input", "match", "trim", "drop_outliers", "reimpute"})
_ROW_FILTER_KINDS = frozenset({"trim", "drop_outliers"})


def _fingerprint_scale(scale: Any) -> tuple[Any, bool]:
    """Return a Plan-safe scale value and an unhashable-callable flag.

    Named scales pass through as strings.  Explicit callable pairs are
    fingerprinted by source (or qualname) so the Plan stays JSON-serializable
    and the hash is sensitive to the actual transformation.
    """
    if isinstance(scale, str):
        return scale, False
    if isinstance(scale, tuple) and len(scale) == 2:
        parts: list[str] = []
        unhashable = False
        for fn in scale:
            try:
                src = inspect.getsource(fn)
                parts.append(hashlib.sha256(src.encode("utf-8")).hexdigest()[:16])
            except (OSError, TypeError):
                qual = getattr(fn, "__qualname__", None) or getattr(
                    fn, "__name__", None
                )
                if qual:
                    parts.append(qual)
                else:
                    parts.append("unhashable_callable")
                    unhashable = True
        return tuple(parts), unhashable
    return str(scale), False


def _fingerprint_callable(fn: Any) -> tuple[str, bool]:
    """Fingerprint a single callable, returning (value, unhashable)."""
    try:
        src = inspect.getsource(fn)
        return f"callable:{hashlib.sha256(src.encode('utf-8')).hexdigest()[:16]}", False
    except (OSError, TypeError):
        qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
        if qual:
            return f"callable:{qual}", False
        return "callable:unhashable_callable", True


def _fingerprint_vcov(vcov: Any) -> Any:
    """Replace a user-supplied covariance ndarray with a stable fingerprint.

    The Plan must remain JSON-serializable and small; a raw Σ̂ array breaks
    both.  Strings and dict specs pass through unchanged.
    """
    if isinstance(vcov, np.ndarray):
        arr = np.asarray(vcov, dtype=float)
        fp = hashlib.sha256(arr.tobytes()).hexdigest()
        return {"kind": "user_ndarray", "fingerprint": fp}
    return vcov


def _json_default_for_at(obj: Any, unhashable_flag: list[bool]) -> Any:
    """JSON-default for non-serializable ``at`` values (arrays, callables, etc.).

    Sets ``unhashable_flag[0] = True`` when a callable cannot be fingerprinted
    by source or qualname, so ``_fingerprint_at`` can propagate the honesty flag
    even for callables nested inside dicts/lists.
    """
    if callable(obj):
        val, unhashable = _fingerprint_callable(obj)
        if unhashable:
            unhashable_flag[0] = True
        return val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return repr(obj)


def _fingerprint_at(at: Any) -> tuple[Any, bool]:
    """Return a Plan-safe ``at`` value and an unhashable-callable flag.

    Strings pass through. Dicts and other objects are serialized to a stable
    JSON string; any non-JSON values inside are fingerprinted rather than
    letting ``Plan.__post_init__`` crash with a cryptic TypeError. Callables are
    source-/qualname-hashed, and the unhashable case sets the honesty flag.
    """
    if isinstance(at, str):
        return at, False
    if callable(at):
        return _fingerprint_callable(at)

    try:
        return json.dumps(at, sort_keys=True), False
    except TypeError:
        flag = [False]
        return (
            json.dumps(
                at, sort_keys=True, default=lambda o: _json_default_for_at(o, flag)
            ),
            flag[0],
        )
    except Exception:
        return repr(at), True


def _walk_graph_topological(node: Node) -> list[Node]:
    """Return nodes in topological order (inputs before dependents)."""
    out: list[Node] = []
    seen: set[int] = set()

    def visit(n: Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        for inp in n.inputs:
            visit(inp)
        out.append(n)

    visit(node)
    return out


def _check_graph_structure(nodes: list[Node]) -> None:
    """C1 structural checks: unknown kinds, fans, match + row-filter."""
    has_match = False
    has_filter = False
    for n in nodes:
        if n.kind not in _KNOWN_NODE_KINDS:
            raise CompileError(
                f"Unknown node kind: {n.kind!r}. "
                f"Supported kinds: {sorted(_KNOWN_NODE_KINDS)}."
            )
        if n.fan is not None:
            raise CompileError(
                f"Fan node kind={n.fan!r} lands with the fan engine in 0.5.0/0.6.0."
            )
        if n.kind == "match":
            has_match = True
        if n.kind in _ROW_FILTER_KINDS:
            has_filter = True
    if has_match and has_filter:
        raise CompileError(
            "match + row-filter stages in one wiring lands with the fan engine in "
            "0.5.0; today, apply filters before matching outside the wiring or use "
            "matching alone."
        )


def _extract_wiring_facts(nodes: list[Node]) -> WiringFacts:
    """Collect dependence and pipeline facts from a topologically-ordered walk.

    Stage order follows the wiring edges from the input outward, not the
    accidental stack order of ``_flatten_graph``.
    """
    design = None
    cluster = None
    block = None
    block_type = None
    matching = None
    transforms: list[Any] = []
    pop_notes: list[str] = []

    for n in nodes:
        if n.kind == "input":
            for k, v in n.params:
                if k == "design" and v is not None and v is not True:
                    design = v
                elif k == "cluster" and v is not None and v is not True:
                    cluster = v
                elif k == "block" and v is not None:
                    block = v
                elif k == "block_type" and v is not None:
                    block_type = v
        elif n.kind == "match":
            matching = n._payload
            if n.population_note:
                pop_notes.append(n.population_note)
        elif n.kind in _ROW_FILTER_KINDS:
            transforms.append(n._payload)
            if n.population_note:
                pop_notes.append(n.population_note)
        elif n.kind == "reimpute":
            transforms.append(n._payload)

    return WiringFacts(
        design=design,
        cluster=cluster,
        block=block,
        block_type=block_type or "moving",
        matching=matching,
        transforms=transforms or None,
        population_note="; ".join(pop_notes) or None,
    )


def _resolve_vcov_spec(vcov: Any, design: Any | None, cluster: Any | None) -> Any:
    """Replicate the legacy vcov_spec resolution rule (G1.3).

    Explicit ``vcov=`` wins.  A survey design plus an explicit non-survey
    ``vcov=`` is a conflict, because the design already determines \u03a3\u0302.
    """
    if design is not None and vcov is not None:
        is_survey_spec = isinstance(vcov, dict) and vcov.get("type") == "survey"
        if not is_survey_spec:
            raise CompileError(
                "Explicit vcov= conflicts with the survey design declared at "
                "steps.input(). The design already determines \u03a3\u0302."
            )
    if isinstance(vcov, str) and vcov.lower() == "cluster":
        if cluster is None:
            raise CompileError(
                'vcov="cluster" requires a cluster variable declared at steps.input(cluster=...).'
            )
        return {"type": "cluster", "groups": cluster}
    if vcov is not None:
        return vcov
    if cluster is not None:
        return {"type": "cluster", "groups": cluster}
    if design is not None:
        return {"type": "survey", "design": design}
    return None


def _representative_design(
    adapter: ModelAdapter,
    base_data: Any,
    seed: int | None,
    n_samples: int = 50,
) -> list[Any]:
    """Sample rows from the base data as design-matrix rows."""
    rng = np.random.default_rng(seed)
    if hasattr(base_data, "iloc"):
        n = len(base_data)
        idx = rng.choice(n, size=min(n_samples, n), replace=False)
        return [adapter.design_matrix_from_df(base_data.iloc[[i]])[0] for i in idx]
    X = adapter.design_matrix_from_df(to_pandas_if_needed(base_data))
    n = X.shape[0]
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    return [X[i] for i in idx]


def _posture_h_factory(
    adapter: ModelAdapter,
    x_row: Any,
    phi_inv: Any | None,
):
    """Build a scalar prediction estimand on a single design row."""
    x_arr = jnp.atleast_2d(x_row)

    def h(beta):
        mu = adapter.predict(beta, x_arr)[0]
        return phi_inv(mu) if phi_inv is not None else mu

    return h


def _resolve_method_auto(
    adapter: ModelAdapter,
    posture: Any,  # CompiledQuery
    beta: jnp.ndarray,
    frozen_cov: np.ndarray,
    phi: Any | None,
    phi_inv: Any | None,
    base_data: Any,
    gradient_backend: str,
    fd_step: float,
    level: float,
    n_sim: int,
    seed: int | None,
    supported: set[str],
) -> tuple[str | None, str]:
    """Decide-once method resolution for ``method='auto'``.

    Tier-1 (autodiff) adapters get a \u03ba pre-pass; tier-2 (FD-wrapped)
    adapters get a delta-vs-simulation disagreement check.  Bootstrap is
    never auto-resolved.
    """
    h = posture.h
    if not is_jax_differentiable(h, beta):
        if "simulation" in supported:
            return "simulation", "auto: estimand not JAX-differentiable"
        return (
            None,
            "auto: estimand not JAX-differentiable and simulation not supported",
        )

    if adapter.supports_jax_autodiff:
        rep_design = _representative_design(adapter, base_data, seed)
        diag = session_kappa(
            lambda x: _posture_h_factory(adapter, x, phi_inv),
            beta,
            frozen_cov,
            rep_design,
            backend=gradient_backend,
            fd_step=fd_step,
            norm="spectral",
        )
        max_k = float(diag["max"])
        if max_k <= KAPPA_BORDERLINE:
            if "delta" in supported:
                return (
                    "delta",
                    f"auto: posture \u03ba={max_k:.3f} \u2264 {KAPPA_BORDERLINE}",
                )
            if "simulation" in supported:
                return (
                    "simulation",
                    f"auto: delta unavailable, posture \u03ba={max_k:.3f}",
                )
        if "simulation" in supported:
            return (
                "simulation",
                f"auto: posture \u03ba={max_k:.3f} > {KAPPA_BORDERLINE}",
            )
        if "delta" in supported:
            return (
                "delta",
                f"auto: posture \u03ba={max_k:.3f} but simulation not supported",
            )
        return None, "auto: could not resolve a supported method"

    # Tier-2: FD-wrapped adapters use delta-vs-sim disagreement.
    estimate = h(beta)
    grad = gradient(h, beta, backend=gradient_backend, fd_step=fd_step)
    disagreement = delta_simulation_disagreement(
        estimate,
        grad,
        frozen_cov,
        h,
        beta,
        level=level,
        n_sim=n_sim,
        rng_seed=seed,
        phi=phi,
    )
    if disagreement <= DISAGREEMENT_WARN:
        if "delta" in supported:
            return (
                "delta",
                f"auto: tier-2 disagreement={disagreement:.1%} \u2264 {DISAGREEMENT_WARN:.0%}",
            )
    if "simulation" in supported:
        return (
            "simulation",
            f"auto: tier-2 disagreement={disagreement:.1%} > {DISAGREEMENT_WARN:.0%}",
        )
    if "delta" in supported:
        return (
            "delta",
            f"auto: tier-2 disagreement={disagreement:.1%} but simulation not supported",
        )
    return None, "auto: could not resolve a supported method"


def _validate_compile_inputs(
    *,
    method: str,
    at: Any,
    scale: Any,
    ci: str | None,
    constants_overrides: tuple[tuple[str, Any], ...],
) -> None:
    """Compile-time validation of user-facing scalar arguments (step 1)."""
    if method not in {"delta", "simulation", "bootstrap", "auto"}:
        raise CompileError(
            f'method="{method}" is not valid. Supported: delta, simulation, bootstrap, auto.'
        )

    if isinstance(at, str):
        allowed = {"overall", "mean", "median", "mode", "typical", "min", "max"}
        is_percentile = at.startswith("p") and at[1:].isdigit()
        if at not in allowed and not is_percentile:
            raise CompileError(
                f'at="{at}" is not valid. Supported strings: overall, mean, median, mode, typical, min, max; '
                f'percentile strings like "p25"; or pass a dict/callable.'
            )
    elif not isinstance(at, dict) and not callable(at):
        raise CompileError(
            f"at= must be a string, dict, or callable, got {type(at).__name__}."
        )

    if isinstance(scale, str):
        if scale not in {"response", "identity", "log", "logit", "probit"}:
            raise CompileError(
                f'scale="{scale}" is not valid. Supported named scales: '
                f"response, identity, log, logit, probit; or pass a (phi, phi_inv) pair."
            )
    elif not (isinstance(scale, tuple) and len(scale) == 2):
        raise CompileError(
            f"scale= must be a named scale string or a (phi, phi_inv) callable pair, "
            f"got {scale!r}."
        )

    if ci is not None and ci != "":
        if ci not in {"wald", "percentile", "basic", "bca", "studentized", "se"}:
            raise CompileError(
                f'ci="{ci}" is not valid. Supported: wald, percentile, basic, bca, studentized, se.'
            )

    if constants_overrides:
        raise CompileError(
            "constants_overrides is not supported in 0.4.0; pass an empty tuple. "
            "Per-analysis constant overrides will be wired in a future release."
        )


def compile(
    wiring: Node,
    outcome: Any,
    *,
    at: str = "overall",
    scale: Any = "response",
    method: str = "delta",
    vcov: Any | None = None,
    ci: str | None = None,
    level: float = 0.95,
    B: int = 1000,
    n_sim: int = 4000,
    seed: int | None = None,
    weights: Any | None = None,
    gradient_backend: str = "autodiff",
    fd_step: float = 1e-6,
    constants_overrides: tuple[tuple[str, Any], ...] = (),
) -> tuple[Plan, CompileReport, Compiled]:
    """Compile a wiring graph + outcome into an immutable Plan.

    Returns
    -------
    plan : Plan
    report : CompileReport
    compiled : Compiled
        Internal object carrying the adapter, wiring facts, point-executed
        base data, frozen \u03a3\u0302, and scale callables.
    """
    report = CompileReport()

    # Validate inference-budget invariants (ported from the legacy session).
    if not isinstance(n_sim, int) or n_sim < 1:
        raise CompileError(f"n_sim must be a positive integer, got {n_sim!r}.")
    if not isinstance(B, int) or B < 1:
        raise CompileError(
            f"B (bootstrap replicates) must be a positive integer, got {B!r}."
        )

    # Step 1: validate user-facing scalar arguments.
    _validate_compile_inputs(
        method=method,
        at=at,
        scale=scale,
        ci=ci,
        constants_overrides=constants_overrides,
    )

    # C1: walk the graph in topological order and run structural checks.
    nodes = _walk_graph_topological(wiring)
    _check_graph_structure(nodes)
    wiring_facts = _extract_wiring_facts(nodes)

    # C2: point-execute the wiring and fingerprint the output.
    try:
        base_data = wiring.collect()
    except Exception as exc:
        raise CompileError(f"Could not collect the wiring output: {exc}") from exc
    wiring_fp = fingerprint_frame(base_data)

    # Resolve outcome to an adapter.
    if isinstance(outcome, ModelAdapter):
        adapter = outcome
    else:
        try:
            adapter = auto_detect_adapter(outcome)
        except Exception as exc:
            raise CompileError(
                f"Could not auto-detect adapter for outcome: {exc}"
            ) from exc

    data_fp = adapter.data_fingerprint()
    if wiring_fp != data_fp:
        report = report.append(
            Severity.REFUSE,
            "template_mismatch",
            f"Template training data fingerprint ({data_fp[:16]}...) does not match "
            f"wiring output fingerprint ({wiring_fp[:16]}...).",
        )
    # Fail early on a template mismatch so heavy work runs on valid data.
    report.raise_for_refusals()

    # Resolve vcov_spec and freeze \u03a3\u0302 exactly once per estimator.
    vcov_spec = _resolve_vcov_spec(vcov, wiring_facts.design, wiring_facts.cluster)
    frozen_cov = np.asarray(adapter.covariance(vcov_spec))

    # Resolve scale and fingerprint it for the Plan.
    phi, phi_inv = resolve_scale(scale)
    scale_fingerprint, unhashable_callable = _fingerprint_scale(scale)

    # Weights normalization and fingerprint.
    weights_arr = None
    weights_fp = None
    if weights is not None:
        weights_arr = np.asarray(weights, dtype=float)
        weights_fp = _fingerprint(weights_arr)

    # Method resolution.
    supported = adapter.supported_inference_methods
    method_resolved = method
    resolution_reason = "user-specified"
    if method == "auto":
        # Build the analytical posture estimand only when needed for auto.
        try:
            posture_ctx = QueryContext(
                adapter=adapter,
                base_data=base_data,
                at=at,
                weights=weights_arr,
                phi=phi,
                phi_inv=phi_inv,
                fd_step=fd_step,
                gradient_backend=gradient_backend,
            )
            posture = compile_query(QuerySpec(kind="predict"), posture_ctx)
        except (ValueError, TypeError) as exc:
            raise CompileError(f"Could not build posture estimand: {exc}") from exc
        beta = adapter.coefficients()
        method_resolved, resolution_reason = _resolve_method_auto(
            adapter,
            posture,
            beta,
            frozen_cov,
            phi,
            phi_inv,
            base_data,
            gradient_backend,
            fd_step,
            level,
            n_sim,
            seed,
            supported,
        )
        if method_resolved is None:
            report = report.append(
                Severity.REFUSE,
                "auto_method_fail",
                f"method='auto' could not resolve: {resolution_reason}.",
            )

    # Method / adapter compatibility.
    report = check_method_adapter_compatibility(method_resolved, supported, report)

    # Bootstrap-only transform stages (e.g. reimpute) require bootstrap inference.
    if wiring_facts.transforms:
        for stage in wiring_facts.transforms:
            if (
                getattr(stage, "requires_resampling", False)
                and method_resolved != "bootstrap"
            ):
                report = report.append(
                    Severity.REFUSE,
                    "method_unsupported",
                    f"method='{method_resolved}' is not compatible with bootstrap-only "
                    "transform stages; use method='bootstrap'.",
                )
                break

    # CI defaults and compatibility.
    if ci is None or ci == "":
        ci = "percentile" if method_resolved == "bootstrap" else "wald"
    report = check_ci_method_compatibility(ci, method_resolved, report)
    # Honesty: bootstrap never produces a Wald interval.
    if method_resolved == "bootstrap" and ci == "wald":
        ci = "percentile"

    # Adequacy predicates.
    if method_resolved == "bootstrap":
        report = check_tail_count_adequacy(B, level, ci, report)
    n_clusters = None
    if wiring_facts.cluster is not None:
        n_clusters = len(np.unique(np.asarray(wiring_facts.cluster)))
    report = check_cluster_count(n_clusters, report)
    report = check_lonely_psu(wiring_facts.design, report)
    report = check_ess(weights_arr, report)

    report.raise_for_refusals()

    # Build the Plan.
    node_kinds = tuple(n.kind for n in nodes)
    node_hashes = tuple(n.hash for n in nodes)
    population_note = "; ".join(n for n in [wiring_facts.population_note] if n) or None

    at_fingerprint, at_unhashable = _fingerprint_at(at)
    unhashable_callable = unhashable_callable or at_unhashable

    plan = Plan(
        node_kinds=node_kinds,
        node_hashes=node_hashes,
        at=at_fingerprint,
        scale=scale_fingerprint,
        method_declared=method,
        method_resolved=method_resolved,
        method_resolution_reason=resolution_reason,
        vcov=_fingerprint_vcov(vcov),
        ci=ci,
        level=level,
        B=B,
        n_sim=n_sim,
        seed=seed,
        gradient_backend=gradient_backend,
        fd_step=fd_step,
        data_fingerprint=data_fp,
        weights_fingerprint=weights_fp,
        unhashable_callable=unhashable_callable,
        population_note=population_note,
        constants_overrides=constants_overrides,
    )

    compiled = Compiled(
        adapter=adapter,
        wiring_facts=wiring_facts,
        base_data=base_data,
        frozen_cov=frozen_cov,
        phi=phi,
        phi_inv=phi_inv,
        weights=weights_arr,
        at=at,
    )

    report.emit_warnings()
    return plan, report, compiled
