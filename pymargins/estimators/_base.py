"""Base estimator noun and GComputation.

Implements the blessed compiler from design §4.2 and req. §4.
"""

from __future__ import annotations

from typing import Any

from pymargins._adapter import ModelAdapter
from pymargins._adapters import auto_detect_adapter
from pymargins._graph._compile import CompileError, compile
from pymargins._graph._node import Node
from pymargins._graph._plan import Plan
from pymargins._result._graphresult import GraphResult
from pymargins.margins import Margins


def _scale_to_phi(scale: str | None):
    """Return (phi, phi_inv) for a named scale."""
    import jax.numpy as jnp

    if scale is None or scale == "response" or scale == "identity":
        return None, None
    if scale == "log":
        return jnp.exp, jnp.log
    if scale == "logit":
        from jax.scipy.special import expit, logit

        return expit, logit
    if scale == "probit":
        from jax.scipy.special import ndtr, ndtri

        return ndtr, ndtri
    raise CompileError(f"Unknown scale: {scale!r}. Supported: response, log, logit, probit.")


def _extract_legacy_kwargs(wiring: Node, plan: Plan):
    """Translate a wiring graph into the kwargs Margins() expects."""
    kwargs = {}

    # Walk graph to extract stages, matching, and input properties
    transforms = []
    matching = None
    survey_design = None
    cluster = None
    block_size = None

    for node in _flatten_graph(wiring):
        if node.kind == "input":
            for k, v in node.params:
                if k == "cluster" and v is not None and v is not True:
                    cluster = v
                if k == "block" and v is not None:
                    block_size = v
                if k == "design" and v is not None and v is not True:
                    survey_design = v
        elif node.kind == "match":
            matching = node._payload
        elif node.kind in ("trim", "drop_outliers", "reimpute"):
            stage = node._payload
            transforms.append(stage)

    if transforms:
        # _flatten_graph yields reverse-topological order; reverse to get
        # the user-specified left-to-right pipeline order.
        kwargs["transforms"] = list(reversed(transforms))
    if matching is not None:
        kwargs["matching"] = matching
    if survey_design is not None:
        kwargs["survey_design"] = survey_design
    if cluster is not None:
        kwargs["cluster"] = cluster
    if block_size is not None:
        kwargs["block_size"] = block_size

    # Scale
    phi, phi_inv = _scale_to_phi(plan.scale)
    if phi is not None:
        kwargs["phi"] = phi
        kwargs["phi_inv"] = phi_inv

    # Inference params
    kwargs["method"] = plan.method_resolved
    kwargs["level"] = plan.level
    if plan.vcov is not None:
        kwargs["vcov"] = plan.vcov
    if plan.B > 0:
        kwargs["n_boot"] = plan.B
    if plan.n_sim > 0:
        kwargs["n_sim"] = plan.n_sim
    if plan.seed is not None:
        kwargs["rng_seed"] = plan.seed

    # Bootstrap config from ci
    if plan.ci not in ("", "wald") and plan.method_resolved == "bootstrap":
        kwargs["bootstrap_config"] = {"ci_method": plan.ci}

    return kwargs


def _flatten_graph(node: Node):
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


def _resolve_model(wiring: Node, outcome: Any, plan: Plan):
    """Return a model object suitable for Margins()."""
    if isinstance(outcome, ModelAdapter):
        return outcome.results if hasattr(outcome, "results") else outcome
    if isinstance(outcome, str):
        raise CompileError("Spec-form outcome not yet supported.")
    # Assume fitted model
    return outcome


class GComputation:
    """Estimator noun for g-computation (standardization).

    This is the v0.4.0 new-surface estimator.  It compiles a wiring graph
    into a Plan and delegates inference to the existing engine.
    """

    def __init__(
        self,
        wiring_or_model=None,
        *,
        outcome=None,
        at: str = "overall",
        scale: str = "response",
        method: str = "delta",
        vcov=None,
        ci: str = "wald",
        level: float = 0.95,
        B: int = 1000,
        n_sim: int = 4000,
        seed: int | None = None,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ):
        # Handle positional-model sugar: GComputation(model)
        if outcome is None and wiring_or_model is not None:
            if isinstance(wiring_or_model, Node):
                raise CompileError(
                    "GComputation requires an outcome= when passed a wiring node."
                )
            outcome = wiring_or_model
            wiring = None
        else:
            wiring = wiring_or_model

        if wiring is None:
            # Implicit input from outcome template
            if isinstance(outcome, ModelAdapter):
                data = outcome.training_data
            else:
                adapter = auto_detect_adapter(outcome)
                data = adapter.training_data
            wiring = Node(kind="input", _payload=data)

        if outcome is None:
            raise CompileError("outcome is required.")

        self._plan, self._report, _ = compile(
            wiring,
            outcome,
            at=at,
            scale=scale,
            method=method,
            vcov=vcov,
            ci=ci,
            level=level,
            B=B,
            n_sim=n_sim,
            seed=seed,
        )

        # Build internal Margins session from the plan
        legacy_kwargs = _extract_legacy_kwargs(wiring, self._plan)
        model = _resolve_model(wiring, outcome, self._plan)
        # Doctrine mode: disable runtime κ fallback (design §5.2)
        # Per-query κ is recorded on the result instead.
        self._session = Margins(
            model,
            at=at,
            **legacy_kwargs,
            kappa_threshold=float("inf"),
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
        self._wiring = wiring
        self._outcome = outcome

    @property
    def plan(self) -> Plan:
        return self._plan

    def predict(self, *, atexog=None, over=None, transform=None, label=None, outcome=None):
        result = self._session.predict(
            atexog=atexog, over=over, transform=transform, label=label, outcome=outcome
        )
        return GraphResult._from_margins_result(result, self._plan)

    def dydx(self, variables=None, *, atexog=None, over=None, transform=None, label=None, outcome=None):
        result = self._session.dydx(
            variables=variables, atexog=atexog, over=over, transform=transform, label=label, outcome=outcome
        )
        return GraphResult._from_margins_result(result, self._plan)

    def eyex(self, variable=None, **kwargs):
        result = self._session.eyex(variable, **kwargs)
        return GraphResult._from_margins_result(result, self._plan)

    def eydx(self, variable=None, **kwargs):
        result = self._session.eydx(variable, **kwargs)
        return GraphResult._from_margins_result(result, self._plan)

    def dyex(self, variable=None, **kwargs):
        result = self._session.dyex(variable, **kwargs)
        return GraphResult._from_margins_result(result, self._plan)

    def contrasts(self, *, scenarios=None, contrasts=None, outcome=None):
        result = self._session.contrasts(scenarios=scenarios, contrasts=contrasts, outcome=outcome)
        return GraphResult._from_margins_result(result, self._plan)

    def evaluate(self, *, scenarios=None, compose=None, outcome=None):
        result = self._session.evaluate(scenarios=scenarios, compose=compose, outcome=outcome)
        return GraphResult._from_margins_result(result, self._plan)

    def rmst(self, *, horizon=None, atexog=None, over=None, n_grid=80):
        result = self._session.rmst(horizon=horizon, atexog=atexog, over=over, n_grid=n_grid)
        return GraphResult._from_margins_result(result, self._plan)

    def joint(self, *results):
        raise NotImplementedError("joint() lands in 0.5.0")
