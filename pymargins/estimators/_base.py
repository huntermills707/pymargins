"""Base estimator noun and GComputation.

Implements the blessed compiler from design §4.2 and req §4.
R6: GComputation now sits directly on ``compile → Plan → BankSet → engine → GraphResult``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from pymargins._adapter import ModelAdapter
from pymargins._adapters import auto_detect_adapter
from pymargins._engine._banks import BankSet
from pymargins._engine._execute import execute_query
from pymargins._engine._queries import (
    QueryContext,
    QuerySpec,
    compile_query,
)
from pymargins._graph._compile import CompileError, compile
from pymargins._graph._node import Node
from pymargins._graph._plan import Plan
from pymargins._result._graphresult import GraphResult


def _resolve_outcome(
    wiring: Node | None,
    outcome: Any,
    adapter: ModelAdapter | None,
) -> ModelAdapter:
    """Turn the user's ``outcome=`` into a ``ModelAdapter``.

    Supported forms:
      - a :class:`ModelAdapter` instance
      - a fitted model object (auto-detected)
      - a formula string ``"y ~ x"`` (OLS on the wiring output)
      - a 2-tuple ``("y ~ x", family)`` (GLM on the wiring output)
      - ``None`` with an explicit ``adapter=`` (use the provided adapter)
    """
    if isinstance(outcome, ModelAdapter):
        return outcome
    if adapter is not None and outcome is None:
        return adapter

    if isinstance(outcome, str):
        formula = outcome
        family = None
    elif (
        isinstance(outcome, (tuple, list))
        and len(outcome) == 2
        and isinstance(outcome[0], str)
    ):
        formula, family = outcome
    else:
        # Fitted model object (or explicit adapter with a model outcome)
        if adapter is not None:
            return adapter
        try:
            return auto_detect_adapter(outcome)
        except Exception as exc:
            raise CompileError(
                f"Could not auto-detect adapter for outcome: {exc}"
            ) from exc

    # Spec-form outcome: fit on the wiring output.
    if wiring is None:
        raise CompileError(
            "Spec-form outcome requires a wiring node (pass steps.input(data) first)."
        )
    import statsmodels.formula.api as smf

    data = wiring.collect()
    if family is not None:
        fitted = smf.glm(formula, data=data, family=family).fit(tol=1e-12)
    else:
        fitted = smf.ols(formula, data=data).fit()
    return auto_detect_adapter(fitted, formula=formula, data=data)


def _implicit_input(outcome: Any, adapter: ModelAdapter | None) -> Node:
    """Build an input node from the model's training data."""
    if isinstance(outcome, ModelAdapter):
        data = outcome.training_data
    elif adapter is not None:
        data = adapter.training_data
    else:
        detected = auto_detect_adapter(outcome)
        data = detected.training_data
    return Node(kind="input", _payload=data)


def _compute_psi_h(
    compiled_query: Any,
    adapter: ModelAdapter,
    frozen_cov: np.ndarray,
) -> np.ndarray | None:
    """Tier-1 influence ψ^h = ∇h · ψ^β when the adapter exposes scores."""
    score_fn = getattr(adapter, "score_obs", None)
    if score_fn is None:
        return None
    try:
        from pymargins._gradients import gradient

        beta = adapter.coefficients()
        grad = gradient(compiled_query.h, beta, backend="autodiff", fd_step=1e-6)
        psi_beta = jnp.asarray(score_fn(beta))
        if psi_beta.ndim != 2:
            return None
        return np.asarray(grad @ psi_beta.T)
    except Exception:
        return None


class GComputation:
    """Estimator noun for g-computation (standardization) on the new engine.

    This is the v0.4.0 new-surface estimator. It compiles a wiring graph into a
    pre-registered :class:`Plan`, then runs queries through the doctrine engine
    (R2/R3) and returns :class:`GraphResult` objects (R4).
    """

    def __init__(
        self,
        wiring_or_model=None,
        *,
        outcome=None,
        adapter: ModelAdapter | None = None,
        at: str = "overall",
        scale: str = "response",
        method: str = "delta",
        vcov=None,
        ci: str | None = None,
        level: float = 0.95,
        B: int = 1000,
        n_sim: int = 4000,
        seed: int | None = None,
        weights=None,
        gradient_backend: str = "autodiff",
        fd_step: float = 1e-6,
        n_jobs: int = 1,
        progress_bar: bool = False,
    ):
        # Handle positional-model sugar: GComputation(model)
        if outcome is None and wiring_or_model is not None:
            if isinstance(wiring_or_model, Node):
                raise CompileError(
                    "GComputation requires outcome= when passed a wiring node."
                )
            outcome = wiring_or_model
            wiring = None
        else:
            wiring = wiring_or_model

        resolved_outcome = _resolve_outcome(wiring, outcome, adapter)

        if wiring is None:
            wiring = _implicit_input(resolved_outcome, adapter)

        self._plan, self._report, self._compiled = compile(
            wiring,
            resolved_outcome,
            at=at,
            scale=scale,
            method=method,
            vcov=vcov,
            ci=ci,
            level=level,
            B=B,
            n_sim=n_sim,
            seed=seed,
            weights=weights,
            gradient_backend=gradient_backend,
            fd_step=fd_step,
        )

        self._banks = BankSet(self._plan.hash, 0, seed)
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar
        self._wiring = wiring
        self._outcome = outcome

    @property
    def plan(self) -> Plan:
        return self._plan

    @property
    def report(self):
        return self._report

    def _query(self, spec: QuerySpec) -> GraphResult:
        """Run one query spec through the compiled estimator."""
        ctx = QueryContext(
            adapter=self._compiled.adapter,
            base_data=self._compiled.base_data,
            at=self._compiled.at,
            weights=self._compiled.weights,
            phi=self._compiled.phi,
            phi_inv=self._compiled.phi_inv,
            fd_step=self._plan.fd_step,
            gradient_backend=self._plan.gradient_backend,
        )
        compiled_query = compile_query(spec, ctx)
        result_data = execute_query(
            compiled_query,
            adapter=self._compiled.adapter,
            plan=self._plan,
            wiring_facts=self._compiled.wiring_facts,
            banks=self._banks,
            frozen_cov=self._compiled.frozen_cov,
            n_jobs=self._n_jobs,
            progress_bar=self._progress_bar,
            phi=self._compiled.phi,
            phi_inv=self._compiled.phi_inv,
        )
        psi_h = None
        if self._plan.method_resolved == "delta":
            psi_h = _compute_psi_h(
                compiled_query,
                self._compiled.adapter,
                self._compiled.frozen_cov,
            )
        return GraphResult.from_engine(
            result_data,
            plan=self._plan,
            labels=compiled_query.labels,
            population_note=self._compiled.wiring_facts.population_note,
            n_obs=len(self._compiled.base_data),
            psi_h=psi_h,
            phi=self._compiled.phi,
            phi_inv=self._compiled.phi_inv,
        )

    def predict(
        self,
        *,
        atexog=None,
        over=None,
        transform=None,
        label=None,
        outcome=None,
    ) -> GraphResult:
        scenario = {}
        if atexog is not None:
            scenario["atexog"] = atexog
        if over is not None:
            scenario["over"] = over
        return self._query(
            QuerySpec(
                kind="predict",
                scenario=scenario or None,
                transform=transform,
                label=label,
                outcome=outcome,
            )
        )

    def dydx(
        self,
        variables: str | list[str] | None = None,
        *,
        atexog=None,
        over=None,
        transform=None,
        label=None,
        outcome=None,
    ) -> GraphResult:
        if variables is None:
            variables = list(self._compiled.adapter.variable_metadata().keys())
        if isinstance(variables, str):
            variables = (variables,)
        else:
            variables = tuple(variables)
        scenario = {}
        if atexog is not None:
            scenario["atexog"] = atexog
        if over is not None:
            scenario["over"] = over
        return self._query(
            QuerySpec(
                kind="dydx",
                scenario=scenario or None,
                variables=variables,
                transform=transform,
                label=label,
                outcome=outcome,
            )
        )

    def contrasts(
        self,
        *,
        scenarios=None,
        contrasts=None,
        outcome=None,
    ) -> GraphResult:
        return self._query(
            QuerySpec(
                kind="contrasts",
                scenarios=tuple(scenarios) if scenarios is not None else (),
                contrast_weights=contrasts,
                outcome=outcome,
            )
        )

    def evaluate(
        self,
        *,
        scenarios=None,
        compose=None,
        outcome=None,
    ) -> GraphResult:
        return self._query(
            QuerySpec(
                kind="evaluate",
                scenarios=tuple(scenarios) if scenarios is not None else (),
                compose=compose,
                outcome=outcome,
            )
        )

    def wtp(
        self,
        attribute: str,
        price: str,
        *,
        atexog=None,
        over=None,
        transform=None,
        label=None,
        outcome=None,
    ) -> GraphResult:
        scenario = {}
        if atexog is not None:
            scenario["atexog"] = atexog
        if over is not None:
            scenario["over"] = over
        return self._query(
            QuerySpec(
                kind="wtp",
                scenario=scenario or None,
                variables=(attribute, price),
                transform=transform,
                label=label,
                outcome=outcome,
            )
        )

    def rmst(
        self,
        *,
        horizon: float | None = None,
        atexog=None,
        over=None,
        n_grid: int = 80,
        outcome=None,
    ) -> GraphResult:
        if horizon is None:
            raise ValueError("rmst() requires horizon=")
        scenario = {}
        if atexog is not None:
            scenario["atexog"] = atexog
        if over is not None:
            scenario["over"] = over
        return self._query(
            QuerySpec(
                kind="rmst",
                scenario=scenario or None,
                horizon=horizon,
                n_grid=n_grid,
                outcome=outcome,
            )
        )

    def joint(self, *results):
        raise NotImplementedError("joint() lands in 0.5.0")

    def __repr__(self):
        return (
            f"GComputation(method={self._plan.method_resolved!r}, "
            f"at={self._plan.at!r}, scale={self._plan.scale!r})"
        )
