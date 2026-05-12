"""
pymargins.margins

The Margins class — the user-facing entry point. Wraps a fitted model with
a committed analytical posture (inference scale, vcov, level, at,
inference method) and exposes methods for adjusted predictions, slopes,
linear combinations, and arbitrary differentiable estimands.

Architectural posture
---------------------
A Margins instance represents an analysis, not just a model wrapper. The
constructor commits to:
  - phi/phi_inv (inference scale)
  - vcov (variance estimator)
  - at (default evaluation point: overall, mean, typical, etc.)
  - level (confidence level)
  - method (default inference method)

These are session-level commitments that apply to every method call.
Switching any of them requires a new Margins instance. This is a deliberate
design choice supporting analytical pre-commitment and reproducibility — see
the package primer for the philosophical motivation.

Per-call arguments specify the *question* (variables, scenarios, contrasts);
session-level arguments specify the *posture*.
"""

from __future__ import annotations
from typing import Callable, Optional, Union, Any
import weakref

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._adapter import (
    ModelAdapter,
    auto_detect_adapter,
    InferenceMethod,
)
from ._gradients import GradientBackend
from ._inference import InferenceConfig, run_inference
from ._estimands import (
    make_prediction_estimand,
    make_slope_estimand,
    make_linear_combination_estimand,
    make_evaluate_estimand,
)
from ._scenarios import (
    expand_scenario,
    make_aggregation_resolver,
    _auto_label_from_atexog,
)
from ._kappa import session_kappa
from ._result import MarginsResult, DiagnosticResult


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AtSpec = Union[str, dict, Callable]
ContrastSpec = Union[
    list[float],                  # single contrast
    dict[str, list[float]],       # multiple named contrasts
    np.ndarray,                   # contrast matrix
]


# ---------------------------------------------------------------------------
# Sentinel for strict-mode "not explicitly given"
# ---------------------------------------------------------------------------

_NOT_GIVEN = object()


# ---------------------------------------------------------------------------
# The Margins class
# ---------------------------------------------------------------------------

class Margins:
    """Wrapper around a fitted model exposing marginal-effects analysis.

    Parameters
    ----------
    model : fitted result object
        From statsmodels, linearmodels, or sklearn (with limitations).
        The adapter is auto-detected unless explicitly provided.

    phi : callable, optional
        Back-transform from inference scale to reporting scale. Applied to
        CI endpoints. Default: None (identity scale). Common choices:
          jnp.exp                  → log scale (ratios, RR, fold-change)
          scipy.special.expit      → logit scale (odds ratios, probabilities)
          jnp.expm1                → lift scale (RR - 1)
          jnp.tanh                 → Fisher z scale (correlations)

    phi_inv : callable, optional
        Forward transform from reporting scale to inference scale. Required
        if phi is non-identity. Used for converting user-supplied null values
        in hypothesis tests onto the inference scale.

    vcov : str, ndarray, dict, or None, default None
        Variance estimator specification. None uses the framework default.
        Strings like "HC0"-"HC3" request robust flavors. A dict like
        {"type": "cluster", "groups": ids} requests cluster-robust SEs.
        An ndarray uses a user-supplied Σ̂ directly.

    weights : array-like, optional
        Aggregation weights (distinct from any sampling weights baked into
        the model fit). Used when computing AME-style averages and when
        constructing weighted summary points for typical/mean/etc.

    at : str, dict, or callable, default "overall"
        Where to evaluate predictions and marginal effects, following
        statsmodels convention. Common values:
          "overall"   : per-row, then average (AME / AAP)
          "mean"      : evaluate at the mean of all variables (MEM)
          "typical"   : type-aware: median continuous, mode discrete
          "median"    : median of all
          "mode"      : mode of all (errors on continuous)
          dict        : per-variable specification with "_default" key
          callable    : (data) -> 1-row representative DataFrame

    level : float, default 0.95
        Confidence level for CIs.

    method : str, default "delta"
        Default inference method: "delta", "simulation", or "bootstrap".

    kappa_threshold : float, default 0.3
        Curvature above which delta auto-falls-back to simulation. Set to
        infinity to disable automatic fallback.

    n_sim : int, default 4000
        Number of simulation draws for Krinsky–Robb simulation inference.

    n_boot : int, default 1000
        Number of bootstrap replicates for bootstrap inference.

    n_jobs : int, default 1
        Number of parallel workers for bootstrap refits. ``-1`` uses all
        available cores. Uses thread-based parallelization with BLAS threads
        limited to 1 per worker to prevent oversubscription.

    gradient_backend : str, default "auto"
        Gradient method. "auto" uses the adapter's recommendation. Manual
        choices: "autodiff", "fd", "wrapped_fd".

    fd_step : float, default 1e-6
        Step size for FD-based gradients. The default is calibrated for
        float64 precision.

    diagnostics : bool, default True
        Whether to compute κ and other diagnostics on every call. Disable
        for performance in tight loops.

    cluster : array-like, optional
        Cluster IDs for cluster bootstrap. When provided and ``method='bootstrap'``,
        resamples clusters with replacement instead of individual rows.
        Must be the same length as the training data and must not contain NaN.
        Any hashable type is accepted (strings, integers, tuples, etc.).
        Mutually exclusive with ``block_size``.

    block_size : int, optional
        Block length for block bootstrap. When provided and ``method='bootstrap'``,
        resamples contiguous blocks with replacement instead of individual rows.
        Mutually exclusive with ``cluster``.

    bootstrap_config : dict, optional
        Advanced bootstrap options. Supported keys:
          - "block_type": "moving" (default), "nonoverlapping", or "circular"

    strict : bool, default False
        If True, no defaults are inferred — every analytical choice must be
        explicit at construction. Useful for pre-registered or audit-relevant
        work. Implementation detail: when strict, any unspecified config
        argument raises rather than using its default.

    adapter : ModelAdapter, optional
        Explicit adapter overriding auto-detection. Useful for unusual
        models or when customizing predict semantics.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        model,
        *,
        phi: Optional[Callable] = _NOT_GIVEN,
        phi_inv: Optional[Callable] = _NOT_GIVEN,
        vcov: Optional[Union[str, np.ndarray, dict]] = _NOT_GIVEN,
        weights: Optional[np.ndarray] = _NOT_GIVEN,
        at: AtSpec = _NOT_GIVEN,
        level: float = _NOT_GIVEN,
        method: InferenceMethod = _NOT_GIVEN,
        kappa_threshold: float = _NOT_GIVEN,
        rng_seed: Optional[int] = _NOT_GIVEN,
        n_sim: int = _NOT_GIVEN,
        n_boot: int = _NOT_GIVEN,
        n_jobs: int = _NOT_GIVEN,
        gradient_backend: GradientBackend = _NOT_GIVEN,
        fd_step: float = _NOT_GIVEN,
        diagnostics: bool = _NOT_GIVEN,
        cluster: Optional[Any] = _NOT_GIVEN,
        block_size: Optional[int] = _NOT_GIVEN,
        bootstrap_config: Optional[dict] = _NOT_GIVEN,
        strict: bool = False,
        adapter: Optional[ModelAdapter] = None,
    ):
        # Strict mode: every config argument must be explicitly given
        if strict:
            for name, value in [
                ("phi", phi), ("phi_inv", phi_inv), ("vcov", vcov),
                ("weights", weights), ("at", at), ("level", level),
                ("method", method), ("kappa_threshold", kappa_threshold),
                ("rng_seed", rng_seed), ("n_sim", n_sim), ("n_boot", n_boot),
                ("n_jobs", n_jobs), ("gradient_backend", gradient_backend),
                ("fd_step", fd_step), ("diagnostics", diagnostics),
                ("cluster", cluster), ("block_size", block_size),
                ("bootstrap_config", bootstrap_config),
            ]:
                if value is _NOT_GIVEN:
                    raise ValueError(
                        f"strict=True: argument {name!r} must be explicitly given"
                    )
            if gradient_backend == "auto":
                raise ValueError(
                    "strict=True: gradient_backend='auto' is not allowed. "
                    "Specify an explicit backend ('autodiff', 'fd', 'wrapped_fd')."
                )

        # Apply defaults for anything not explicitly given
        phi = None if phi is _NOT_GIVEN else phi
        phi_inv = None if phi_inv is _NOT_GIVEN else phi_inv
        vcov = None if vcov is _NOT_GIVEN else vcov
        weights = None if weights is _NOT_GIVEN else weights
        at = "overall" if at is _NOT_GIVEN else at
        level = 0.95 if level is _NOT_GIVEN else level
        method = "delta" if method is _NOT_GIVEN else method
        kappa_threshold = 0.3 if kappa_threshold is _NOT_GIVEN else kappa_threshold
        rng_seed = None if rng_seed is _NOT_GIVEN else rng_seed
        n_sim = 4000 if n_sim is _NOT_GIVEN else n_sim
        n_boot = 1000 if n_boot is _NOT_GIVEN else n_boot
        n_jobs = 1 if n_jobs is _NOT_GIVEN else n_jobs
        if not isinstance(n_jobs, int) or (n_jobs != -1 and n_jobs < 1):
            raise ValueError(
                f"n_jobs must be a positive integer or -1 (all CPUs), got {n_jobs}"
            )
        gradient_backend = "auto" if gradient_backend is _NOT_GIVEN else gradient_backend
        fd_step = 1e-6 if fd_step is _NOT_GIVEN else fd_step
        diagnostics = True if diagnostics is _NOT_GIVEN else diagnostics
        cluster = None if cluster is _NOT_GIVEN else cluster
        block_size = None if block_size is _NOT_GIVEN else block_size
        bootstrap_config = None if bootstrap_config is _NOT_GIVEN else bootstrap_config

        # Validation: phi/phi_inv must come as a pair
        if (phi is None) != (phi_inv is None):
            raise ValueError(
                "phi and phi_inv must be provided together (or neither)."
            )

        # Validation: numeric session parameters
        if not (0.0 < level < 1.0):
            raise ValueError(f"level must be in (0, 1), got {level}")
        if not isinstance(n_sim, int) or n_sim < 1:
            raise ValueError(f"n_sim must be a positive integer, got {n_sim}")
        if not isinstance(n_boot, int) or n_boot < 1:
            raise ValueError(f"n_boot must be a positive integer, got {n_boot}")
        if fd_step <= 0 or not np.isfinite(fd_step):
            raise ValueError(f"fd_step must be a positive finite float, got {fd_step}")

        self.model = model
        self.phi = phi
        self.phi_inv = phi_inv
        self.vcov_spec = vcov
        self.weights = weights
        self.at = at
        self.level = level
        self.method = method
        self.kappa_threshold = kappa_threshold
        self.rng_seed = rng_seed
        self.n_sim = n_sim
        self.n_boot = n_boot
        self.n_jobs = n_jobs
        self.fd_step = fd_step
        self.diagnostics = diagnostics
        self.cluster = cluster
        self.block_size = block_size
        self.bootstrap_config = bootstrap_config
        self.strict = strict

        # Adapter setup
        self.adapter = adapter if adapter is not None else auto_detect_adapter(model)
        self.adapter.attach(self)

        # Validate weights
        if self.weights is not None:
            w_arr = np.asarray(self.weights)
            if not np.all(np.isfinite(w_arr)):
                raise ValueError("weights must be finite (no NaN or Inf).")
            if np.any(w_arr < 0):
                raise ValueError("weights must be non-negative.")

        # Validate cluster IDs against training data length
        if self.cluster is not None:
            cluster_arr = np.asarray(self.cluster)
            if np.any(pd.isna(cluster_arr)):
                raise ValueError("cluster IDs must not contain NaN values.")
            try:
                n_data = len(self.adapter.training_data)
            except (NotImplementedError, AttributeError, TypeError):
                n_data = None
            if n_data is not None and len(cluster_arr) != n_data:
                raise ValueError(
                    f"cluster IDs length ({len(cluster_arr)}) must match "
                    f"training data length ({n_data})."
                )

        # Mutual exclusion: cluster and block_size
        if self.cluster is not None and self.block_size is not None:
            raise ValueError(
                "cluster and block_size are mutually exclusive. "
                "Use cluster for cluster bootstrap or block_size for block bootstrap, not both."
            )

        # Validate block_size
        if self.block_size is not None:
            if not isinstance(self.block_size, int) or self.block_size < 1:
                raise ValueError("block_size must be a positive integer.")
            try:
                n_data = len(self.adapter.training_data)
            except (NotImplementedError, AttributeError, TypeError):
                n_data = None
            if n_data is not None and self.block_size > n_data:
                raise ValueError(
                    f"block_size ({self.block_size}) cannot exceed "
                    f"training data length ({n_data})."
                )
            cfg = self.bootstrap_config or {}
            block_type = cfg.get("block_type", "moving")
            if block_type not in ("moving", "nonoverlapping", "circular"):
                raise ValueError(
                    f"Unsupported block_type: {block_type!r}. "
                    f"Supported: 'moving', 'nonoverlapping', 'circular'."
                )

        # Eagerly freeze Σ̂ at construction so mutations between now and the
        # first method call cannot change the session's analytical posture.
        _ = self._frozen_cov()

        # Gradient backend resolution
        if gradient_backend == "auto":
            self.gradient_backend = self.adapter.gradient_backend_recommendation
        else:
            self.gradient_backend = gradient_backend

    # -----------------------------------------------------------------------
    # Convenience constructors for common analytical postures
    # -----------------------------------------------------------------------

    @classmethod
    def linear_scale(cls, model, **kwargs) -> "Margins":
        """Identity scale: contrasts are absolute differences. The default."""
        return cls(model, phi=None, phi_inv=None, **kwargs)

    @classmethod
    def log_scale(cls, model, **kwargs) -> "Margins":
        """Log scale: contrasts are log-ratios; reported as ratios.

        Use for relative risks, fold-change, rate ratios, hazard ratios.
        Predictions reported with asymmetric CIs naturally bounded by zero.
        """
        return cls(model, phi=jnp.exp, phi_inv=jnp.log, **kwargs)

    @classmethod
    def logit_scale(cls, model, **kwargs) -> "Margins":
        """Logit scale: contrasts are log-odds-ratios; reported as ORs.

        Predictions reported as probabilities with asymmetric CIs naturally
        bounded in (0, 1).
        """
        from jax.scipy.special import expit, logit
        return cls(model, phi=expit, phi_inv=logit, **kwargs)

    @classmethod
    def lift_scale(cls, model, **kwargs) -> "Margins":
        """Lift scale: contrasts on log(1+lift); reported as (1+p1)/(1+p0) - 1.

        For marketing/uplift analysis where 0 represents no effect and
        positive values represent multiplicative excess.

        Note: This is *not* the same as RR - 1 (p1/p0 - 1). For true lift
        (p1-p0)/p0, use ``evaluate(compose=...)`` directly.
        """
        return cls(model, phi=jnp.expm1, phi_inv=jnp.log1p, **kwargs)

    @classmethod
    def correlation_scale(cls, model, **kwargs) -> "Margins":
        """Fisher z scale: contrasts on z; reported as correlations."""
        return cls(model, phi=jnp.tanh, phi_inv=jnp.arctanh, **kwargs)

    # -----------------------------------------------------------------------
    # Core entry points
    # -----------------------------------------------------------------------

    def predict(
        self,
        *,
        atexog: Optional[Union[dict, "pd.DataFrame"]] = None,
        over: Optional[Union[str, list[str]]] = None,
        transform: Optional[Callable] = None,
        label: Optional[str] = None,
        outcome: Optional[Union[int, list[int]]] = None,
    ) -> MarginsResult:
        """Adjusted prediction (level quantity).

        Computes E[y | scenario] for the specified scenario(s), with
        inference on the session's inference scale.

        The session's `at` setting controls the default evaluation point:
          - "overall"  → AAP: average over observed rows, optionally with
                         `atexog` substituted in
          - "typical"  → APR-style: evaluate at the typical individual
          - "mean"     → APM: evaluate at the sample mean of every variable

        When `atexog` provides multiple values (e.g., a list for one
        variable), a Cartesian-product grid is built and each combination
        yields a row in the result vector with joint inference.

        Parameters
        ----------
        atexog : dict or DataFrame, optional
            Values for exogenous variables, following the statsmodels
            convention. Examples:
              {"treatment": 1}              single value
              {"treatment": [0, 1]}         grid; produces 2 estimands
              DataFrame                      explicit row(s)

        over : str or list of str, optional
            Subgroup variable(s). Estimand is computed within each group
            level; result is vector-valued with one entry per group.

        transform : callable, optional
            Differentiable per-row mapping applied to predictions before
            aggregation (averaging). Receives the per-row prediction array
            ``μ`` of shape ``(n_rows,)`` and returns an array of the same
            shape (or broadcastable). Must be JAX-compatible for delta
            inference.

            Distinct from ``evaluate(compose=…)``, which composes a function
            *across already-aggregated scenario predictions*; ``transform``
            here is a per-row mapping that runs *before* aggregation.

        label : str, optional
            Override label used in output summaries. Only applies when the
            call produces a single estimand (no grid expansion and no ``over``).

        outcome : int or list of int, optional
            For multi-outcome models (e.g. MNLogit, OrderedModel), subset
            to the specified outcome class(es). Default returns all outcomes.

        Returns
        -------
        result : MarginsResult
            For ``over=`` with k group levels, a length-k vector result
            with one estimate per level and joint inference. For multiple
            ``over=`` variables, a vector over the Cartesian product of
            observed level combinations.
        """
        if transform is not None and not callable(transform):
            raise TypeError(f"transform must be callable, got {type(transform).__name__}")
        if atexog is not None and hasattr(atexog, "iloc"):
            scenario = {"data": atexog, "over": over, "label": label}
        else:
            scenario = {"atexog": atexog, "over": over, "label": label}
        h, labels, scenarios = self._build_prediction_estimand(scenario, transform)
        config = self._inference_config()
        meta = {"kind": "prediction"}
        if label is not None:
            if labels is None or (isinstance(labels, list) and len(labels) == 1):
                labels = [label]
            else:
                import warnings
                warnings.warn(
                    "label= is ignored when atexog or over produces multiple estimands",
                    UserWarning,
                    stacklevel=2,
                )
        if labels is not None:
            meta["labels"] = labels
        if scenarios:
            meta["scenarios"] = scenarios
        if over is not None:
            meta["over"] = [over] if isinstance(over, str) else list(over)
        _over_values = [s.get("_over_values") for s in scenarios]
        if any(v is not None for v in _over_values):
            meta["_over_values"] = _over_values
        h_factory = None
        if config.method == "bootstrap":
            h_factory = lambda new_adapter: self._build_prediction_estimand(
                scenario, transform, adapter=new_adapter
            )[0]
        result_data = run_inference(
            h, self.adapter, config,
            estimand_metadata=meta,
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    def dydx(
        self,
        variables: Union[str, list[str]],
        *,
        atexog: Optional[Union[dict, "pd.DataFrame"]] = None,
        over: Optional[Union[str, list[str]]] = None,
        transform: Optional[Callable] = None,
        label: Optional[str] = None,
        outcome: Optional[Union[int, list[int]]] = None,
    ) -> MarginsResult:
        """Slope (∂μ/∂x_j) for continuous covariates.

        The session's `at` setting controls AME vs MEM behavior. For
        discrete variables, use contrasts() instead — calling dydx() on a
        discrete variable raises ValueError.

        Parameters
        ----------
        variables : str or list of str
            Variable(s) to compute slopes for. For multiple variables,
            returns a vector estimand with joint inference.

        label : str, optional
            Override label used in output summaries. Only applies when the
            call produces a single estimand (no grid expansion and no ``over``).

        outcome : int or list of int, optional
            For multi-outcome models, subset to the specified outcome
            class(es). Default returns all outcomes.

        Other parameters : see predict().

        Returns
        -------
        result : MarginsResult
        """
        if transform is not None and not callable(transform):
            raise TypeError(f"transform must be callable, got {type(transform).__name__}")
        var_list = [variables] if isinstance(variables, str) else list(variables)
        for v in var_list:
            info = self.adapter.variable_metadata().get(v)
            if info is None:
                raise ValueError(f"Unknown variable: {v}")
            if info.var_type in ("binary", "categorical"):
                raise ValueError(
                    f"Variable {v!r} is {info.var_type}; use contrasts() "
                    "for discrete contrasts."
                )

        if atexog is not None and hasattr(atexog, "iloc"):
            scenario = {"data": atexog, "over": over, "label": label}
        else:
            scenario = {"atexog": atexog, "over": over, "label": label}
        h, labels, scenarios = self._build_slope_estimand(scenario, var_list, transform)
        config = self._inference_config()
        meta = {"kind": "slope", "variables": var_list}
        if label is not None:
            if labels is None or (isinstance(labels, list) and len(labels) == 1):
                labels = [label]
            else:
                import warnings
                warnings.warn(
                    "label= is ignored when atexog or over produces multiple estimands",
                    UserWarning,
                    stacklevel=2,
                )
        if labels is not None:
            meta["labels"] = labels
        if scenarios:
            meta["scenarios"] = scenarios
        if over is not None:
            meta["over"] = [over] if isinstance(over, str) else list(over)
        _over_values = [s.get("_over_values") for s in scenarios]
        if any(v is not None for v in _over_values):
            meta["_over_values"] = _over_values
        h_factory = None
        if config.method == "bootstrap":
            h_factory = lambda new_adapter: self._build_slope_estimand(
                scenario, var_list, transform, adapter=new_adapter
            )[0]
        result_data = run_inference(
            h, self.adapter, config,
            estimand_metadata=meta,
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    def contrasts(
        self,
        *,
        scenarios: list[dict],
        contrasts: ContrastSpec,
        outcome: Optional[Union[int, list[int]]] = None,
    ) -> MarginsResult:
        """Linear combination(s) of predictions across scenarios.

        Computes one or more weighted sums of per-scenario predictions on
        the session's inference scale:

            contrast_k = Σᵢ weight_ki * h(β, scenario_i)

        Subsumes pairwise contrasts, diff-in-diff, triple difference, and
        arbitrary linear hypotheses about combinations of predictions.

        Note that the linear combination happens on the **inference scale**,
        not the response scale. For a log-scale session, the per-scenario
        predictions are log-transformed before combination, so a contrast
        with weights [+1, -1] produces log(p₁) − log(p₀) = log(p₁/p₀), a
        log-ratio. The session's φ exponentiates this for reporting as RR.
        This is the correct behavior; it's what makes log-scale "ratios"
        and identity-scale "differences" both expressible by the same
        primitive.

        Parameters
        ----------
        scenarios : list of dicts
            Per-scenario specifications. Each dict has optional 'atexog',
            'over', and 'label' keys. The label is used in output for
            identification.

        contrasts : list of floats, dict, or ndarray
            Weight specification:
              list of floats               : single contrast → single result
              dict {name: list of floats}  : named contrasts → multi-row,
                                             joint inference across them
              ndarray of shape (k, n)      : contrast matrix; rows are
                                             contrasts, auto-labeled
            All weight vectors must have length len(scenarios).

        Returns
        -------
        result : MarginsResult
            For a single weight vector: scalar result. For multiple
            contrasts (dict or matrix), vector result with one entry per
            contrast and joint inference across them via the shared Σ̂.

        Notes
        -----
        contrasts() forms a *linear combination on the inference scale*:

            result = φ( Σᵢ wᵢ · φ⁻¹(pᵢ) )

        where pᵢ is the response-scale prediction for scenario i.
        On an identity-scale session this is just a weighted sum of
        probabilities; on a log-scale session it becomes a log-ratio
        (weights [+1, −1] give log(p₁) − log(p₂) = log(p₁/p₂)).

        For nonlinear functions of predictions (ratios, NNT, lift,
        custom utility), use evaluate() instead — it applies the
        function on the response scale before φ⁻¹.

        Example: pairwise risk difference on a linear-scale session
            m = Margins.linear_scale(fitted_logit, at="overall")
            rd = m.contrasts(
                scenarios=[
                    {"atexog": {"treatment": 1}},
                    {"atexog": {"treatment": 0}},
                ],
                contrasts=[+1, -1],
            )
            # rd.estimate is P(Y=1 | treated) − P(Y=1 | control)
        """
        if len(scenarios) == 0:
            raise ValueError("contrasts() requires at least one scenario")

        # Normalize the contrasts argument into the dict-or-vector forms
        # accepted by make_linear_combination_estimand.
        if isinstance(contrasts, dict):
            weights_arg = {name: jnp.asarray(w) for name, w in contrasts.items()}
            labels = list(contrasts.keys())
        elif isinstance(contrasts, (np.ndarray, jnp.ndarray)) and contrasts.ndim == 2:
            weights_arg = {
                f"contrast[{i}]": jnp.asarray(contrasts[i])
                for i in range(contrasts.shape[0])
            }
            labels = list(weights_arg.keys())
        elif isinstance(contrasts, list) and contrasts and isinstance(contrasts[0], list):
            # list-of-lists: validate and convert to jnp.ndarray
            contrasts_arr = jnp.asarray(contrasts)
            if contrasts_arr.ndim != 2:
                raise ValueError(
                    f"list-of-lists contrast must be 2D after conversion, got {contrasts_arr.ndim}D"
                )
            weights_arg = {
                f"contrast[{i}]": contrasts_arr[i]
                for i in range(contrasts_arr.shape[0])
            }
            labels = list(weights_arg.keys())
        else:
            weights_arg = jnp.asarray(contrasts)
            labels = [scenarios[0].get("label", "contrast")]

        # Validate scenarios element type
        for i, s in enumerate(scenarios):
            if not isinstance(s, dict):
                raise TypeError(
                    f"Each scenario must be a dict, got {type(s).__name__} at index {i}"
                )

        # Validate weight lengths and finiteness
        n_scenarios = len(scenarios)
        _weights_to_check = []
        if isinstance(weights_arg, dict):
            for name, w in weights_arg.items():
                if w.shape[0] != n_scenarios:
                    raise ValueError(
                        f"Contrast {name!r} has {w.shape[0]} weights but "
                        f"{n_scenarios} scenarios were provided."
                    )
                _weights_to_check.append(w)
        else:
            if weights_arg.shape[0] != n_scenarios:
                raise ValueError(
                    f"Contrast has {weights_arg.shape[0]} weights but "
                    f"{n_scenarios} scenarios were provided."
                )
            _weights_to_check.append(weights_arg)

        for w in _weights_to_check:
            if not jnp.all(jnp.isfinite(w)):
                raise ValueError("Contrast weights must be finite (no NaN or Inf)")

        h = self._build_contrast_estimand(scenarios, weights_arg)
        config = self._inference_config()

        h_factory = None
        if config.method == "bootstrap":
            h_factory = lambda new_adapter: self._build_contrast_estimand(
                scenarios, weights_arg, adapter=new_adapter
            )
        result_data = run_inference(
            h, self.adapter, config,
            estimand_metadata={"kind": "contrasts", "labels": labels, "scenarios": scenarios},
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    def evaluate(
        self,
        *,
        scenarios: list[dict],
        compose: Callable,
        outcome: Optional[Union[int, list[int]]] = None,
    ) -> MarginsResult:
        """Arbitrary differentiable function of scenario predictions.

        For linear combinations, prefer contrasts() — it's clearer and uses
        a faster path. evaluate() is the escape hatch for nonlinear cases:
        NNT (1/(p_a − p_b)), ratios across non-paired scenarios, custom
        utility functions, anything not expressible as a weighted sum.

        The compose function operates on the **response-scale** predictions
        from each scenario (before phi_inv). The session's phi_inv is
        applied to the compose output to lift it onto the inference scale
        for delta-method computation. The session's phi is applied to CI
        endpoints for reporting.

        Parameters
        ----------
        scenarios : list of dicts
            Per-scenario specs.

        compose : callable (predictions: jax array) -> scalar or vector
            Receives an array of per-scenario aggregated predictions in the
            order they were provided, returns the derived quantity. Must be
            JAX-compatible (use jnp ops, no Python conditionals on tracer
            values). If non-differentiable, the engine auto-routes to
            simulation or bootstrap.

        Returns
        -------
        result : MarginsResult

        Notes
        -----
        evaluate() applies an *arbitrary function on the response scale*:

            result = φ( φ⁻¹( compose(p₁, p₂, …) ) )

        where pᵢ is the response-scale prediction for scenario i.
        This is the escape hatch for nonlinear estimands: ratios, NNT,
        lift, interaction terms, or any custom utility function.

        Contrast with contrasts(), which forms a *linear combination on
        the inference scale*:

            contrasts() :  φ( Σᵢ wᵢ · φ⁻¹(pᵢ) )
            evaluate()  :  φ( φ⁻¹( compose(p) ) )

        For linear combinations (risk differences, log-ratios, etc.)
        prefer contrasts() — it is clearer and uses a faster path.

        Auto-routing behavior
        ---------------------
        If ``compose`` is not JAX-differentiable (e.g. it uses Python
        ``if`` on tracer values), the delta method cannot compute a
        gradient and the engine **auto-routes** to simulation or
        bootstrap, depending on the session's ``method``. A
        ``UserWarning`` is emitted when this fallback occurs. The
        fallback uses ``n_sim`` (simulation) or ``n_boot`` (bootstrap)
        draws from the session config. This is correct but slower than
        the delta-method path.

        Example 1: Risk ratio via evaluate() on a linear-scale session
            m = Margins.linear_scale(fitted_logit, at="overall")
            rr = m.evaluate(
                scenarios=[
                    {"atexog": {"treatment": 1}},
                    {"atexog": {"treatment": 0}},
                ],
                compose=lambda p: p[0] / p[1],
            )
            # rr.estimate is P(Y=1 | treated) / P(Y=1 | control)

        Example 2: Number needed to treat (NNT)
            m = Margins.linear_scale(fitted_logit, at="overall")
            nnt = m.evaluate(
                scenarios=[
                    {"atexog": {"treatment": 1}},
                    {"atexog": {"treatment": 0}},
                ],
                compose=lambda p: 1.0 / (p[0] - p[1]),
            )
            # nnt.estimate is 1 / (P_treated − P_control)

        Example 3: Lift (relative effect)
            lift = m.evaluate(
                scenarios=[
                    {"atexog": {"treatment": 1}},
                    {"atexog": {"treatment": 0}},
                ],
                compose=lambda p: (p[0] - p[1]) / p[1],
            )
            # lift.estimate is (P_treated − P_control) / P_control
        """
        if not callable(compose):
            raise TypeError(f"compose must be callable, got {type(compose).__name__}")
        h = self._build_evaluate_estimand(scenarios, compose)
        config = self._inference_config()

        labels = [
            s.get("label", _auto_label_from_atexog(s.get("atexog")) or f"scenario[{i}]")
            for i, s in enumerate(scenarios)
        ]
        h_factory = None
        if config.method == "bootstrap":
            h_factory = lambda new_adapter: self._build_evaluate_estimand(
                scenarios, compose, adapter=new_adapter
            )
        result_data = run_inference(
            h, self.adapter, config,
            estimand_metadata={"kind": "evaluate", "labels": labels, "scenarios": scenarios},
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    # -----------------------------------------------------------------------
    # Outcome slicing for multi-outcome models
    # -----------------------------------------------------------------------

    def _slice_by_outcome(
        self,
        result_data: dict,
        outcome: Union[int, list[int]],
    ) -> dict:
        """Slice result arrays to the requested outcome indices.

        For multi-outcome models (MNLogit, OrderedModel), the inference
        engine returns estimates/SEs/CIs for all outcomes. This helper
        subsets them and updates labels accordingly.
        """
        n_outcomes = self.adapter.n_outcomes
        labels = self.adapter.outcome_labels or [str(i) for i in range(n_outcomes)]

        keys = [outcome] if isinstance(outcome, int) else list(outcome)
        for k in keys:
            if not (0 <= k < n_outcomes):
                raise ValueError(
                    f"Outcome index {k} is out of range for model with "
                    f"{n_outcomes} outcomes (valid: 0..{n_outcomes - 1})."
                )
        idx = np.asarray(keys, dtype=int)

        def _slice(arr):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return arr[idx]
            elif arr.ndim == 2:
                # (n_atoms, n_outcomes) or (n_outcomes, n_params)
                # We want to slice along the outcome axis. Heuristic:
                # if the last dim equals n_outcomes, slice last axis.
                if arr.shape[-1] == n_outcomes:
                    return arr[..., idx]
                elif arr.shape[0] == n_outcomes:
                    return arr[idx]
                else:
                    return arr  # Can't determine outcome axis
            elif arr.ndim == 3:
                # (n_atoms, n_outcomes, n_params) or (n_sim, n_atoms, n_outcomes)
                if arr.shape[-1] == n_outcomes:
                    return arr[..., idx]
                elif arr.shape[1] == n_outcomes:
                    return arr[:, idx]
                else:
                    return arr
            return arr

        result = dict(result_data)
        for key in ("estimate", "std_error", "conf_int_lower", "conf_int_upper",
                    "gradient", "draws", "kappa"):
            if key in result:
                result[key] = _slice(result[key])

        # Update labels
        meta = result.get("estimand_metadata", {})
        old_labels = meta.get("labels")
        if old_labels is not None:
            new_labels = []
            for lab in old_labels:
                for k in idx:
                    suffix = labels[k]
                    new_labels.append(f"{lab} ({suffix})")
            meta = dict(meta)
            meta["labels"] = new_labels
            meta["outcome_sliced"] = True
            result["estimand_metadata"] = meta

        return result

    # -----------------------------------------------------------------------
    # Diagnostics and reporting
    # -----------------------------------------------------------------------

    def diagnose(
        self,
        n_samples: int = 50,
        rng_seed: Optional[int] = None,
    ) -> DiagnosticResult:
        """Session-level κ diagnostic across the design space.

        Samples representative covariate vectors from the training data and
        computes κ at each, summarizing the distribution. Use to assess
        delta-method validity for the configured analytical posture before
        computing specific estimands.

        The verdict is conservative: based on max κ across samples. Inspect
        the full distribution in the result to see whether the worst-case
        is in a region of the design space that matters for your analysis.

        Parameters
        ----------
        n_samples : int, default 50
            Number of design points to sample.

        rng_seed : int, optional
            For reproducibility.

        Returns
        -------
        diag : DiagnosticResult
        """
        rng = np.random.default_rng(rng_seed)
        base = self._base_data
        if not hasattr(base, "iloc"):
            raise TypeError(
                "diagnose() currently requires base data to be a pandas "
                "DataFrame. Adapters using non-pandas data should override "
                "or supply alternative sampling."
            )

        n = len(base)
        sample_idx = rng.choice(n, size=min(n_samples, n), replace=False)
        sample_X = [
            self.adapter.design_matrix_from_df(base.iloc[[i]])[0]
            for i in sample_idx
        ]

        beta = self.adapter.coefficients()
        Sigma = self._frozen_cov()

        # Build a per-row prediction estimand factory
        def h_factory(x_row):
            x_arr = jnp.atleast_2d(x_row)
            phi_inv = self.phi_inv
            def h(beta_):
                mu = self.adapter.predict(beta_, x_arr)[0]
                return phi_inv(mu) if phi_inv is not None else mu
            return h

        diag_dict = session_kappa(
            h_factory, beta, Sigma, sample_X,
            backend=self.gradient_backend,
            fd_step=self.fd_step,
        )
        return DiagnosticResult(
            kappa_min=diag_dict["min"],
            kappa_median=diag_dict["median"],
            kappa_max=diag_dict["max"],
            kappa_distribution=np.asarray(diag_dict["distribution"]),
            verdict=diag_dict["verdict"],
            n_samples=diag_dict["n_samples"],
            recommendation=diag_dict["recommendation"],
            session_summary=self.summary(),
        )

    def summary(self) -> str:
        """One-paragraph summary of the analytical posture.

        Suitable for inclusion in a methods section or reproducibility log.
        Captures the full set of session-level commitments made by this
        Margins instance.
        """
        scale_name = self._scale_label()
        jobs_str = f"n_jobs={self.n_jobs}" if self.method == "bootstrap" else ""
        return (
            f"Margins session\n"
            f"  Model: {type(self.model).__name__}\n"
            f"  Adapter: {type(self.adapter).__name__}\n"
            f"  Inference scale: {scale_name}\n"
            f"  Variance: {self.vcov_spec or 'default'}\n"
            f"  Confidence level: {self.level}\n"
            f"  At: {self.at}\n"
            f"  Method: {self.method} (κ-threshold={self.kappa_threshold})\n"
            f"  n_sim: {self.n_sim}\n"
            f"  n_boot: {self.n_boot} {jobs_str}\n"
            f"  Gradient backend: {self.gradient_backend}\n"
            f"  Diagnostics: {'enabled' if self.diagnostics else 'disabled'}\n"
            f"  Strict: {self.strict}"
        )

    def _scale_label(self) -> str:
        """Identify the inference scale by reference equality with known
        functions, falling back to 'custom' for user-supplied transforms."""
        if self.phi is None:
            return "identity"
        try:
            from jax.scipy.special import expit
        except ImportError:
            expit = None
        if self.phi is jnp.exp:
            return "log"
        if self.phi is jnp.expm1:
            return "lift"
        if expit is not None and self.phi is expit:
            return "logit"
        if self.phi is jnp.tanh:
            return "fisher_z"
        return "custom"

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @property
    def _base_data(self):
        """The training data used by the model.

        Delegates to the adapter's ``training_data`` property. Adapters
        that cannot expose training data should raise NotImplementedError
        from that property, which propagates here with context.
        """
        try:
            return self.adapter.training_data
        except NotImplementedError as exc:
            raise NotImplementedError(
                "Base data extraction not implemented for this adapter. "
                "Adapters should set self.training_data in __init__, or "
                "override Margins._base_data."
            ) from exc

    def _inference_config(self) -> InferenceConfig:
        """Build the InferenceConfig for a single call.

        All inference-related settings are session-level; per-call overrides
        are not supported by design. Switching method, level, vcov, or scale
        requires constructing a new ``Margins`` instance.
        """
        return InferenceConfig(
            method=self.method,
            level=self.level,
            phi=self.phi,
            phi_inv=self.phi_inv,
            kappa_threshold=self.kappa_threshold,
            gradient_backend=self.gradient_backend,
            fd_step=self.fd_step,
            n_sim=self.n_sim,
            n_boot=self.n_boot,
            n_jobs=self.n_jobs,
            rng_seed=self.rng_seed,
            diagnostics=self.diagnostics,
            cov_params=self._frozen_cov(),
            cluster=self.cluster,
            block_size=self.block_size,
            bootstrap_config=self.bootstrap_config,
        )

    def _frozen_cov(self) -> jnp.ndarray:
        """Resolve Σ̂ once per call and cache on the instance.

        Σ̂ is part of the session's analytical posture (vcov_spec is
        session-level). Caching ensures every result from this session
        carries the same Σ̂ reference even if the underlying model object
        is later mutated or re-fit by the user.
        """
        if not hasattr(self, "_cov_cache"):
            self._cov_cache = self.adapter.covariance(self.vcov_spec)
        return self._cov_cache

    def _wrap_result(self, result_data: dict) -> MarginsResult:
        """Wrap a raw result dict from the engine in a MarginsResult.

        The session's resolved Σ̂ is frozen onto the result so downstream
        composition and hypothesis tests do not re-fetch it from the
        adapter (which could change if the underlying model is mutated).
        """
        n_obs = 0
        try:
            n_obs = len(self.adapter.training_data)
        except (NotImplementedError, AttributeError, TypeError):
            pass

        # Expand labels with outcome suffixes for multi-outcome models
        meta = dict(result_data.get("estimand_metadata", {}))
        if self.adapter.n_outcomes > 1 and not meta.get("outcome_sliced"):
            old_labels = meta.get("labels")
            outcome_labels = self.adapter.outcome_labels or [
                str(i) for i in range(self.adapter.n_outcomes)
            ]
            if old_labels is None:
                old_labels = [""]
            expanded = []
            for lab in old_labels:
                for k in range(self.adapter.n_outcomes):
                    suffix = outcome_labels[k]
                    expanded.append(f"{lab} ({suffix})" if lab else suffix)
            meta["labels"] = expanded

        return MarginsResult(
            estimate=np.asarray(result_data["estimate"]),
            std_error=np.asarray(result_data["std_error"]),
            conf_int_lower=np.asarray(result_data["conf_int_lower"]),
            conf_int_upper=np.asarray(result_data["conf_int_upper"]),
            method=result_data["method"],
            level=result_data["level"],
            n_obs=n_obs,
            kappa=result_data.get("kappa"),
            delta_sim_disagreement=result_data.get("delta_sim_disagreement"),
            fallback_triggered=result_data.get("fallback_triggered", False),
            fallback_reason=result_data.get("fallback_reason"),
            estimand_metadata=meta,
            gradient=result_data.get("gradient"),
            draws=result_data.get("draws"),
            draws_inf=result_data.get("draws_inf"),
            cov_params=np.asarray(self._frozen_cov()),
            phi=self.phi,
            phi_inv=self.phi_inv,
            session=weakref.ref(self),
            ci_method=result_data.get("ci_method"),
            bootstrap_extras=result_data.get("bootstrap_extras"),
        )

    def _build_prediction_estimand(
        self,
        scenario: dict,
        transform: Optional[Callable],
        adapter: Optional[ModelAdapter] = None,
    ) -> tuple[Callable, Optional[list[str]], list[dict]]:
        """Construct the prediction estimand for predict() calls.

        Resolves the scenario into a design matrix using the session's
        ``at`` setting, then wraps it in ``make_prediction_estimand`` with
        ``phi_inv`` applied to lift onto the inference scale.

        When the scenario produces multiple atoms (over-stratification, an
        atexog grid, or both), returns a stacked vector estimand with one
        component per (group × grid point) and the corresponding labels.
        Returns ``(h, None)`` for the single-atom case.

        Also returns a list of scenario dicts, one per atom, containing the
        atexog and over values for plot-ready DataFrame construction.
        """
        adapter = adapter if adapter is not None else self.adapter
        base_data = self._get_base_data(adapter)
        var_meta = adapter.variable_metadata()
        resolver = make_aggregation_resolver(self.at, self.weights)
        groups, over_keys = self._enumerate_groups(scenario, base_data, var_meta)

        sub_scenario = {k: v for k, v in scenario.items() if k != "over"}
        atoms: list[tuple[Optional[str], Callable]] = []
        scenarios: list[dict] = []

        for group_label, group_df in groups:
            df, meta = expand_scenario(
                sub_scenario, group_df, resolver, var_meta,
            )
            from ._tabular import to_pandas_if_needed
            X = adapter.design_matrix_from_df(to_pandas_if_needed(df))
            n_grid = meta.get("n_grid_points", 1)
            rows_per = meta.get("rows_per_grid_point", len(df))
            if n_grid > 1 and X.shape[0] != n_grid * rows_per:
                raise ValueError(
                    f"Design matrix rows ({X.shape[0]}) do not match expected grid layout "
                    f"({n_grid} × {rows_per} = {n_grid * rows_per}). The adapter may have dropped rows."
                )

            for i in range(n_grid):
                start = i * rows_per
                end = (i + 1) * rows_per
                if end > X.shape[0]:
                    raise ValueError(
                        f"Grid block {i} would exceed design matrix rows "
                        f"({end} > {X.shape[0]}). The adapter's "
                        "design_matrix_from_df may have dropped rows."
                    )
                X_i = X[start:end]
                if self.at == "overall":
                    agg_kind = "overall"
                else:
                    agg_kind = "none" if X_i.shape[0] == 1 else "overall"
                h_atom = make_prediction_estimand(
                    adapter, X_i,
                    aggregate=agg_kind,
                    weights=jnp.asarray(self.weights) if self.weights is not None else None,
                    phi_inv=self.phi_inv,
                    transform=transform,
                )
                if n_grid > 1:
                    grid_row = meta.get("grid_rows", [])[i] if i < len(meta.get("grid_rows", [])) else ()
                    grid_keys = meta.get("atexog_keys", [])
                    if grid_row and grid_keys:
                        grid_suffix = ", ".join(f"{k}={v}" for k, v in zip(grid_keys, grid_row))
                    else:
                        grid_suffix = _auto_label_from_atexog(sub_scenario.get("atexog")) or f"grid[{i}]"
                else:
                    grid_suffix = _auto_label_from_atexog(sub_scenario.get("atexog"))
                label = self._format_atom_label(group_label, over_keys, grid_suffix)
                atoms.append((label, h_atom))

                # Build scenario dict for this atom
                scen = {}
                if over_keys is not None:
                    gl = group_label if isinstance(group_label, tuple) else (group_label,)
                    for k, v in zip(over_keys, gl):
                        scen[k] = v
                    scen["_over_values"] = {ok: gl[i] for i, ok in enumerate(over_keys)}
                if n_grid > 1:
                    grid_row = meta.get("grid_rows", [])[i] if i < len(meta.get("grid_rows", [])) else ()
                    grid_keys = meta.get("atexog_keys", [])
                    for k, v in zip(grid_keys, grid_row):
                        scen[k] = v
                else:
                    atexog = sub_scenario.get("atexog", {})
                    if atexog:
                        for k, v in atexog.items():
                            if not isinstance(v, list):
                                scen[k] = v
                scenarios.append(scen)

        h, labels = self._finalize_atoms(atoms)
        return h, labels, scenarios

    def _enumerate_groups(
        self,
        scenario: dict,
        base_data,
        variable_metadata: dict,
    ):
        """Resolve ``scenario['over']`` into a list of (group_label, df) pairs.

        Returns a singleton ``[(None, base_data)]`` when no ``over`` is set,
        so downstream code has a uniform shape.
        """
        over_spec = scenario.get("over")
        if over_spec is None:
            return [(None, base_data)], None
        over_keys = [over_spec] if isinstance(over_spec, str) else list(over_spec)
        unknown = set(over_keys) - set(variable_metadata.keys())
        if unknown:
            raise ValueError(
                f"Unknown over variable(s): {sorted(unknown)}. "
                f"Known variables: {sorted(variable_metadata.keys())}."
            )
        if not hasattr(base_data, "groupby"):
            raise TypeError(
                f"over= requires base_data to support groupby, got {type(base_data).__name__}"
            )
        groups = [(g, gdf) for g, gdf in base_data.groupby(over_keys, sort=True)]
        if not groups:
            raise ValueError(
                f"over={over_keys!r} produced no groups; base data may be empty."
            )
        return groups, over_keys

    @staticmethod
    def _format_atom_label(
        group_label,
        over_keys: Optional[list[str]],
        suffix: Optional[str],
    ) -> Optional[str]:
        """Build a stable label for one atom of a stacked estimand.

        Combines an over-group identifier (``"region=west"``) with an
        optional suffix (a grid index for atexog grids, a variable name
        for multi-variable slopes). Returns ``None`` when the atom is
        unique and unlabeled.
        """
        parts: list[str] = []
        if over_keys is not None:
            gl = group_label if isinstance(group_label, tuple) else (group_label,)
            parts.extend(f"{k}={v}" for k, v in zip(over_keys, gl))
        if suffix is not None:
            parts.append(suffix)
        return ", ".join(parts) if parts else None

    @staticmethod
    def _finalize_atoms(
        atoms: list[tuple[Optional[str], Callable]],
    ) -> tuple[Callable, Optional[list[str]]]:
        """Reduce a list of (label, h_atom) pairs to (h, labels).

        Single atom: return its h directly with no labels. Multiple atoms:
        stack into a vector estimand and return the labels list.

        .. note:: Performance
           The current implementation uses a Python list comprehension
           inside ``jnp.stack``. For many atoms (>50), a ``jax.vmap`` over
           a single parametrized function would be faster because it avoids
           per-atom Python overhead and enables XLA fusion. This is a known
           optimization opportunity tracked in CODE_AUDIT §3.4.
        """
        if len(atoms) == 1:
            label = atoms[0][0]
            return atoms[0][1], ([label] if label is not None else None)
        individual_h = [h for _, h in atoms]
        labels = [lab for lab, _ in atoms]
        def h_vector(beta):
            return jnp.stack([hi(beta) for hi in individual_h])
        return h_vector, labels

    def _get_base_data(self, adapter: Optional[ModelAdapter] = None):
        """Get base data from an adapter."""
        adapter = adapter if adapter is not None else self.adapter
        try:
            return adapter.training_data
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Adapter {type(adapter).__name__} does not expose training_data. "
                "Bootstrap inference and scenario expansion require it."
            ) from exc

    def _build_slope_estimand(
        self,
        scenario: dict,
        var_list: list[str],
        transform: Optional[Callable],
        adapter: Optional[ModelAdapter] = None,
    ) -> tuple[Callable, Optional[list[str]], list[dict]]:
        """Construct the slope estimand for dydx() calls.

        Produces one atom per (over-group × variable). With a single
        variable and no ``over``, returns a scalar estimand. Otherwise
        returns a stacked vector estimand with one component per atom.

        Also returns a list of scenario dicts, one per atom.
        """
        adapter = adapter if adapter is not None else self.adapter
        base_data = self._get_base_data(adapter)
        var_meta = adapter.variable_metadata()
        resolver = make_aggregation_resolver(self.at, self.weights)
        groups, over_keys = self._enumerate_groups(scenario, base_data, var_meta)

        # Type-check each variable up front. column_index_of_variable raises
        # for categorical/binary/discrete, which is the contract we want for
        # dydx(). The returned index is unused — slopes are now data-side
        # central differences (R/Stata-style total derivatives).
        for v in var_list:
            adapter.column_index_of_variable(v)

        sub_scenario = {k: v for k, v in scenario.items() if k != "over"}
        atoms: list[tuple[Optional[str], Callable]] = []
        scenarios: list[dict] = []

        for group_label, group_df in groups:
            df, meta = expand_scenario(
                sub_scenario, group_df, resolver, var_meta,
            )
            if self.at == "overall":
                agg_kind = "overall"
            else:
                agg_kind = "none" if len(df) == 1 else "overall"

            # Build base scenario dict for this group (shared across variables)
            base_scen = {}
            if over_keys is not None:
                gl = group_label if isinstance(group_label, tuple) else (group_label,)
                for k, v in zip(over_keys, gl):
                    base_scen[k] = v
                base_scen["_over_values"] = {ok: gl[i] for i, ok in enumerate(over_keys)}
            atexog = sub_scenario.get("atexog", {})
            if atexog and meta.get("n_grid_points", 1) == 1:
                for k, v in atexog.items():
                    if not isinstance(v, list):
                        base_scen[k] = v

            for var_name in var_list:
                h_atom = make_slope_estimand(
                    adapter, df, var_name,
                    aggregate=agg_kind,
                    weights=jnp.asarray(self.weights) if self.weights is not None else None,
                    phi_inv=self.phi_inv,
                    transform=transform,
                    fd_step=self.fd_step,
                )
                atexog_label = _auto_label_from_atexog(sub_scenario.get("atexog"))
                suffix = f"{atexog_label}, {var_name}" if atexog_label else var_name
                label = self._format_atom_label(group_label, over_keys, suffix)
                atoms.append((label, h_atom))
                scenarios.append(base_scen.copy())

        h, labels = self._finalize_atoms(atoms)
        return h, labels, scenarios

    def _build_contrast_estimand(
        self,
        scenarios: list[dict],
        weights_arg,
        adapter: Optional[ModelAdapter] = None,
    ) -> Callable:
        """Construct a linear combination estimand for contrasts() calls."""
        adapter = adapter if adapter is not None else self.adapter
        base_data = self._get_base_data(adapter)

        scenarios_X = []
        for scenario in scenarios:
            df, _ = expand_scenario(
                scenario,
                base_data=base_data,
                aggregation_resolver=make_aggregation_resolver(
                    self.at, self.weights,
                ),
                variable_metadata=adapter.variable_metadata(),
            )
            from ._tabular import to_pandas_if_needed
            scenarios_X.append(adapter.design_matrix_from_df(to_pandas_if_needed(df)))

        return make_linear_combination_estimand(
            adapter,
            scenarios_X=scenarios_X,
            weights=weights_arg,
            phi_inv=self.phi_inv,
        )

    def _build_evaluate_estimand(
        self,
        scenarios: list[dict],
        compose: Callable,
        adapter: Optional[ModelAdapter] = None,
    ) -> Callable:
        """Construct an arbitrary composition estimand for evaluate() calls."""
        adapter = adapter if adapter is not None else self.adapter
        base_data = self._get_base_data(adapter)

        scenarios_X = []
        for scenario in scenarios:
            df, _ = expand_scenario(
                scenario,
                base_data=base_data,
                aggregation_resolver=make_aggregation_resolver(
                    self.at, self.weights,
                ),
                variable_metadata=adapter.variable_metadata(),
            )
            from ._tabular import to_pandas_if_needed
            scenarios_X.append(adapter.design_matrix_from_df(to_pandas_if_needed(df)))

        return make_evaluate_estimand(
            adapter,
            scenarios_X=scenarios_X,
            compose=compose,
            phi_inv=self.phi_inv,
        )


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: Linear scale (the default) for risk differences
----------------------------------------------------------

    import statsmodels.api as sm
    from pymargins import Margins

    model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    m = Margins.linear_scale(model, vcov="HC3")

    # Risk difference for treatment
    rd = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": "treated"}, "label": "treated"},
            {"atexog": {"treatment": "control"}, "label": "control"},
        ],
        contrasts=[+1, -1],
    )
    print(rd.summary())


Example 2: Log scale for relative risks
---------------------------------------

    m = Margins.log_scale(model, vcov="HC3")

    rr = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": "treated"}},
            {"atexog": {"treatment": "control"}},
        ],
        contrasts=[+1, -1],
    )
    # rr.estimate is RR; CI is asymmetric


Example 3: Diff-in-diff
-----------------------

    m = Margins.linear_scale(model)

    did = m.contrasts(
        scenarios=[
            {"atexog": {"treated": 1, "post": 1}, "label": "TT"},
            {"atexog": {"treated": 1, "post": 0}, "label": "TC"},
            {"atexog": {"treated": 0, "post": 1}, "label": "CT"},
            {"atexog": {"treated": 0, "post": 0}, "label": "CC"},
        ],
        contrasts={
            "DiD":               [+1, -1, -1, +1],
            "treatment_at_post": [+1,  0, -1,  0],
            "time_in_treated":   [+1, -1,  0,  0],
        },
    )
    # did is a vector-valued result with three labeled rows; joint inference


Example 4: AME of a continuous variable
---------------------------------------

    ame_age = m.dydx("age")  # session at="overall" → AME
    print(ame_age.summary())


Example 5: APR (predictions at representative values)
-----------------------------------------------------

    pred = m.predict(atexog={"age": [30, 50, 70]})
    # pred is a 3-row result (one per age value); `at` setting fills
    # other variables per the overall/typical/mean rule


Example 6: NNT via evaluate (escape hatch)
------------------------------------------

    nnt = m.evaluate(
        scenarios=[
            {"atexog": {"treatment": "control"}},
            {"atexog": {"treatment": "treated"}},
        ],
        compose=lambda p: 1.0 / (p[0] - p[1]),
    )
    # nnt has appropriate inference for this nonlinear estimand;
    # κ may be high (1/x is curved); engine may auto-route to simulation


Example 7: Diagnostic and methods-section summary
-------------------------------------------------

    m = Margins.logit_scale(model, vcov="HC3", level=0.95)

    # Methods-section paragraph
    print(m.summary())

    # Pre-flight diagnostic
    diag = m.diagnose(n_samples=50)
    print(diag.summary())


Example 8: Inter-call composability
-----------------------------------

    ame_overall = m.dydx("treatment")
    ame_old     = m.dydx("treatment", atexog={"age_group": "65+"})

    # Difference of two AMEs with proper joint inference
    deviation = ame_old - ame_overall
    test = deviation.test(value=0.0)
    print(test.summary())
"""
