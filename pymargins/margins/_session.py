"""pymargins.margins._session

The Margins class — the user-facing entry point. Wraps a fitted model with
a committed analytical posture (inference scale, vcov, level, at,
inference method) and exposes methods for adjusted predictions, slopes,
linear combinations, and arbitrary differentiable estimands.
"""

from __future__ import annotations

import hashlib
import inspect
import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import (
    InferenceMethod,
    ModelAdapter,
    auto_detect_adapter,
)
from .._gradients import GradientBackend
from .._kappa import session_kappa
from .._result import DiagnosticResult, MarginsResult
from .._result._margins import compose_results
from .._result._text import SummaryString
from .._scenarios import (
    _auto_label_from_atexog,
)
from . import run_inference
from ._atoms import (
    _enumerate_groups,
    _finalize_atoms,
    _format_atom_label,
    _slice_by_outcome,
)
from ._estimands import (
    _build_contrast_estimand,
    _build_evaluate_estimand,
    _build_prediction_estimand,
    _build_slope_estimand,
    _get_base_data,
)
from ._inference_glue import _frozen_cov, _inference_config, _wrap_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_callable_arity(fn: Callable, name: str, min_positional: int = 1) -> None:
    """Validate that ``fn`` accepts at least ``min_positional`` positional args.

    Raises ``ValueError`` with a clear message if the signature is inspectable
    and shows too few positional parameters (and no *args).
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return  # built-in or callable without inspectable signature
    n_positional = 0
    has_var_positional = False
    for param in sig.parameters.values():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            n_positional += 1
        elif param.kind == param.VAR_POSITIONAL:
            has_var_positional = True
    if not has_var_positional and n_positional < min_positional:
        raise ValueError(
            f"{name} must accept at least {min_positional} positional "
            f"argument(s), but signature shows {n_positional}: {sig}"
        )


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AtSpec = str | dict | Callable
ContrastSpec = (
    list[float]  # single contrast
    | dict[str, list[float]]  # multiple named contrasts
    | np.ndarray  # contrast matrix
)


# ---------------------------------------------------------------------------
# Sentinel for strict-mode "not explicitly given"
# ---------------------------------------------------------------------------

_NOT_GIVEN = object()


class Margins:
    """Wrapper around a fitted model exposing marginal-effects analysis.

    Parameters
    ----------
    model : fitted result object
        From statsmodels, linearmodels, or sklearn (with limitations).
        The adapter is auto-detected unless explicitly provided.

    phi : callable, optional
        Back-transform from inference scale to reporting scale. Applied to
        CI endpoints. Default: None (identity scale). Common choices::

            jnp.exp                  # log scale (ratios, RR, fold-change)
            scipy.special.expit      # logit scale (odds ratios, probabilities)
            jnp.expm1                # lift scale (RR - 1)
            jnp.tanh                 # Fisher z scale (correlations)

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
        statsmodels convention. Common values::

            "overall"   # per-row, then average (AME / AAP)
            "mean"      # evaluate at the mean of all variables (MEM)
            "typical"   # type-aware: median continuous, mode discrete
            "median"    # median of all
            "mode"      # mode of all (errors on continuous)
            dict        # per-variable specification with "_default" key
            callable    # (data) -> 1-row representative DataFrame

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

        - ``"block_type"``: ``"moving"`` (default), ``"nonoverlapping"``, or ``"circular"``

    progress_bar : bool, default False
        Whether to show a progress bar during bootstrap execution.
        Requires ``tqdm`` to be installed.

    strict : bool, default False
        If True, no defaults are inferred — every analytical choice must be
        explicit at construction. Useful for pre-registered or audit-relevant
        work. Implementation detail: when strict, any unspecified config
        argument raises rather than using its default.

    adapter : ModelAdapter, optional
        Explicit adapter overriding auto-detection. Useful for unusual
        models or when customizing predict semantics.
    """

    _FROZEN_AFTER_CACHE = frozenset(
        {
            "method",
            "n_boot",
            "rng_seed",
            "n_sim",
            "cluster",
            "block_size",
            "bootstrap_config",
            "matching",
        }
    )

    def __setattr__(self, name, value):
        if name in self._FROZEN_AFTER_CACHE and self._any_cache_materialized():
            raise RuntimeError(
                f"Cannot mutate {name!r} after the inference cache has been "
                f"materialized. Construct a new Margins session."
            )
        super().__setattr__(name, value)

    def _any_cache_materialized(self):
        # Only inference-distribution caches trigger the freeze.
        # _cov_cache is set eagerly at construction but does not commit
        # the session to a specific bootstrap/simulation distribution.
        return any(
            hasattr(self, attr)
            for attr in (
                "_bootstrap_bank_cache",
                "_bootstrap_states_cache",
                "_simulation_draws_cache",
            )
        )

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        model,
        *,
        phi: Callable | None = _NOT_GIVEN,
        phi_inv: Callable | None = _NOT_GIVEN,
        vcov: str | np.ndarray | dict | None = _NOT_GIVEN,
        weights: np.ndarray | None = _NOT_GIVEN,
        at: AtSpec = _NOT_GIVEN,
        level: float = _NOT_GIVEN,
        method: InferenceMethod = _NOT_GIVEN,
        kappa_threshold: float = _NOT_GIVEN,
        rng_seed: int | None = _NOT_GIVEN,
        n_sim: int = _NOT_GIVEN,
        n_boot: int = _NOT_GIVEN,
        n_jobs: int = _NOT_GIVEN,
        gradient_backend: GradientBackend = _NOT_GIVEN,
        fd_step: float = _NOT_GIVEN,
        diagnostics: bool = _NOT_GIVEN,
        cluster: Any | None = _NOT_GIVEN,
        block_size: int | None = _NOT_GIVEN,
        bootstrap_config: dict | None = _NOT_GIVEN,
        progress_bar: bool = _NOT_GIVEN,
        matching: Any | None = _NOT_GIVEN,
        survey_design: Any | None = _NOT_GIVEN,
        formula: str | None = _NOT_GIVEN,
        data: pd.DataFrame | None = _NOT_GIVEN,
        strict: bool = False,
        adapter: ModelAdapter | None = None,
    ):
        # Strict mode: every config argument must be explicitly given
        if strict:
            for name, value in [
                ("phi", phi),
                ("phi_inv", phi_inv),
                ("vcov", vcov),
                ("weights", weights),
                ("at", at),
                ("level", level),
                ("method", method),
                ("kappa_threshold", kappa_threshold),
                ("rng_seed", rng_seed),
                ("n_sim", n_sim),
                ("n_boot", n_boot),
                ("n_jobs", n_jobs),
                ("gradient_backend", gradient_backend),
                ("fd_step", fd_step),
                ("diagnostics", diagnostics),
                ("cluster", cluster),
                ("block_size", block_size),
                ("bootstrap_config", bootstrap_config),
                ("progress_bar", progress_bar),
                ("matching", matching),
                ("survey_design", survey_design),
                ("formula", formula),
                ("data", data),
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
        gradient_backend = (
            "auto" if gradient_backend is _NOT_GIVEN else gradient_backend
        )
        fd_step = 1e-6 if fd_step is _NOT_GIVEN else fd_step
        diagnostics = True if diagnostics is _NOT_GIVEN else diagnostics
        cluster = None if cluster is _NOT_GIVEN else cluster
        block_size = None if block_size is _NOT_GIVEN else block_size
        bootstrap_config = None if bootstrap_config is _NOT_GIVEN else bootstrap_config
        progress_bar = False if progress_bar is _NOT_GIVEN else progress_bar
        matching = None if matching is _NOT_GIVEN else matching
        survey_design = None if survey_design is _NOT_GIVEN else survey_design
        formula = None if formula is _NOT_GIVEN else formula
        data = None if data is _NOT_GIVEN else data

        # Validation: phi/phi_inv must come as a pair
        if (phi is None) != (phi_inv is None):
            raise ValueError("phi and phi_inv must be provided together (or neither).")

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
        self.progress_bar = progress_bar
        self.matching = matching
        self.survey_design = survey_design
        self.strict = strict

        # Adapter setup
        if adapter is not None:
            self.adapter = adapter
        else:
            detect_kwargs = {}
            if formula is not None:
                detect_kwargs["formula"] = formula
            if data is not None:
                detect_kwargs["data"] = data
            self.adapter = auto_detect_adapter(model, **detect_kwargs)
        self.adapter.attach(self)

        # Validate weights (eager — must happen before any JAX tracing)
        if self.weights is not None:
            w_arr = np.asarray(self.weights)
            if not np.all(np.isfinite(w_arr)):
                raise ValueError("weights must be finite (no NaN or Inf).")
            if np.any(w_arr < 0):
                raise ValueError("weights must be non-negative.")
            if np.sum(w_arr) == 0:
                raise ValueError("weights must not sum to zero.")

        # Validate matching object
        if self.matching is not None:
            if not hasattr(self.matching, "matched_data"):
                raise ValueError(
                    "matching object must have a 'matched_data' attribute."
                )
            if not hasattr(self.matching, "cluster_ids"):
                raise ValueError("matching object must have a 'cluster_ids' attribute.")

            n_train = len(self.adapter.training_data)
            n_matched = len(self.matching.matched_data)
            if n_matched != n_train:
                raise ValueError(
                    f"matched_data length ({n_matched}) must equal training_data "
                    f"length ({n_train}). Fit the model on the matched sample, not "
                    f"the full sample."
                )

            cluster_arr = np.asarray(self.matching.cluster_ids)
            if len(cluster_arr) != n_matched:
                raise ValueError(
                    f"cluster_ids length ({len(cluster_arr)}) must equal "
                    f"matched_data length ({n_matched})."
                )
            if np.any(pd.isna(cluster_arr)):
                raise ValueError("cluster_ids must not contain NaN values.")

            # Auto-derive cluster for bootstrap if user did not supply one
            if self.cluster is None:
                self.cluster = cluster_arr

            # Auto-derive vcov if user did not supply one
            if self.vcov_spec is None:
                self.vcov_spec = {"type": "cluster", "groups": cluster_arr}

            # Warn if user supplied a non-cluster vcov
            if isinstance(self.vcov_spec, str):
                if self.vcov_spec.upper() not in {"CLUSTER", "CLUSTERED"}:
                    warnings.warn(
                        "matching was provided but vcov is not cluster-robust. "
                        "Standard errors may be anti-conservative because matched-set "
                        "dependence is ignored. Consider vcov='cluster' or omitting "
                        "vcov to use the matching object's cluster_ids.",
                        UserWarning,
                        stacklevel=2,
                    )

            # Validate that bootstrap users have rematch()
            if self.method == "bootstrap" and not callable(
                getattr(self.matching, "rematch", None)
            ):
                raise ValueError(
                    "method='bootstrap' with matching requires the matching object "
                    "to implement a 'rematch(data) -> DataFrame' method."
                )

            # Validate weights align with matched data
            if self.weights is not None:
                w_arr = np.asarray(self.weights)
                if len(w_arr) != n_matched:
                    raise ValueError(
                        f"weights length ({len(w_arr)}) must equal matched_data "
                        f"length ({n_matched}) when matching is provided."
                    )

        # Validate survey design
        if self.survey_design is not None:
            from ..survey import SurveyDesign
            if not isinstance(self.survey_design, SurveyDesign):
                raise TypeError(
                    "survey_design must be a pymargins.SurveyDesign instance."
                )
            try:
                n_data = len(self.adapter.training_data)
            except (NotImplementedError, AttributeError, TypeError):
                n_data = None
            if n_data is not None and len(self.survey_design.weights) != n_data:
                raise ValueError(
                    f"survey_design length ({len(self.survey_design.weights)}) "
                    f"must match training data length ({n_data})."
                )
            # Mutual exclusion with cluster/block (survey carries its own clustering)
            if self.cluster is not None or self.block_size is not None:
                raise ValueError(
                    "survey_design is mutually exclusive with cluster= and "
                    "block_size=; the design's PSU/strata define the resampling."
                )
            # Default vcov to the survey linearization unless user overrode it.
            if self.vcov_spec is None:
                self.vcov_spec = {"type": "survey", "design": self.survey_design}
            # Warn on double-counted weights (model already weighted at fit time).
            model = getattr(self.adapter, "results", None)
            fit_model = getattr(model, "model", None)
            for attr in ("var_weights", "freq_weights"):
                wv = getattr(fit_model, attr, None)
                if wv is not None and not np.allclose(np.asarray(wv), 1.0):
                    warnings.warn(
                        f"The fitted model carries non-trivial {attr}; combining it "
                        "with survey_design weights may double-count. Fit the model "
                        "unweighted and declare weights only via survey_design.",
                        UserWarning, stacklevel=2,
                    )
            # Design weights and aggregation weights are conceptually distinct.
            # Do NOT silently copy survey_design.weights into the aggregation slot;
            # the user must explicitly request a weighted point estimate.
            if self.weights is not None:
                if len(self.weights) != len(self.survey_design.weights):
                    raise ValueError(
                        f"weights length ({len(self.weights)}) != "
                        f"survey_design.weights length "
                        f"({len(self.survey_design.weights)})."
                    )
            else:
                warnings.warn(
                    "survey_design is set but weights= is not. The point estimate "
                    "will be unweighted. Pass weights=survey_design.weights "
                    "to obtain a population-weighted average marginal effect "
                    "(Stata svy: margins behavior).",
                    UserWarning, stacklevel=2,
                )
            # Fail fast if the adapter can't do linearization, pointing to bootstrap.
            if not hasattr(self.adapter, "score_obs"):
                if self.method != "bootstrap":
                    raise NotImplementedError(
                        f"{type(self.adapter).__name__} does not expose score_obs(), "
                        "so design-based (delta/simulation) survey SEs are "
                        "unavailable. Use method='bootstrap' with survey_design for "
                        "design-correct resampling instead."
                    )

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
    def linear_scale(cls, model, **kwargs) -> Margins:
        """Identity scale: contrasts are absolute differences. The default."""
        return cls(model, phi=None, phi_inv=None, **kwargs)

    @classmethod
    def from_formula(cls, model, formula: str, data: pd.DataFrame, **kwargs) -> Margins:
        """Construct a Margins session with an explicit formula.

        For models fit without native formula support (array-fit statsmodels,
        linearmodels), passing ``formula`` and ``data`` ensures that
        ``design_matrix_from_df`` re-evaluates derived terms (interactions,
        polynomials, splines) correctly. Without this, ``dydx()`` on variables
        involved in derived terms is silently incorrect.

        Parameters
        ----------
        model : fitted result object
            The fitted model to wrap.
        formula : str
            Formula string matching the model specification.
        data : pd.DataFrame
            Training data used to fit the model.
        **kwargs
            Forwarded to ``Margins.__init__``.

        Returns
        -------
        Margins
        """
        return cls(model, formula=formula, data=data, **kwargs)

    @classmethod
    def log_scale(cls, model, **kwargs) -> Margins:
        """Log scale: contrasts are log-ratios; reported as ratios.

        Use for relative risks, fold-change, rate ratios, hazard ratios.
        Predictions reported with asymmetric CIs naturally bounded by zero.
        """
        return cls(model, phi=jnp.exp, phi_inv=jnp.log, **kwargs)

    @classmethod
    def logit_scale(cls, model, **kwargs) -> Margins:
        """Logit scale: contrasts are log-odds-ratios; reported as ORs.

        Predictions reported as probabilities with asymmetric CIs naturally
        bounded in (0, 1).
        """
        from jax.scipy.special import expit, logit

        return cls(model, phi=expit, phi_inv=logit, **kwargs)

    @classmethod
    def correlation_scale(cls, model, **kwargs) -> Margins:
        """Fisher z scale: contrasts on z; reported as correlations."""
        return cls(model, phi=jnp.tanh, phi_inv=jnp.arctanh, **kwargs)

    @classmethod
    def from_posterior(
        cls,
        model,
        posterior_draws: np.ndarray,
        *,
        point_estimate: np.ndarray | None = None,
        **kwargs,
    ) -> Margins:
        """Construct a Margins session from posterior draws of β.

        The simulation inference path uses the user-supplied draws instead
        of normal draws around β̂. Useful for PyMC, NumPyro, Stan, etc.

        Parameters
        ----------
        posterior_draws : array of shape (n_draws, p)
        point_estimate : array of shape (p,), optional
            Reported estimate. Defaults to posterior mean.
        **kwargs
            Forwarded to ``Margins.__init__``. ``method`` and ``n_sim``
            default to ``"simulation"`` and ``n_draws``.
        """
        kwargs.setdefault("method", "simulation")
        kwargs.setdefault("n_sim", posterior_draws.shape[0])
        self = cls(model, **kwargs)

        draws = np.asarray(posterior_draws)
        if draws.shape[0] > 1:
            self._cov_cache = np.cov(draws.T)

        cov = self._cov_cache
        parts = [
            str(self.rng_seed),
            str(self.n_sim),
            hashlib.sha256(np.asarray(cov).tobytes()).hexdigest()[:16],
        ]
        bank_key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
        self._simulation_draws_cache = (bank_key, draws)

        if point_estimate is not None:
            pe = np.asarray(point_estimate)
            self.adapter._posterior_point_estimate = pe
            orig_coeff = self.adapter.coefficients
            self.adapter._orig_coefficients = orig_coeff
            self.adapter.coefficients = lambda: jnp.asarray(pe)

        return self

    # -----------------------------------------------------------------------
    # Core entry points
    # -----------------------------------------------------------------------

    def predict(
        self,
        *,
        atexog: dict | pd.DataFrame | None = None,
        over: str | list[str] | None = None,
        transform: Callable | None = None,
        label: str | None = None,
        outcome: int | list[int] | None = None,
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
        if transform is not None:
            if not callable(transform):
                raise TypeError(
                    f"transform must be callable, got {type(transform).__name__}"
                )
            _validate_callable_arity(transform, "transform", min_positional=1)
        if atexog is not None and hasattr(atexog, "iloc"):
            scenario = {"data": atexog, "over": over, "label": label}
        else:
            scenario = {"atexog": atexog, "over": over, "label": label}
        h, labels, scenarios = self._build_prediction_estimand(scenario, transform)
        config = self._inference_config()
        meta = {"kind": "prediction", "at": self.at}
        if label is not None:
            if labels is None or (isinstance(labels, list) and len(labels) == 1):
                labels = [label]
            else:
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

            def h_factory(new_adapter):
                return self._build_prediction_estimand(
                    scenario, transform, adapter=new_adapter
                )[0]

        result_data = run_inference(
            h,
            self.adapter,
            config,
            estimand_metadata=meta,
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    def dydx(
        self,
        variables: str | list[str],
        *,
        atexog: dict | pd.DataFrame | None = None,
        over: str | list[str] | None = None,
        transform: Callable | None = None,
        label: str | None = None,
        outcome: int | list[int] | None = None,
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
        if transform is not None:
            if not callable(transform):
                raise TypeError(
                    f"transform must be callable, got {type(transform).__name__}"
                )
            _validate_callable_arity(transform, "transform", min_positional=1)
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
        meta = {"kind": "slope", "variables": var_list, "at": self.at}
        if label is not None:
            if labels is None or (isinstance(labels, list) and len(labels) == 1):
                labels = [label]
            else:
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

            def h_factory(new_adapter):
                return self._build_slope_estimand(
                    scenario, var_list, transform, adapter=new_adapter
                )[0]

        result_data = run_inference(
            h,
            self.adapter,
            config,
            estimand_metadata=meta,
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    def eyex(self, variable: str, **kwargs) -> MarginsResult:
        """Elasticity: (∂y/∂x) * x / y, evaluated at the session's ``at``."""
        return self._elasticity(variable, kind="eyex", **kwargs)

    def eydx(self, variable: str, **kwargs) -> MarginsResult:
        """Semi-elasticity: (∂y/∂x) / y."""
        return self._elasticity(variable, kind="eydx", **kwargs)

    def dyex(self, variable: str, **kwargs) -> MarginsResult:
        """Semi-elasticity: (∂y/∂x) * x."""
        return self._elasticity(variable, kind="dyex", **kwargs)

    def wtp(
        self,
        attribute: str,
        price: str,
        **kwargs,
    ) -> MarginsResult:
        """Willingness to pay: -(∂U/∂attribute) / (∂U/∂price).

        Parameters
        ----------
        attribute : str
            Continuous attribute variable.
        price : str
            Continuous price / cost variable.
        **kwargs
            Forwarded to the underlying ``dydx`` calls (``atexog``,
            ``over``, ``outcome``, etc.).

        Returns
        -------
        MarginsResult
            The WTP ratio with proper joint inference via the delta method.

        Notes
        -----
        For multi-outcome models (e.g. MNLogit), slice the result to a
        single outcome first: ``m.dydx("price").outcome(k)``.
        """
        slope_attr = self.dydx(attribute, **kwargs)
        slope_price = self.dydx(price, **kwargs)
        return compose_results(
            [slope_attr, slope_price],
            fn=lambda t: -t[0] / t[1],
            label=f"WTP({attribute})",
        )

    def _elasticity(self, variable, *, kind, clip_near_zero=1e-12, **kwargs):
        slope = self.dydx(variable, **kwargs)
        pred_kwargs = {k: v for k, v in kwargs.items() if k != "label"}
        pred = self.predict(**pred_kwargs)
        base = self._base_data
        if hasattr(base, "__getitem__") and variable in base.columns:
            x_bar = float(np.asarray(base[variable]).mean())
        else:
            raise ValueError(
                f"Cannot compute elasticity: variable {variable!r} not found "
                f"in training data columns."
            )
        fn = {
            "eyex": lambda t: (
                t[0]
                * x_bar
                / jnp.where(
                    jnp.abs(t[1]) < clip_near_zero,
                    jnp.sign(t[1]) * clip_near_zero,
                    t[1],
                )
            ),
            "eydx": lambda t: (
                t[0]
                / jnp.where(
                    jnp.abs(t[1]) < clip_near_zero,
                    jnp.sign(t[1]) * clip_near_zero,
                    t[1],
                )
            ),
            "dyex": lambda t: t[0] * x_bar,
        }[kind]
        return compose_results([slope, pred], fn=fn, label=f"{kind}({variable})")

    def contrasts(
        self,
        *,
        scenarios: list[dict],
        contrasts: ContrastSpec,
        outcome: int | list[int] | None = None,
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
        elif (
            isinstance(contrasts, list) and contrasts and isinstance(contrasts[0], list)
        ):
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

            def h_factory(new_adapter):
                return self._build_contrast_estimand(
                    scenarios, weights_arg, adapter=new_adapter
                )

        result_data = run_inference(
            h,
            self.adapter,
            config,
            estimand_metadata={
                "kind": "contrasts",
                "labels": labels,
                "scenarios": scenarios,
                "at": self.at,
            },
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
        outcome: int | list[int] | None = None,
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
        _validate_callable_arity(compose, "compose", min_positional=1)
        h = self._build_evaluate_estimand(scenarios, compose)
        config = self._inference_config()

        labels = [
            s.get("label", _auto_label_from_atexog(s.get("atexog")) or f"scenario[{i}]")
            for i, s in enumerate(scenarios)
        ]
        h_factory = None
        if config.method == "bootstrap":

            def h_factory(new_adapter):
                return self._build_evaluate_estimand(
                    scenarios, compose, adapter=new_adapter
                )

        result_data = run_inference(
            h,
            self.adapter,
            config,
            estimand_metadata={
                "kind": "evaluate",
                "labels": labels,
                "scenarios": scenarios,
                "at": self.at,
            },
            h_factory=h_factory,
        )
        if outcome is not None and self.adapter.n_outcomes > 1:
            result_data = self._slice_by_outcome(result_data, outcome)
        return self._wrap_result(result_data)

    def rmst(
        self,
        *,
        horizon: float,
        atexog: dict | None = None,
        over: str | list[str] | None = None,
        n_grid: int = 80,
    ) -> MarginsResult:
        """Restricted mean survival time up to ``horizon``.

        Computed by trapezoidal integration of the survival function on
        ``n_grid`` evenly spaced times in [0, ``horizon``].

        Parameters
        ----------
        horizon : float
            Upper time limit for integration.
        atexog : dict, optional
            Exogenous variable values for the scenario.
        over : str or list of str, optional
            Subgroup variable(s).
        n_grid : int, default 80
            Number of grid points for integration.

        Returns
        -------
        MarginsResult
        """
        if not hasattr(self.adapter, "with_prediction_time"):
            raise ValueError(
                f"Adapter {type(self.adapter).__name__} does not support "
                f"per-scenario prediction time; rmst() requires a "
                f"time-aware survival adapter."
            )
        times = np.linspace(0.0, float(horizon), int(n_grid))
        scenarios = [
            {"atexog": atexog, "over": over, "prediction_time": float(t)} for t in times
        ]
        times_jax = jnp.asarray(times)

        def trapz(s):
            dt = jnp.diff(times_jax)
            return jnp.sum(0.5 * (s[:-1] + s[1:]) * dt)

        return self.evaluate(scenarios=scenarios, compose=trapz)

    # -----------------------------------------------------------------------
    # Outcome slicing for multi-outcome models
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Diagnostics and reporting
    # -----------------------------------------------------------------------

    def diagnose(
        self,
        n_samples: int = 50,
        rng_seed: int | None = None,
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
            self.adapter.design_matrix_from_df(base.iloc[[i]])[0] for i in sample_idx
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
            h_factory,
            beta,
            Sigma,
            sample_X,
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
        return SummaryString(
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
        return self._get_base_data(self.adapter)

    # -----------------------------------------------------------------------
    # Re-bound helpers from sibling modules
    # -----------------------------------------------------------------------

    _get_base_data = _get_base_data
    _build_prediction_estimand = _build_prediction_estimand
    _build_slope_estimand = _build_slope_estimand
    _build_contrast_estimand = _build_contrast_estimand
    _build_evaluate_estimand = _build_evaluate_estimand
    _enumerate_groups = _enumerate_groups
    _slice_by_outcome = _slice_by_outcome
    _inference_config = _inference_config
    _frozen_cov = _frozen_cov
    _wrap_result = _wrap_result
    _format_atom_label = staticmethod(_format_atom_label)
    _finalize_atoms = staticmethod(_finalize_atoms)
