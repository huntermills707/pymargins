"""
pymargins._adapter

Adapter layer between fitted models from various frameworks (statsmodels,
linearmodels, sklearn) and the inference engine. Adapters are the only
framework-aware code in the library; everything downstream operates through
the uniform ModelAdapter interface.

Adapter responsibilities
------------------------
1. Expose β̂ as a JAX array (`coefficients`)
2. Expose Σ̂ as a JAX array, with support for robust/clustered flavors
   (`covariance`)
3. Provide a JAX-compatible predict function (`predict`) — either via JAX
   reimplementation, custom JVP using the framework's analytical derivative,
   or custom JVP wrapping a non-differentiable predict via FD
4. Build design matrices from DataFrames (`design_matrix_from_df`)
5. Expose variable type metadata for at=typical averaging (`variable_metadata`)
6. Declare what inference methods this model class supports
   (`supported_inference_methods`)
7. Optionally support refit-on-resampled-data for bootstrap (`refit`)

Adapter shapes
--------------
The library ships several adapter base classes corresponding to common
"shapes" of derivative structure:

  GLMAdapter           — for any model with prediction μ = g⁻¹(Xβ).
                         One adapter parameterized by the family object
                         covers all GLM families and links.

  LinearPredictionAdapter — for OLS, WLS, GLS, IV, panel models. Prediction
                            is Xβ; complexity is in Σ̂ and scenario semantics.
                            No JVP wrapper needed (Path A trivially).

  WrappedFDAdapter     — for any model with a smooth predict but no exposed
                         analytical derivative. Wraps predict via FD-JVP.
                         Universal fallback.

  BootstrapOnlyAdapter — for non-parametric / algorithmic models with no
                         meaningful Σ̂ (random forest, gradient boosting,
                         neural net). Declares only bootstrap support.

Each concrete framework adapter (StatsmodelsGLMAdapter, LinearmodelsPanelAdapter,
SklearnLinearAdapter, etc.) inherits from one of these shapes and fills in
framework-specific details: how to extract β̂, how to request a vcov flavor,
how to build a design matrix from formulae or feature lists.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp

if TYPE_CHECKING:
    import pandas as pd

import numpy as np

from ._gradients import GradientBackend
from ._tabular import fingerprint_frame

# ---------------------------------------------------------------------------
# Type aliases and small dataclasses
# ---------------------------------------------------------------------------

InferenceMethod = Literal["delta", "simulation", "bootstrap"]
VariableType = Literal["continuous", "binary", "categorical"]


@dataclass
class VariableInfo:
    """Per-variable metadata used for at=typical averaging, contrast
    validation, and warning generation.

    Attributes
    ----------
    name : str
        Variable name as used in the model's formula or feature list.
    var_type : str
        One of "continuous", "binary", "categorical".
    levels : list, optional
        For categorical/binary variables, the set of valid level values.
        For ordered factors, the order is meaningful.
    support : tuple, optional
        For continuous variables, (min, max) observed in the fit data.
        Useful for warning when atexog= specifies a value far outside the
        observed range.
    encoding : str, optional
        How the variable is encoded in the design matrix (e.g.,
        "treatment_coded", "polynomial", "one_hot"). Affects how
        scenario specifications map to design columns.
    """

    name: str
    var_type: VariableType
    levels: list | None = None
    support: tuple[float, float] | None = None
    encoding: str | None = None


# ---------------------------------------------------------------------------
# Base adapter interface
# ---------------------------------------------------------------------------


class ModelAdapter(abc.ABC):
    """Abstract base class for model adapters.

    Concrete subclasses target specific frameworks (statsmodels,
    linearmodels, sklearn) and possibly specific model classes within them.
    The Margins session calls only the methods defined here; all framework-
    specific knowledge is encapsulated in the adapter.

    Subclassing contract
    --------------------
    Subclasses MUST implement:
      - coefficients()
      - covariance()
      - predict()
      - design_matrix_from_df()
      - column_index_of_variable()
      - variable_metadata()
      - training_data (property)
      - supported_inference_methods (property)

    Subclasses MAY implement:
      - attach() to validate session compatibility
      - refit() if bootstrap inference should be supported
      - native_predict() if direct framework calls are needed elsewhere
    """

    # -----------------------------------------------------------------------
    # Capability declaration
    # -----------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def supports_jax_autodiff(self) -> bool:
        """True if predict() uses jax.numpy throughout, producing exact
        autodiff gradients. False if predict() wraps non-JAX code via a
        custom JVP using FD or analytical derivatives.

        The library's gradient layer treats both as "autodiff" because
        jax.grad works through both — the distinction is informational,
        used in result metadata and diagnostic messages.
        """
        ...

    @property
    @abc.abstractmethod
    def supported_inference_methods(self) -> set[InferenceMethod]:
        """Which inference methods this adapter supports.

        Returned as a set; the inference engine checks the user's requested
        method against this set and raises a clear error if unsupported.
        Typical values:
          {"delta", "simulation", "bootstrap"}  — full support (most GLMs)
          {"simulation", "bootstrap"}            — no analytical gradients
          {"bootstrap"}                          — algorithmic models
        """
        ...

    @property
    @abc.abstractmethod
    def gradient_backend_recommendation(self) -> GradientBackend:
        """The gradient backend the engine should use by default with this
        adapter. Subclasses override to indicate "autodiff" (clean JAX
        path) or "wrapped_fd" (wrapped predict). The engine respects user
        overrides via the session's gradient_backend argument."""
        ...

    # -----------------------------------------------------------------------
    # Session integration
    # -----------------------------------------------------------------------

    def attach(self, session: Any) -> None:
        """Attach this adapter to a Margins session. Receive the session's
        configuration (scale, vcov_spec, weights, etc.) and validate
        compatibility.

        Subclasses should raise ValueError or NotImplementedError with
        a clear message if the session's configuration is not supportable.
        For example, a survival adapter that doesn't support log scale
        would refuse a session with phi=exp.

        Base implementation validates that ``phi`` and ``phi_inv`` are
        approximate inverses when both are provided. Subclasses that
        override this method should call ``super().attach(session)`` to
        preserve this check.

        Parameters
        ----------
        session : Margins
            The session attaching this adapter.
        """
        phi = getattr(session, "phi", None)
        phi_inv = getattr(session, "phi_inv", None)
        if phi is not None and phi_inv is not None:
            test_val = jnp.array(0.5)
            try:
                recon = float(phi(phi_inv(test_val)))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"phi/phi_inv validation failed at test point {float(test_val)}: {exc}"
                ) from exc
            if not np.isclose(recon, float(test_val), rtol=1e-4):
                raise ValueError(
                    f"phi and phi_inv do not appear to be inverses: "
                    f"phi(phi_inv({float(test_val)})) = {recon}"
                )

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    @abc.abstractmethod
    def coefficients(self) -> jnp.ndarray:
        """Return β̂ as a 1D JAX array.

        For models with multiple coefficient blocks (e.g., multinomial
        logit), this returns the flattened parameter vector; the predict
        method is responsible for un-flattening internally.

        Returns
        -------
        beta_hat : jax array of shape (n_params,)
        """
        ...

    @abc.abstractmethod
    def covariance(
        self,
        vcov_spec: Any | None = None,
    ) -> jnp.ndarray:
        """Return Σ̂ as a 2D JAX array.

        Parameters
        ----------
        vcov_spec : optional
            Specification for which Σ̂ flavor to return. Format depends on
            the framework:
              - None: framework default (typically OIM or expected info)
              - "HC0", "HC1", "HC2", "HC3": robust to heteroskedasticity
              - dict like {"type": "cluster", "groups": cluster_ids}: cluster-
                robust
              - 2D array: user-supplied Σ̂ (overrides any framework default)

            Adapters validate the spec against what their framework supports
            and raise ValueError with a clear message if unsupported.

        Returns
        -------
        Sigma_hat : jax array of shape (n_params, n_params)
        """
        ...

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    @abc.abstractmethod
    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Prediction on the response scale, JAX-compatible.

        This is the central differentiable primitive of the adapter. The
        inference engine calls jax.grad(predict, argnums=0) and
        jax.jacobian(predict, argnums=1) on this function (the latter for
        ∂μ/∂x, used by dydx slopes).

        Implementations either:
          (a) Use jax.numpy operations natively for an exact-autodiff path
          (b) Wrap a non-JAX predict via custom_jvp using analytical
              derivatives (e.g., link.inverse_deriv for GLMs)
          (c) Wrap a non-JAX predict via custom_jvp using FD

        All three paths present the same interface to the engine.

        Parameters
        ----------
        beta : jax array of shape (n_params,)
            Parameter vector at which to predict. Typically β̂ but may be
            a perturbed value during gradient computation.

        X : jax array of shape (n_obs, n_features)
            Design matrix.

        offset : jax array of shape (n_obs,), optional
            Offset added to the linear predictor before applying the link
            inverse. None for models without offset support.

        Returns
        -------
        mu : jax array of shape (n_obs,) or (n_obs, n_outcomes)
            Predicted values on the response scale. For single-output models
            returns a 1D array; for multi-outcome models (e.g. MNLogit,
            OrderedModel) returns a 2D array with one column per outcome
            class.
        """
        ...

    # -----------------------------------------------------------------------
    # Design and metadata
    # -----------------------------------------------------------------------

    @abc.abstractmethod
    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        """Build a design matrix from a concrete DataFrame of evaluation rows.

        Handles formula expansion, factor encoding, interactions, splines,
        and any other transformations the model fit applied to the original
        variables. The output is suitable for direct use in predict().

        Parameters
        ----------
        df : pandas DataFrame
            Concrete evaluation rows produced by _scenarios.expand_scenario.
            Column names correspond to variable names in the model.

        Returns
        -------
        X : jax array of shape (n_rows, n_features)
            Design matrix for the scenario.
        """
        ...

    @property
    def training_data(self):
        """The training data used to fit the model.

        Required for diagnose() and for scenario expansion when the session's
        `at` setting is "overall". Adapters should expose this attribute
        or override Margins._base_data.

        Returns
        -------
        data : pandas DataFrame or similar
            The data the model was fit on.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose training_data. "
            "Either set self.training_data in __init__, or override "
            "Margins._base_data."
        )

    @property
    def n_outcomes(self) -> int:
        """Number of outcome classes for multi-outcome models, default 1."""
        return 1

    @property
    def outcome_labels(self) -> list[str] | None:
        """Outcome class labels for multi-outcome models, or None."""
        return None

    @abc.abstractmethod
    def column_index_of_variable(self, name: str) -> int:
        """Return the design-matrix column index corresponding to a variable.

        Used by ``dydx()`` to compute slopes ∂μ/∂x_j. Only meaningful for
        continuous variables that map to a single design column. For
        categorical or factor-expanded variables this should raise
        ``ValueError`` — slope is undefined for such variables, and
        ``Margins.dydx`` validates ``var_type`` before calling this.

        Parameters
        ----------
        name : str
            Variable name as it appears in ``variable_metadata()`` and in
            user-facing scenario specs.

        Returns
        -------
        index : int
            Zero-based column index in the design matrix produced by
            ``design_matrix_from_df``.
        """
        ...

    @abc.abstractmethod
    def variable_metadata(self) -> dict[str, VariableInfo]:
        """Return per-variable metadata used by averaging and validation.

        Returns
        -------
        metadata : dict
            Mapping from variable name (as used in scenarios) to VariableInfo.
            Should include all variables in the model's design.
        """
        ...

    # -----------------------------------------------------------------------
    # Influence function (optional)
    # -----------------------------------------------------------------------

    def influence(self) -> jnp.ndarray | None:
        """Per-observation influence of β̂.

        For sources with per-row scores (tier-1), returns ψ^β where row *i*
        is ``score_obs[i] @ covariance().T``, shape ``(n_obs, n_params)``.
        Rows are aligned to :attr:`training_data`'s index.

        ``None`` means this source has no per-row score (tier 2/3).
        """
        return None

    def data_fingerprint(self) -> str:
        """Content hash of :attr:`training_data`.

        Used for template-vs-wiring consistency checks and plan identity.
        The result is cached on the instance.
        """
        if hasattr(self, "_data_fingerprint"):
            return self._data_fingerprint
        self._data_fingerprint = fingerprint_frame(self.training_data)
        return self._data_fingerprint

    # -----------------------------------------------------------------------
    # Bootstrap support (optional)
    # -----------------------------------------------------------------------

    def refit(self, resampled_data, *, index=None) -> ModelAdapter:
        """Refit the model on resampled data, returning a new adapter.

        Required for bootstrap inference. Implementations should re-run the
        model's fitting routine on resampled_data and return a new adapter
        wrapping the new fit. The original adapter is not modified.

        Default implementation raises NotImplementedError; adapters that
        can't support refit (external/cloud-fitted models, models requiring
        special data structures) should leave it raising and rely on
        bootstrap_supported being False.

        Parameters
        ----------
        resampled_data : framework-specific
            Resampled training data. Format depends on the framework: pandas
            DataFrame for statsmodels formula API, NumPy arrays for direct
            API, etc.
        index : array-like of int, optional
            The index array used to produce ``resampled_data`` from the
            original training data. Adapters that store external arrays
            (offset, exposure, weights) alongside the data should use this
            to resample those arrays so they align with ``resampled_data``.

        Returns
        -------
        new_adapter : ModelAdapter
            Adapter wrapping the model refit on resampled_data.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support refit; "
            "bootstrap inference is unavailable for this adapter."
        )

    def bootstrap_state(self) -> ModelAdapter:
        """Replay state for a refitted adapter.

        Returning ``self`` is correct — a refitted adapter is itself a
        valid state. Override only when (a) you want a slimmer or more
        picklable representation, or (b) the framework's refit retains
        mutable state you'd rather drop before caching.
        """
        return self


# ---------------------------------------------------------------------------
# Adapter shapes (intermediate base classes)
# ---------------------------------------------------------------------------


class GLMAdapter(ModelAdapter):
    """Base adapter for any model with prediction μ = g⁻¹(Xβ + offset).

    Covers all GLM families uniformly. Concrete subclasses for statsmodels'
    GLM and similar frameworks fill in framework-specific extraction.

    Implementations of `predict` typically use either:
      - jax-native: write the link inverse in JAX (logit, log, identity,
        probit are all available in jax.scipy.special)
      - custom JVP with link.inverse_deriv: use _gradients.make_glm_jvp_wrapper

    Both produce identical numerical results; choose based on the discussion
    in the design docs.
    """

    def attach(self, session: Any) -> None:
        super().attach(session)

    @property
    def supports_jax_autodiff(self) -> bool:
        """GLM adapters typically use JAX-native link inverses or analytical JVP."""
        return True  # Either native JAX or analytical-derivative JVP both
        # appear as autodiff to downstream consumers

    @property
    def supported_inference_methods(self) -> set[InferenceMethod]:
        """GLM adapters support delta, simulation, and bootstrap inference."""
        return {"delta", "simulation", "bootstrap"}

    @property
    def gradient_backend_recommendation(self) -> GradientBackend:
        """Recommend pure autodiff for GLM adapters."""
        return "autodiff"


class LinearPredictionAdapter(ModelAdapter):
    """Base for models whose prediction is exactly Xβ (no link function).

    Covers OLS, WLS, GLS, IV, panel models. predict() is trivially Path A:
        def predict(self, beta, X, offset=None):
            return X @ beta + (offset if offset is not None else 0.0)

    The complexity for these adapters lies in covariance(): handling the
    framework's various vcov flavors (HC, cluster, HAC, GMM-style adjustments
    for IV, etc.).
    """

    @property
    def supports_jax_autodiff(self) -> bool:
        """Linear prediction is trivially JAX-differentiable (X @ beta)."""
        return True

    @property
    def supported_inference_methods(self) -> set[InferenceMethod]:
        """Linear adapters support delta, simulation, and bootstrap inference."""
        return {"delta", "simulation", "bootstrap"}

    @property
    def gradient_backend_recommendation(self) -> GradientBackend:
        """Recommend pure autodiff for linear adapters."""
        return "autodiff"

    @staticmethod
    def predict(
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Linear prediction: X @ beta + offset."""
        eta = X @ beta
        if offset is not None:
            eta = eta + offset
        return eta


class WrappedFDAdapter(ModelAdapter):
    """Base for models with smooth predict but no exposed analytical derivative.

    Uses _gradients.make_predict_with_fd_jvp to wrap the framework's native
    predict with a JAX-compatible custom JVP. FD is hidden inside the JVP;
    downstream autodiff over the estimand structure remains exact.

    Concrete subclasses provide `native_predict(beta_np, X)` and the wrapper
    is constructed automatically.
    """

    @property
    def supports_jax_autodiff(self) -> bool:
        """FD-based JVP is not native autodiff; flag for diagnostics."""
        return False  # The JVP uses FD; flag this for diagnostic context

    @property
    def supported_inference_methods(self) -> set[InferenceMethod]:
        """Wrapped-FD adapters support delta, simulation, and bootstrap."""
        return {"delta", "simulation", "bootstrap"}

    @property
    def gradient_backend_recommendation(self) -> GradientBackend:
        """Recommend wrapped_fd because predict uses finite-difference JVP."""
        return "wrapped_fd"

    def native_predict(self, beta_np: np.ndarray, X) -> np.ndarray:
        """Framework-native predict. Receives NumPy beta and returns NumPy
        predictions. Implemented by subclass.
        """
        raise NotImplementedError

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """FD-wrapped predict. Thread-safe lazy construction of JAX primitive."""
        if offset is not None:
            raise ValueError(
                "WrappedFDAdapter does not support offset in the base "
                "implementation. Subclasses should override predict() to "
                "handle offset."
            )
        # Thread-safe lazy build: double-checked locking with a per-instance
        # lock created on demand. The wrapper is functionally pure (same
        # native_predict and fd_step), so even if two threads race, both
        # produce identical wrappers; the lock merely prevents redundant work.
        if not hasattr(self, "_predict_wrapped"):
            import threading

            lock = getattr(self, "_predict_wrapped_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._predict_wrapped_lock = lock
            with lock:
                if not hasattr(self, "_predict_wrapped"):
                    from ._gradients import make_predict_with_fd_jvp

                    fd_step = getattr(self, "_fd_step", 1e-6)
                    self._predict_wrapped = make_predict_with_fd_jvp(
                        self.native_predict,
                        fd_step=fd_step,
                    )
        return self._predict_wrapped(beta, X)


class BootstrapOnlyAdapter(ModelAdapter):
    """Base for non-parametric / algorithmic models with no meaningful Σ̂.

    Covers tree ensembles, kNN, neural networks, and other algorithmic
    estimators. Declares only bootstrap support; the inference engine
    routes all requests for these models through refit-and-recompute.

    Subclasses must implement refit().
    """

    @property
    def supports_jax_autodiff(self) -> bool:
        """Bootstrap-only adapters have no parametric predict; no autodiff."""
        return False

    @property
    def supported_inference_methods(self) -> set[InferenceMethod]:
        """Bootstrap-only adapters declare only bootstrap support."""
        return {"bootstrap"}

    @property
    def gradient_backend_recommendation(self) -> GradientBackend:
        """Not used (no delta path); declared for completeness."""
        return "fd"

    def coefficients(self) -> jnp.ndarray:
        """Raise: bootstrap-only models have no meaningful coefficient vector."""
        raise NotImplementedError(
            "BootstrapOnlyAdapter has no meaningful coefficient vector; "
            "use refit-based bootstrap inference instead."
        )

    def covariance(self, vcov_spec=None) -> jnp.ndarray:
        """Raise: bootstrap-only models have no meaningful covariance."""
        raise NotImplementedError(
            "BootstrapOnlyAdapter has no meaningful covariance; "
            "use refit-based bootstrap inference instead."
        )

    def predict(self, beta, X, offset=None) -> jnp.ndarray:
        """Raise: bootstrap-only models do not provide parametric predict."""
        raise NotImplementedError(
            "BootstrapOnlyAdapter does not provide a parametric predict; "
            "use refit-based bootstrap inference instead."
        )


# ---------------------------------------------------------------------------
# Auto-detection of adapter from a fitted model
# ---------------------------------------------------------------------------


def auto_detect_adapter(model, formula=None, data=None) -> ModelAdapter:
    """Inspect a fitted model and return an appropriate adapter.

    Used by GComputation() when no adapter is explicitly provided. Delegates to
    the concrete dispatch table in _adapters to keep framework imports lazy.

    Parameters
    ----------
    model : fitted result object from any supported framework
    formula : str, optional
        Formula string for models that were fit without native formula support.
    data : pd.DataFrame, optional
        Training data corresponding to ``formula``.

    Returns
    -------
    adapter : ModelAdapter
        Appropriate adapter for the model.

    Raises
    ------
    NotImplementedError
        If no adapter is registered for the model's class.
    """
    from ._adapters import auto_detect_adapter as _auto_detect

    return _auto_detect(model, formula=formula, data=data)


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example 1: Implementing a statsmodels GLM adapter
-------------------------------------------------

    import jax.numpy as jnp
    import numpy as np
    from pymargins._adapter import GLMAdapter, VariableInfo
    from pymargins._gradients import make_glm_jvp_wrapper

    class StatsmodelsGLMAdapter(GLMAdapter):
        def __init__(self, results):
            self.results = results
            self._predict_jax = make_glm_jvp_wrapper(results.family)

        def coefficients(self):
            return jnp.asarray(self.results.params)

        def covariance(self, vcov_spec=None):
            if vcov_spec is None:
                return jnp.asarray(self.results.cov_params())
            elif vcov_spec in ("HC0", "HC1", "HC2", "HC3"):
                # Refit with cov_type, or use the existing if compatible
                return jnp.asarray(self.results.cov_HC0)  # etc.
            else:
                raise ValueError(f"Unsupported vcov: {vcov_spec}")

        def predict(self, beta, X, offset=None):
            return self._predict_jax(beta, X, offset=offset)

        def design_matrix_from_df(self, df):
            # Use patsy/formulaic to build design matrix from DataFrame
            ...

        def variable_metadata(self):
            ...

        def refit(self, resampled_data):
            new_results = self.results.model.__class__(...).fit()
            return StatsmodelsGLMAdapter(new_results)


Example 2: Wrapping a black-box predict with FD
-----------------------------------------------

    from pymargins._adapter import WrappedFDAdapter

    class StatsmodelsMixedLMAdapter(WrappedFDAdapter):
        def __init__(self, results):
            self.results = results
            self._fd_step = 1e-6

        def coefficients(self):
            return jnp.asarray(self.results.fe_params)

        def covariance(self, vcov_spec=None):
            return jnp.asarray(self.results.cov_params())

        def native_predict(self, beta_np, X):
            # statsmodels MixedLM doesn't have a clean differentiable predict;
            # use its native predict, which involves random-effects machinery
            return self.results.predict(...)  # framework call

        def design_matrix_from_df(self, df): ...
        def variable_metadata(self): ...


Example 3: A bootstrap-only adapter for sklearn random forest
-------------------------------------------------------------

    from pymargins._adapter import BootstrapOnlyAdapter
    from sklearn.ensemble import RandomForestRegressor

    class SklearnTreeAdapter(BootstrapOnlyAdapter):
        def __init__(self, model, X_train, y_train):
            self.model = model
            self.X_train = X_train
            self.y_train = y_train

        def design_matrix_from_df(self, df):
            # Build feature matrix from DataFrame
            ...

        def variable_metadata(self):
            ...

        def refit(self, resampled_data):
            X_new, y_new = resampled_data
            new_model = RandomForestRegressor(**self.model.get_params())
            new_model.fit(X_new, y_new)
            return SklearnTreeAdapter(new_model, X_new, y_new)
"""
