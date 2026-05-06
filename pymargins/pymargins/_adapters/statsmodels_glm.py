"""
pymargins._adapters.statsmodels_glm

Concrete adapter for statsmodels GLM result objects. Serves as the
reference implementation for the adapter pattern.

This adapter covers all standard GLM families (Binomial, Poisson, Gaussian,
Gamma, InverseGaussian, NegativeBinomial, Tweedie) across all standard
links (logit, probit, log, identity, power, inverse, cloglog). One adapter
class for the entire GLM family because the chain-rule structure is uniform:
the family's `link.inverse_deriv` provides the analytical derivative
factor, and the rest of the gradient machinery is the same.

Status
------
SKELETON — most methods need filling in. The shape is correct; the work
left is framework-specific extraction:
  - design_matrix_from_df via patsy/formulaic
  - vcov flavor dispatch (HC0/HC3/cluster) against statsmodels' machinery
  - variable_metadata extraction from results.model.exog_names and the
    formula's term info
  - column_index_of_variable mapping

Implementers: see IMPLEMENTATION_GUIDE.md for prioritized tasks.
"""

from __future__ import annotations
from typing import Optional, Any
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .._adapter import GLMAdapter, VariableInfo
from .._gradients import make_glm_jvp_wrapper


class StatsmodelsGLMAdapter(GLMAdapter):
    """Adapter for statsmodels.genmod.generalized_linear_model.GLMResults.

    Parameters
    ----------
    results : GLMResults
        Fitted statsmodels GLM result object.

    training_data : pd.DataFrame, optional
        The data the model was fit on. statsmodels exposes this via
        results.model.data.frame for formula-fit models, but not always
        for direct-array fits — provide explicitly in that case.
    """

    def __init__(self, results, training_data: Optional[pd.DataFrame] = None):
        self.results = results
        self.family = results.family

        # Build the JAX-compatible predict using analytical link derivative
        self._predict_jax = make_glm_jvp_wrapper(self.family)

        # Training data: prefer explicit, fall back to model attribute
        if training_data is not None:
            self.training_data = training_data
        elif hasattr(results.model, "data") and hasattr(results.model.data, "frame"):
            self.training_data = results.model.data.frame
        else:
            raise ValueError(
                "training_data must be provided when the model wasn't fit "
                "via the formula API (no results.model.data.frame available)."
            )

        # Cache exog column names for variable lookup
        self._exog_names = list(results.model.exog_names)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        """Return Σ̂, dispatching to the requested flavor.

        statsmodels supports several robust SE flavors via cov_type at fit
        time. After fitting, results.cov_HC0 / cov_HC1 / cov_HC2 / cov_HC3
        provide the heteroskedasticity-robust variants. For cluster-robust,
        the model must be refit with cov_type='cluster' and cov_kwds, since
        the cluster computation isn't simply derivable from the existing fit.

        Implementers: this is one of the messier parts of the GLM adapter
        because statsmodels' vcov interface isn't uniform. Suggestions:
          - For HC0-HC3: pull from results.cov_HC0 etc. directly
          - For cluster: refit if not already cluster-robust, with a clear
            error if cluster IDs aren't supplied
          - For "default": use results.cov_params()
          - For an ndarray: use as-is (advanced override)
        """
        if vcov_spec is None:
            return jnp.asarray(self.results.cov_params())

        if isinstance(vcov_spec, np.ndarray):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
                attr = f"cov_{vcov_spec.upper()}"
                if hasattr(self.results, attr):
                    return jnp.asarray(getattr(self.results, attr))
                raise ValueError(
                    f"{vcov_spec} not available on this fit. Refit the model "
                    f"with cov_type={vcov_spec.lower()!r}."
                )
            raise ValueError(f"Unsupported vcov string: {vcov_spec!r}")

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
                # Requires refit; not currently implemented
                raise NotImplementedError(
                    "Cluster-robust vcov via this adapter requires the model "
                    "to be refit with cov_type='cluster' and cov_kwds. Either "
                    "refit the model that way and pass vcov=None, or "
                    "supply the cluster-robust matrix directly as an ndarray."
                )
            raise ValueError(f"Unsupported vcov dict type: {kind!r}")

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._predict_jax(beta, X, offset)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        """Build a design matrix from a DataFrame using the model's formula.

        Implementers: this needs to reproduce the formula expansion that
        was used at fit time — factor encoding, interactions, splines,
        polynomial terms. The cleanest approach for formula-fit models is
        to reuse statsmodels' design_info:

            from patsy import dmatrix
            design_info = self.results.model.data.design_info
            X = dmatrix(design_info, df, return_type='matrix')

        For direct-array fits (no formula), df should already match the
        column order of results.model.exog; this method just returns
        df[exog_names].values.

        Returns
        -------
        X : jax array of shape (n_rows, n_features)
        """
        # Try formula-based construction first
        if hasattr(self.results.model.data, "design_info"):
            from patsy import dmatrix
            design_info = self.results.model.data.design_info
            X_np = np.asarray(dmatrix(design_info, df, return_type="matrix"))
            return jnp.asarray(X_np)
        # Fall back: direct column lookup
        return jnp.asarray(df[self._exog_names].values)

    def column_index_of_variable(self, variable_name: str) -> int:
        """Return the index of `variable_name` in the design matrix.

        For variables that map to a single column (continuous, binary in
        treatment coding), this is straightforward. For categorical
        variables expanded into multiple columns, this returns the index
        of the first non-reference level — though dydx() should refuse
        these (categorical → use contrasts).

        Implementers: needs to handle factor expansions correctly. For a
        first cut, raise NotImplementedError when variable_name doesn't
        appear in exog_names directly.
        """
        if variable_name in self._exog_names:
            return self._exog_names.index(variable_name)
        # Heuristic: look for a column whose name starts with variable_name
        for i, name in enumerate(self._exog_names):
            if name.startswith(variable_name):
                return i
        raise ValueError(
            f"Cannot locate variable {variable_name!r} in design matrix. "
            f"exog_names: {self._exog_names}"
        )

    # -----------------------------------------------------------------------
    # Variable metadata
    # -----------------------------------------------------------------------

    def variable_metadata(self) -> dict[str, VariableInfo]:
        """Extract per-variable metadata from the training data.

        Implementers: variable types are inferred heuristically from the
        column dtypes and unique-value counts:
          - bool, or 2 unique values: binary
          - object/category dtype: categorical
          - integer with few unique values: discrete
          - float or integer with many unique values: continuous

        For better metadata, consult the formula's term info if available;
        statsmodels exposes some of this via results.model.data.design_info.
        """
        metadata = {}
        for col in self.training_data.columns:
            series = self.training_data[col]
            metadata[col] = VariableInfo(
                name=col,
                var_type=self._infer_type(series),
                levels=(list(series.unique())
                        if self._infer_type(series) in ("binary", "categorical")
                        else None),
                support=((float(series.min()), float(series.max()))
                         if pd.api.types.is_numeric_dtype(series)
                         else None),
            )
        return metadata

    @staticmethod
    def _infer_type(series: pd.Series) -> str:
        if series.dtype == bool:
            return "binary"
        if not pd.api.types.is_numeric_dtype(series):
            return "categorical"
        unique = series.dropna().unique()
        if len(unique) == 2:
            return "binary"
        if pd.api.types.is_integer_dtype(series) and len(unique) < 20:
            return "discrete"
        return "continuous"

    # -----------------------------------------------------------------------
    # Bootstrap support
    # -----------------------------------------------------------------------

    def refit(self, resampled_data: pd.DataFrame) -> "StatsmodelsGLMAdapter":
        """Refit the model on resampled data.

        Reconstructs the formula and family from the original results and
        fits a new GLM on the resampled data, returning a new adapter.

        Implementers: be careful about the formula API. If the original
        was fit via formula, results.model.formula and results.model.data.frame
        are typically available; reuse them. If it was fit via direct arrays,
        you need to reconstruct exog and endog from resampled_data.
        """
        from statsmodels.formula.api import glm as smf_glm

        formula = getattr(self.results.model, "formula", None)
        if formula is None:
            raise NotImplementedError(
                "Refit only supported for formula-fit models currently. "
                "For array-fit models, the formula reconstruction would need "
                "explicit handling."
            )
        new_results = smf_glm(
            formula, data=resampled_data, family=self.family,
        ).fit()
        return StatsmodelsGLMAdapter(new_results, training_data=resampled_data)


# ---------------------------------------------------------------------------
# Expected usage
# ---------------------------------------------------------------------------
"""
Example: Wrapping a fitted statsmodels GLM
------------------------------------------

    import statsmodels.formula.api as smf
    from pymargins import Margins
    from pymargins._adapters.statsmodels_glm import StatsmodelsGLMAdapter

    fit = smf.glm(
        "outcome ~ treatment + age + sex",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # Auto-detection should find this adapter; for now pass explicitly:
    adapter = StatsmodelsGLMAdapter(fit, training_data=df)
    m = Margins.log_scale(fit, vcov="HC3", adapter=adapter)

    rr = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    print(rr.summary())
"""
