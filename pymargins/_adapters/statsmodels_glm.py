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
"""

from __future__ import annotations

import warnings
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .._adapter import GLMAdapter, VariableInfo
from .._gradients import make_glm_jvp_wrapper
from ._common import (
    build_variable_metadata,
    column_index_of_variable,
    design_matrix_from_df,
    extract_training_data,
    validate_vcov_spec,
)


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

    def __init__(self, results, training_data: pd.DataFrame | None = None):
        self.results = results
        self.family = results.family

        # Build the JAX-compatible predict using analytical link derivative
        self._predict_jax = make_glm_jvp_wrapper(self.family)

        self._training_data = extract_training_data(results, training_data)
        self._exog_names = list(results.model.exog_names)

    @property
    def training_data(self):
        return self._training_data

    def attach(self, session) -> None:
        """Validate session configuration at attach time."""
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="StatsmodelsGLMAdapter")
        super().attach(session)

    # -----------------------------------------------------------------------
    # Core data access
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        return jnp.asarray(self.results.params)

    def score_obs(self) -> np.ndarray:
        """Per-observation score ∂ℓ_i/∂β, shape (n_obs, p).

        Used by :meth:`MarginsResult.influence` to form the analytical
        empirical influence function of an estimand. Statsmodels GLM
        exposes this directly; columns sum to ~0 at the MLE (the FOC).
        """
        return np.asarray(self.results.model.score_obs(self.results.params))

    def covariance(self, vcov_spec: Any | None = None) -> jnp.ndarray:
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

        if isinstance(vcov_spec, (np.ndarray, jnp.ndarray)):
            return jnp.asarray(vcov_spec)

        if isinstance(vcov_spec, str):
            spec_lower = vcov_spec.lower()
            spec_upper = vcov_spec.upper()
            if spec_lower in ("hc0", "hc1", "hc2", "hc3"):
                # GLM stores the robust cov in cov_params() when fit with that
                # cov_type; unlike OLS it does not expose cov_HC3 as a separate
                # attribute. Check whether the fit already used this flavor.
                if getattr(self.results, "cov_type", "").upper() == spec_upper:
                    return jnp.asarray(self.results.cov_params())
                # Otherwise refit with the requested cov_type
                return self._refit_and_extract_cov(cov_type=spec_lower)
            raise ValueError(f"Unsupported vcov string: {vcov_spec!r}")

        if isinstance(vcov_spec, dict):
            kind = vcov_spec.get("type")
            if kind == "cluster":
                groups = vcov_spec.get("groups")
                if groups is None:
                    raise ValueError("cluster vcov requires 'groups' in the spec dict.")
                return self._refit_and_extract_cov(
                    cov_type="cluster",
                    cov_kwds={"groups": groups},
                )
            if kind == "survey":
                return self._survey_covariance(vcov_spec["design"])
            raise ValueError(f"Unsupported vcov dict type: {kind!r}")

        raise ValueError(f"Unsupported vcov_spec: {vcov_spec!r}")

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        return self._predict_jax(beta, X, offset)

    # -----------------------------------------------------------------------
    # Design matrix construction
    # -----------------------------------------------------------------------

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        return design_matrix_from_df(self.results, self._exog_names, df)

    def column_index_of_variable(self, variable_name: str) -> int:
        return column_index_of_variable(
            self._exog_names,
            self.variable_metadata(),
            variable_name,
        )

    def variable_metadata(self) -> dict[str, VariableInfo]:
        if not hasattr(self, "_variable_metadata"):
            self._variable_metadata = build_variable_metadata(self.training_data)
        return self._variable_metadata

    # -----------------------------------------------------------------------
    # Bootstrap support
    # -----------------------------------------------------------------------

    def _survey_covariance(self, design) -> jnp.ndarray:
        import numpy as np

        from .._inference._linearization import (
            linearization_cov,
            weights_proportional,
        )

        # Bread: the model's non-robust cov_params. Reuse, do not recompute.
        bread = np.asarray(self.results.cov_params())
        scores = self.score_obs()  # (n, p)

        w = np.asarray(design.weights)
        psu = None if design.psu is None else np.asarray(design.psu)
        strata = None if design.strata is None else np.asarray(design.strata)

        if len(w) != scores.shape[0]:
            raise ValueError(
                f"survey weights length {len(w)} != n_obs {scores.shape[0]}; "
                "fit the model on the same rows the design describes."
            )

        # Detect fitting weights. If the model was fit with freq_weights or
        # var_weights, statsmodels' cov_params() and score_obs() already
        # incorporate them (weighted bread and weighted scores). Passing
        # survey weights again would double-count, so we use unit weights
        # and let the design variables (strata, PSU, FPC) do the work.
        fit_model = getattr(self.results, "model", None)
        for attr in ("freq_weights", "var_weights"):
            fw = getattr(fit_model, attr, None)
            if fw is not None and not np.allclose(np.asarray(fw), 1.0):
                # The fit weights are baked into both the bread and the scores;
                # using unit weights here avoids double-counting. If they are not
                # proportional to the design weights, the variance (fit weights)
                # and the design-weighted point estimate describe different
                # weightings, so warn rather than silently disagree.
                if not weights_proportional(np.asarray(fw, dtype=float), w):
                    warnings.warn(
                        f"The fitted model's {attr} are not proportional to "
                        "survey_design.weights, so the design-based variance "
                        "(from the fit weights) and the design-weighted point "
                        "estimate may be inconsistent. Fit the model unweighted "
                        "and let survey_design supply the weights, or pass "
                        "matching weights to both.",
                        UserWarning,
                        stacklevel=2,
                    )
                w = np.ones(scores.shape[0])
                break

        # Derive per-row FPC fraction f_h if requested.
        fpc_fraction = self._survey_fpc_fraction(design, psu, strata)

        V = linearization_cov(
            bread, scores, w, psu, strata, fpc_fraction, nest=design.nest
        )
        return jnp.asarray(V)

    @staticmethod
    def _survey_fpc_fraction(design, psu, strata):
        import numpy as np

        if design.fpc is None:
            return None
        fpc = np.asarray(design.fpc, dtype=float)
        if design.fpc_is_fraction:
            return fpc
        # fpc holds N_h (population PSU counts). f_h = n_h / N_h, where n_h is the
        # number of *distinct PSUs* sampled in the stratum.
        n = len(fpc)
        if strata is None:
            strata = np.zeros(n, dtype=int)
        if psu is None:
            psu = np.arange(n)
        out = np.zeros(n)
        for h in np.unique(strata):
            in_h = strata == h
            n_h = len(np.unique(psu[in_h]))
            N_h = float(fpc[in_h][0])  # population count, constant within stratum
            out[in_h] = n_h / N_h
        return out

    def _refit_and_extract_cov(self, cov_type: str, cov_kwds=None) -> jnp.ndarray:
        """Refit the model with a specific cov_type and return its covariance.

        Used when the user requests a vcov flavor that the original fit did
        not compute. For formula-fit models this is straightforward; for
        array-fit models we reconstruct exog/endog and refit.
        """
        from statsmodels.formula.api import glm as smf_glm

        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            if cov_kwds and "groups" in cov_kwds:
                groups = cov_kwds["groups"]
                if hasattr(groups, "__len__") and len(groups) != len(
                    self._training_data
                ):
                    raise ValueError(
                        f"groups length ({len(groups)}) must match training_data "
                        f"length ({len(self._training_data)})."
                    )
            # Preserve model-specific args from the original fit where possible
            fit_kwargs = self._collect_original_fit_kwargs()
            new_results = smf_glm(
                formula,
                data=self._training_data,
                family=self.family,
            ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, **fit_kwargs)
            return jnp.asarray(new_results.cov_params())

        # Array-fit refit
        endog = self.results.model.endog
        exog = self.results.model.exog
        fit_kwargs = self._collect_original_fit_kwargs()
        new_results = sm.GLM(
            endog,
            exog,
            family=self.family,
        ).fit(cov_type=cov_type, cov_kwds=cov_kwds or {}, **fit_kwargs)
        return jnp.asarray(new_results.cov_params())

    def _collect_original_fit_kwargs(self) -> dict:
        """Capture model-specific kwargs from the original fit for refit."""
        kwargs = {}
        for attr in ("offset", "exposure", "freq_weights", "var_weights"):
            val = getattr(self.results.model, attr, None)
            if val is not None:
                kwargs[attr] = val
        return kwargs

    def refit(
        self, resampled_data: pd.DataFrame, *, index=None
    ) -> StatsmodelsGLMAdapter:
        """Refit the model on resampled data.

        Reconstructs the formula and family from the original results and
        fits a new GLM on the resampled data, returning a new adapter.
        """
        from statsmodels.formula.api import glm as smf_glm

        fit_kwargs = self._collect_original_fit_kwargs()
        if index is not None:
            for attr in ("offset", "exposure", "freq_weights", "var_weights"):
                if attr in fit_kwargs and hasattr(fit_kwargs[attr], "__len__"):
                    fit_kwargs[attr] = np.asarray(fit_kwargs[attr])[index]
        formula = getattr(self.results.model, "formula", None)
        if formula is not None:
            new_results = smf_glm(
                formula,
                data=resampled_data,
                family=self.family,
            ).fit(**fit_kwargs)
            return StatsmodelsGLMAdapter(new_results, training_data=resampled_data)

        # Array-fit refit: reconstruct exog and endog from resampled_data.
        # We assume the training_data columns match the model's exog_names
        # (plus the endog variable, which we need to identify).
        endog_name = getattr(self.results.model, "endog_names", None)
        if endog_name is None:
            # Fallback: try to find the response variable by excluding exog columns
            exog_cols = set(self._exog_names)
            # Remove intercept-like column names if they were inserted
            exog_cols.discard("const")
            exog_cols.discard("Intercept")
            possible_endog = [c for c in resampled_data.columns if c not in exog_cols]
            if len(possible_endog) == 1:
                endog_name = possible_endog[0]
            else:
                raise NotImplementedError(
                    "Array-fit refit requires the response variable name. "
                    "Pass training_data with a clear response column, or use "
                    "formula-fit models."
                )
        exog_cols = [c for c in self._exog_names if c in resampled_data.columns]
        if not exog_cols:
            raise ValueError(
                f"None of the model's exog_names {self._exog_names} are present "
                f"in resampled_data columns {list(resampled_data.columns)}. "
                "Pass training_data whose columns match the fitted exog_names."
            )
        endog = resampled_data[endog_name].values
        exog_df = resampled_data[exog_cols]
        # Add intercept if the original model had one
        if "const" in self._exog_names or "Intercept" in self._exog_names:
            if "const" not in exog_df.columns and "Intercept" not in exog_df.columns:
                exog_df = exog_df.copy()
                exog_df.insert(0, "const", 1.0)
        # Resample offset/exposure/weights if index provided
        if index is not None:
            for attr in ("offset", "exposure", "freq_weights", "var_weights"):
                arr = getattr(self.results.model, attr, None)
                if arr is not None:
                    fit_kwargs[attr] = np.asarray(arr)[index]
        new_results = sm.GLM(endog, exog_df, family=self.family).fit(**fit_kwargs)
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
