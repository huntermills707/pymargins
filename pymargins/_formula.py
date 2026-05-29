"""
pymargins._formula

FormulaSpec — a pymargins-owned formula interface for adapters that do not
have native formula support (array-fit statsmodels, linearmodels, sklearn).

Problem
-------
``make_slope_estimand`` computes ``dydx()`` as a data-side total derivative:
it perturbs the source DataFrame column by ±ε and rebuilds the design via
``adapter.design_matrix_from_df(df±)``. For this to be correct, the design
matrix must re-evaluate every interaction, polynomial, spline, and ``I(...)``
term that depends on the perturbed column.

Array-fit statsmodels and linearmodels adapters use column-selection fallback
(``df.reindex(columns=exog_names)``), which does NOT re-evaluate derived terms.
This silently produces wrong slopes — a correctness defect, not a convenience.

Solution
--------
FormulaSpec is constructed once from a formula string + training DataFrame at
adapter attach time. It captures and freezes stateful transform parameters
(centering means, spline knots, contrast coding) from the training data using
patsy ``design_info``. When applied to new rows, ``get_model_matrix`` re-applies
the *training-time* state, so counterfactual scenarios and dydx ±ε perturbations
get the correct derived-term propagation.

Resolution order in ``design_matrix_from_df``:
  1. Live framework ``design_info`` (statsmodels formula-fit, lifelines) —
     unchanged, zero risk;
  2. pymargins ``FormulaSpec`` if the user supplied ``formula=``;
  3. Column-selection fallback **with a warning** when derived terms would be
     unrepresentable.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd


class FormulaSpec:
    """Frozen formula specification built from training data.

    Uses patsy to capture ``design_info`` with frozen transform parameters
    (centering means, spline knots, contrast coding). When applied to new
    data, ``get_model_matrix`` re-evaluates the formula using the training-time
    state, ensuring that perturbations propagate into derived terms.
    """

    def __init__(self, formula: str, training_data: pd.DataFrame):
        from patsy import ModelDesc, dmatrix

        if not isinstance(formula, str):
            raise TypeError(f"formula must be a string, got {type(formula).__name__}")
        if not isinstance(training_data, pd.DataFrame):
            raise TypeError(
                f"training_data must be a pandas DataFrame, got {type(training_data).__name__}"
            )
        if len(training_data) == 0:
            raise ValueError("training_data must not be empty.")

        self.formula = formula
        # patsy dmatrix expects only the RHS; extract it if user passed a full formula
        rhs_formula = formula
        try:
            md = ModelDesc.from_formula(formula)
            if md.lhs_termlist:
                rhs_md = ModelDesc([], md.rhs_termlist)
                rhs_formula = rhs_md.describe()
        except Exception:
            pass  # If parsing fails, pass the original string and let patsy raise

        # Build design_info on training data to freeze stateful transforms
        X_df = dmatrix(rhs_formula, training_data, return_type="dataframe")
        self.design_info = X_df.design_info
        self.column_names = list(X_df.columns)

    def get_model_matrix(self, df: pd.DataFrame) -> jnp.ndarray:
        """Re-apply the frozen formula to ``df`` and return a JAX array."""
        from patsy import dmatrix

        X_df = dmatrix(self.design_info, df, return_type="dataframe")
        return jnp.asarray(np.asarray(X_df))

    @property
    def exog_names(self) -> list[str]:
        """Column names produced by this formula specification."""
        return self.column_names

    def verify_against(
        self,
        adapter,
        tol: float = 1e-5,
    ) -> None:
        """Verify that this FormulaSpec reproduces the fitted linear predictor.

        Computes ``X_formula @ β̂`` on the training data and compares it to the
        model's native fitted linear predictor. Raises ``ValueError`` with a
        diagnostic message if the mismatch exceeds ``tol``.

        For GLM-family models the verification uses the *linear predictor* (η),
        not the mean response (μ), so the check remains valid across link
        functions.

        Parameters
        ----------
        adapter : ModelAdapter
            The adapter wrapping the fitted model.
        tol : float, default 1e-5
            Maximum allowed infinity-norm difference.
        """
        X_form = np.asarray(self.get_model_matrix(adapter.training_data))
        n_params = len(adapter.coefficients())
        if X_form.shape[1] != n_params:
            raise ValueError(
                f"Formula verification failed: formula produced {X_form.shape[1]} "
                f"columns but model has {n_params} parameters. "
                f"Check that the formula matches the model specification exactly "
                f"(including intercept, factor coding, and term order)."
            )

        beta = np.asarray(adapter.coefficients())
        lp_form = X_form @ beta

        # Try to get native fitted values / linear predictor from the model
        lp_native = None
        results = getattr(adapter, "results", None)
        if results is not None:
            # Prefer an explicit linear predictor when available
            if hasattr(results, "linear_predictor"):
                lp_native = np.asarray(results.linear_predictor)
            # statsmodels GLM: fittedvalues is μ; convert to η via link
            elif hasattr(results, "model") and hasattr(results.model, "family"):
                mu = np.asarray(results.fittedvalues)
                lp_native = np.asarray(results.model.family.link(mu))
            # statsmodels and lifelines expose fittedvalues (identity link)
            elif hasattr(results, "fittedvalues"):
                lp_native = np.asarray(results.fittedvalues)
            # linearmodels uses fitted_values
            elif hasattr(results, "fitted_values"):
                lp_native = np.asarray(results.fitted_values)

        if lp_native is None:
            warnings.warn(
                "Formula verification skipped: the model does not expose fitted "
                "values or a linear predictor, so the numeric correctness check "
                "cannot be performed. Only the column count was verified.",
                UserWarning,
                stacklevel=2,
            )
            return

        # Flatten to 1-D so (n,) − (n,1) doesn't broadcast to (n,n)
        lp_native = lp_native.ravel()
        lp_form = np.asarray(lp_form).ravel()

        if len(lp_native) != len(lp_form):
            warnings.warn(
                "Formula verification skipped: the native fitted values have "
                f"{len(lp_native)} rows but the formula design has {len(lp_form)}. "
                "Only the column count was verified.",
                UserWarning,
                stacklevel=2,
            )
            return

        diff = np.abs(lp_form - lp_native)
        max_diff = float(np.max(diff))
        if max_diff > tol:
            first_bad = int(np.argmax(diff))
            raise ValueError(
                f"Formula verification failed: max absolute difference in linear "
                f"predictor is {max_diff:.6f} (tolerance {tol}). First divergent "
                f"row: {first_bad}. Check that the formula exactly matches the "
                f"model specification (including intercept, factor coding, and "
                f"term order)."
            )


def _has_derived_terms(exog_names: list[str]) -> bool:
    """Detect whether ``exog_names`` contains derived / transformed terms.

    Returns ``True`` if the column names contain patterns indicating
    interactions, polynomials, splines, or other formula-derived terms
    that column-selection fallback cannot reproduce.
    """
    derived_indicators = [":", "I(", "bs(", "poly(", "cc(", "cr(", "te(", "ti("]
    for name in exog_names:
        for indicator in derived_indicators:
            if indicator in name:
                return True
    return False
