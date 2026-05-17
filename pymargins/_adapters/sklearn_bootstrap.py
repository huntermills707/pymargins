"""
pymargins._adapters.sklearn_bootstrap

Bootstrap-only adapter for scikit-learn estimators.

sklearn models do not expose a meaningful parametric coefficient vector or
covariance matrix, so only bootstrap inference is supported.  The adapter
wraps ``model.predict()`` and ignores the dummy ``beta`` parameter that the
bootstrap path passes — predictions are always evaluated on the *fitted model*
carried by the adapter instance.

When ``formula=`` and ``data=`` are provided, ``design_matrix_from_df`` uses a
patsy-backed :class:`pymargins._formula.FormulaSpec` so that ``dydx()``
correctly propagates perturbations into derived terms (interactions,
polynomials, splines).  Without a formula, ``dydx()`` on a variable that is
involved in derived terms falls back to column selection and **raises**
a clear error if the fallback cannot represent the term.
"""

from __future__ import annotations
from typing import Optional, Any
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd

from sklearn.base import clone

from .._adapter import BootstrapOnlyAdapter, VariableInfo
from ._common import (
    column_index_of_variable,
    build_variable_metadata,
    validate_vcov_spec,
    _has_derived_terms,
)


class SklearnBootstrapAdapter(BootstrapOnlyAdapter):
    """Adapter for scikit-learn estimators (bootstrap-only).

    Parameters
    ----------
    model : fitted sklearn estimator
        Any sklearn estimator with ``fit()`` and ``predict()``.
    X_train : array-like, optional
        Training features. Ignored if ``data`` is provided.
    y_train : array-like, optional
        Training target. Ignored if ``data`` is provided.
    formula : str, optional
        Formula string for models that need correct ``dydx()`` on derived
        terms. When provided, ``data`` must also be provided.
    data : pd.DataFrame, optional
        Training DataFrame. If provided and ``X_train``/``y_train`` are not,
        the adapter extracts features and target from this frame.
    target_name : str, optional
        Name of the target column in ``data``. Required when ``data`` is
        provided and the target column cannot be inferred automatically.
    """

    def __init__(
        self,
        model,
        *,
        X_train=None,
        y_train=None,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        target_name: Optional[str] = None,
    ):
        self.model = model
        self._formula_spec = None
        self._training_data = None
        self._target_name = target_name

        # Resolve training data and formula spec
        if formula is not None:
            if data is None:
                raise ValueError(
                    "SklearnBootstrapAdapter: formula= requires data= "
                    "to build the FormulaSpec."
                )
            from .._formula import FormulaSpec

            self._formula_spec = FormulaSpec(formula, data)
            self._training_data = data
            if X_train is None or y_train is None:
                X_train, y_train = self._split_data(data, target_name)
        elif data is not None:
            self._training_data = data
            if X_train is None or y_train is None:
                X_train, y_train = self._split_data(data, target_name)
        elif X_train is not None:
            # Convert array-like X_train to DataFrame for bootstrap compatibility
            if hasattr(X_train, "columns"):
                self._training_data = X_train.copy()
            else:
                cols = [f"x{i}" for i in range(X_train.shape[1])]
                self._training_data = pd.DataFrame(np.asarray(X_train), columns=cols)

        self._X_train = X_train
        self._y_train = y_train

        # Feature names for design-matrix construction
        if self._formula_spec is not None:
            self._feature_names = self._formula_spec.exog_names
        elif hasattr(model, "feature_names_in_"):
            self._feature_names = list(model.feature_names_in_)
        elif X_train is not None and hasattr(X_train, "columns"):
            self._feature_names = list(X_train.columns)
        elif X_train is not None:
            self._feature_names = [f"x{i}" for i in range(X_train.shape[1])]
        else:
            self._feature_names = None

        # Dummy coefficient for the bootstrap path (ignored by predict)
        self._dummy_beta = jnp.array([0.0])

    @staticmethod
    def _split_data(data: pd.DataFrame, target_name: Optional[str] = None):
        """Split a DataFrame into X and y, inferring target if needed."""
        if target_name is not None:
            if target_name not in data.columns:
                raise ValueError(
                    f"target_name={target_name!r} not in data columns: "
                    f"{list(data.columns)}"
                )
            y = data[target_name].values
            X = data.drop(columns=[target_name])
            return X, y

        # Try to infer: sklearn models usually have target as the last column
        # or a column not used in the formula/feature list
        raise ValueError(
            "SklearnBootstrapAdapter: target_name must be provided when "
            "constructing from a DataFrame without an explicit y_train."
        )

    # -----------------------------------------------------------------------
    # Core data access (dummy implementations — bootstrap path only)
    # -----------------------------------------------------------------------

    def coefficients(self) -> jnp.ndarray:
        """Return a dummy coefficient vector.

        The bootstrap path requires a coefficient array, but sklearn models
        do not expose a meaningful parametric vector.  The dummy is ignored
        by ``predict()``.
        """
        return self._dummy_beta

    def covariance(self, vcov_spec: Optional[Any] = None) -> jnp.ndarray:
        """Return a dummy covariance matrix."""
        return jnp.array([[0.0]])

    def predict(
        self,
        beta: jnp.ndarray,
        X: jnp.ndarray,
        offset: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Predict using the fitted sklearn model, ignoring ``beta``."""
        X_np = np.asarray(X)
        # sklearn models fitted with a DataFrame warn when predict() receives
        # a bare array ("X does not have valid feature names").  Re-wrap in a
        # DataFrame using the model's own recorded feature names so the
        # warning is suppressed without risking a name mismatch when a formula
        # is active.
        model_features = getattr(self.model, "feature_names_in_", None)
        if model_features is not None and X_np.ndim == 2:
            X_in = pd.DataFrame(X_np, columns=list(model_features))
        else:
            X_in = X_np
        preds = self.model.predict(X_in)
        return jnp.asarray(preds)

    # -----------------------------------------------------------------------
    # Design and metadata
    # -----------------------------------------------------------------------

    @property
    def training_data(self):
        return self._training_data

    def design_matrix_from_df(self, df: pd.DataFrame) -> jnp.ndarray:
        if self._formula_spec is not None:
            return self._formula_spec.get_model_matrix(df)

        if self._feature_names is None:
            raise ValueError(
                "SklearnBootstrapAdapter cannot build a design matrix: "
                "no formula= was provided and feature_names could not be "
                "inferred. Pass formula= and data= for correct dydx() on "
                "derived terms."
            )

        if _has_derived_terms(self._feature_names):
            raise ValueError(
                "SklearnBootstrapAdapter: feature names contain derived terms "
                "(interactions, polynomials, splines, etc.) but no formula= was "
                "provided. dydx() on variables involved in these terms would be "
                "silently incorrect because the adapter cannot re-evaluate "
                "derived terms without a formula. Pass formula= and data= to fix "
                "this."
            )

        aligned = df.reindex(columns=self._feature_names)
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. "
                f"Available: {list(df.columns)}."
            )
        return jnp.asarray(aligned.values)

    def column_index_of_variable(self, variable_name: str) -> int:
        if self._feature_names is None:
            raise ValueError(
                "SklearnBootstrapAdapter: feature_names not available."
            )
        return column_index_of_variable(
            self._feature_names,
            self.variable_metadata(),
            variable_name,
        )

    def variable_metadata(self) -> dict[str, VariableInfo]:
        if not hasattr(self, "_variable_metadata"):
            if self._training_data is not None:
                self._variable_metadata = build_variable_metadata(
                    self._training_data
                )
            elif self._X_train is not None and hasattr(self._X_train, "columns"):
                self._variable_metadata = build_variable_metadata(self._X_train)
            else:
                # No DataFrame metadata available — build minimal metadata
                self._variable_metadata = {
                    name: VariableInfo(name=name, var_type="continuous")
                    for name in (self._feature_names or [])
                }
        return self._variable_metadata

    # -----------------------------------------------------------------------
    # Bootstrap support
    # -----------------------------------------------------------------------

    def _verify_formula_spec(self, tol: float = 1e-4) -> None:
        """Verify that FormulaSpec reproduces the model's predictions."""
        if self._formula_spec is None or self._training_data is None:
            return
        X_form = np.asarray(self._formula_spec.get_model_matrix(self._training_data))
        X_orig = np.asarray(self._X_train)
        model_features = getattr(self.model, "feature_names_in_", None)
        # Wrap in DataFrame when shapes match to avoid sklearn's feature-name
        # warning.  If shapes differ, pass the array through so sklearn raises
        # its usual column-count error, which we translate below.
        if model_features is not None and X_form.shape[1] == len(model_features):
            X_form_in = pd.DataFrame(X_form, columns=list(model_features))
        else:
            X_form_in = X_form
        if model_features is not None and X_orig.shape[1] == len(model_features):
            X_orig_in = pd.DataFrame(X_orig, columns=list(model_features))
        else:
            X_orig_in = X_orig
        try:
            preds_form = self.model.predict(X_form_in)
        except ValueError as exc:
            if "feature" in str(exc).lower():
                raise ValueError(
                    "Formula verification failed: the formula-built design matrix "
                    f"has {X_form.shape[1]} columns, but the sklearn model expects "
                    f"{getattr(self.model, 'n_features_in_', '?')} features. "
                    "This usually means the formula produces a different number of "
                    "columns than the model was trained on (e.g., an unexpected "
                    "intercept). Use '0 + ...' in the formula to suppress the "
                    "intercept, or ensure the training data includes an explicit "
                    "'Intercept' column."
                ) from exc
            raise
        preds_orig = self.model.predict(X_orig_in)
        if len(preds_form) != len(preds_orig):
            return
        max_diff = float(np.max(np.abs(preds_form - preds_orig)))
        if max_diff > tol:
            raise ValueError(
                f"Formula verification failed: model predictions on formula-built "
                f"design matrix differ from original predictions by {max_diff:.6f} "
                f"(tolerance {tol}). This usually means the formula produces "
                f"different columns than the model was trained on (e.g., an "
                f"unexpected intercept). Use '0 + ...' in the formula to suppress "
                f"the intercept, or ensure the training data includes an explicit "
                f"'Intercept' column."
            )

    def refit(self, resampled_data, *, index=None) -> "SklearnBootstrapAdapter":
        """Refit the sklearn model on resampled data."""
        if hasattr(resampled_data, "iloc"):
            # DataFrame input — could be full data (with target) or just X
            if self._target_name is not None and self._target_name in resampled_data.columns:
                X_new, y_new = self._split_data(resampled_data, self._target_name)
            else:
                # No target in resampled data — use stored y_train
                X_new = resampled_data
                if index is not None and self._y_train is not None:
                    y_new = np.asarray(self._y_train)[index]
                else:
                    y_new = self._y_train
        else:
            # Array-like input
            X_new = resampled_data
            if index is not None and self._y_train is not None:
                y_new = np.asarray(self._y_train)[index]
            else:
                y_new = self._y_train

        new_model = clone(self.model)
        new_model.fit(X_new, y_new)

        return SklearnBootstrapAdapter(
            new_model,
            X_train=X_new,
            y_train=y_new,
            formula=getattr(self._formula_spec, "formula", None)
            if self._formula_spec is not None
            else None,
            data=self._training_data,
            target_name=self._target_name,
        )

    def attach(self, session) -> None:
        vcov = getattr(session, "vcov_spec", None)
        validate_vcov_spec(vcov, adapter_name="SklearnBootstrapAdapter")
        if vcov is not None:
            raise ValueError(
                "SklearnBootstrapAdapter does not support custom vcov "
                "specifications. Bootstrap inference does not use a "
                "parametric covariance matrix."
            )
        super().attach(session)
        if self._formula_spec is not None:
            self._verify_formula_spec()
