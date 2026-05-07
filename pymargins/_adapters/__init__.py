"""
pymargins._adapters

Concrete adapter implementations and the auto-detection dispatch table.

To add support for a new framework:
  1. Subclass one of the adapter shapes from pymargins._adapter
     (GLMAdapter, LinearPredictionAdapter, WrappedFDAdapter,
     BootstrapOnlyAdapter)
  2. Implement the abstract methods (coefficients, covariance, predict,
     design_matrix_from_df, variable_metadata, column_index_of_variable,
     and optionally refit and training_data)
  3. Register the dispatch in _detect_adapter_class below
"""

# Future adapters to add (one per file in this directory):
#   StatsmodelsOLSAdapter         — uses LinearPredictionAdapter
#   StatsmodelsMixedLMAdapter     — uses WrappedFDAdapter
#   StatsmodelsTSAAdapter         — uses WrappedFDAdapter; for ARIMA etc.
#   LinearmodelsPanelAdapter      — uses LinearPredictionAdapter
#   LinearmodelsIVAdapter         — uses LinearPredictionAdapter
#   SklearnLinearAdapter          — uses LinearPredictionAdapter, computes Σ̂
#   SklearnTreeAdapter            — uses BootstrapOnlyAdapter


def _detect_adapter_class(model):
    """Inspect a fitted model and return the appropriate adapter class.

    Order matters: more specific checks first. As new adapters are added,
    register their dispatch here.

    This is the body of the public auto_detect_adapter() function in
    _adapter.py; it lives here so the imports of concrete adapters don't
    pollute the base module.
    """
    cls_name = type(model).__name__
    module = type(model).__module__

    # statsmodels GLM
    if module.startswith("statsmodels.") and "GLM" in cls_name:
        from .statsmodels_glm import StatsmodelsGLMAdapter
        return StatsmodelsGLMAdapter

    # statsmodels OLS / WLS / GLS
    if module.startswith("statsmodels.") and cls_name == "RegressionResultsWrapper":
        from .statsmodels_ols import StatsmodelsOLSAdapter
        return StatsmodelsOLSAdapter

    # Fall through with a clear error
    raise NotImplementedError(
        f"No adapter registered for {module}.{cls_name}. "
        "Either pass an explicit `adapter=` to Margins(), or register a "
        "new adapter in pymargins._adapters._detect_adapter_class."
    )


def auto_detect_adapter(model):
    """Public entry point for adapter auto-detection.

    Returns an instantiated adapter for `model`. For most adapters the
    constructor takes only `model` (and optional training_data); subclasses
    that need more arguments should be instantiated explicitly by the user
    and passed to Margins via the `adapter=` keyword.
    """
    cls = _detect_adapter_class(model)
    return cls(model)
