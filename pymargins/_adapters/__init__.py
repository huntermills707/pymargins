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
#   StatsmodelsMixedLMAdapter     — uses WrappedFDAdapter
#   StatsmodelsTSAAdapter         — uses WrappedFDAdapter; for ARIMA etc.
#   LinearmodelsPanelAdapter      — uses LinearPredictionAdapter
#   LinearmodelsIVAdapter         — uses LinearPredictionAdapter
#   SklearnLinearAdapter          — uses LinearPredictionAdapter, computes Σ̂
#   SklearnTreeAdapter            — uses BootstrapOnlyAdapter


# ---------------------------------------------------------------------------
# Adapter registry for error suggestions
# ---------------------------------------------------------------------------

_REGISTERED_ADAPTERS = [
    {
        "name": "StatsmodelsGLMAdapter",
        "description": "statsmodels GLM (Logit, Probit, Poisson, Gamma, etc.)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["GLM"],
    },
    {
        "name": "StatsmodelsOLSAdapter",
        "description": "statsmodels OLS / WLS / GLS",
        "hint_modules": ["statsmodels."],
        "hint_names": ["RegressionResultsWrapper", "OLS", "WLS", "GLS"],
    },
    {
        "name": "StatsmodelsMNLogitAdapter",
        "description": "statsmodels MNLogit (multinomial logit)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["MultinomialResultsWrapper", "MNLogit"],
    },
    {
        "name": "StatsmodelsOrderedAdapter",
        "description": "statsmodels OrderedModel (ordered probit/logit)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["OrderedResultsWrapper", "OrderedModel"],
    },
]


def _suggest_adapters(cls_name, module):
    """Build a suggestion message for unsupported models."""
    # Heuristic: strong match when both module prefix and class name hint align
    strong = []
    weak = []
    for entry in _REGISTERED_ADAPTERS:
        mod_match = any(module.startswith(m) for m in entry["hint_modules"])
        cls_match = any(hint in cls_name for hint in entry["hint_names"])
        if mod_match and cls_match:
            strong.append(entry)
        elif mod_match or cls_match:
            weak.append(entry)

    lines = []
    if strong:
        lines.append("Did you mean one of these adapters?")
        for entry in strong:
            lines.append(f"  - {entry['name']}: {entry['description']}")
    elif weak:
        lines.append("Possibly related adapters:")
        for entry in weak:
            lines.append(f"  - {entry['name']}: {entry['description']}")
    else:
        lines.append("Currently registered adapters:")
        for entry in _REGISTERED_ADAPTERS:
            lines.append(f"  - {entry['name']}: {entry['description']}")

    lines.append(
        "To write a custom adapter, see the guide in "
        "pymargins._adapters.__init__ and the ModelAdapter base class."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

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
    if module.startswith("statsmodels.") and cls_name in (
        "RegressionResultsWrapper",
    ):
        from .statsmodels_ols import StatsmodelsOLSAdapter
        return StatsmodelsOLSAdapter

    # statsmodels Logit / Probit (discrete choice, wrapper around GLM family)
    if module.startswith("statsmodels.") and cls_name in (
        "LogitResultsWrapper",
        "ProbitResultsWrapper",
    ):
        from .statsmodels_glm import StatsmodelsGLMAdapter
        return StatsmodelsGLMAdapter

    # statsmodels MNLogit
    if module.startswith("statsmodels.") and cls_name in (
        "MultinomialResultsWrapper",
    ):
        from .statsmodels_mnlogit import StatsmodelsMNLogitAdapter
        return StatsmodelsMNLogitAdapter

    # statsmodels OrderedModel
    if module.startswith("statsmodels.") and cls_name in (
        "OrderedResultsWrapper",
    ):
        from .statsmodels_ordered import StatsmodelsOrderedAdapter
        return StatsmodelsOrderedAdapter

    # Fall through with a clear error that suggests the closest adapter
    suggestion = _suggest_adapters(cls_name, module)
    raise NotImplementedError(
        f"No adapter registered for {module}.{cls_name}.\n"
        f"{suggestion}\n"
        "Alternatively, pass an explicit `adapter=` to Margins()."
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
