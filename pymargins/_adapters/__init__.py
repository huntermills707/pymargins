"""
pymargins._adapters

Concrete adapter implementations and the auto-detection dispatch table.

To add support for a new framework **inside pymargins core**:
  1. Subclass one of the adapter shapes from pymargins._adapter
     (GLMAdapter, LinearPredictionAdapter, WrappedFDAdapter,
     BootstrapOnlyAdapter)
  2. Implement the abstract methods (coefficients, covariance, predict,
     design_matrix_from_df, variable_metadata, column_index_of_variable,
     and optionally refit and training_data)
  3. Register the dispatch in _detect_adapter_class below

To add support **from an external package**:
  1. Implement the adapter as above.
  2. Call ``pymargins.register_adapter(adapter_class, predicate=...)``
     at import time.
  3. No changes to pymargins core are required.
"""

# Future adapters to add (one per file in this directory):
#   LinearmodelsPanelAdapter      — uses LinearPredictionAdapter
#   LinearmodelsIVAdapter         — uses LinearPredictionAdapter
#   SklearnLinearAdapter          — uses LinearPredictionAdapter, computes Σ̂
#   SklearnTreeAdapter            — uses BootstrapOnlyAdapter


# ---------------------------------------------------------------------------
# Third-party adapter registry (A11)
# ---------------------------------------------------------------------------

_DETECTION_REGISTRY = []
"""List of (predicate, adapter_class) tuples for auto-detection.

Registered adapters are checked *before* the built-in hardcoded dispatch,
so external packages can override default behaviour.
"""


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
    {
        "name": "StatsmodelsDiscreteBinaryAdapter",
        "description": "statsmodels discrete binary models (Logit, Probit)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["BinaryResultsWrapper", "Logit", "Probit"],
    },
    {
        "name": "StatsmodelsDiscreteCountAdapter",
        "description": "statsmodels discrete count models (Poisson, NegativeBinomial, etc.)",
        "hint_modules": ["statsmodels."],
        "hint_names": [
            "PoissonResultsWrapper",
            "NegativeBinomialResultsWrapper",
            "NegativeBinomialPResultsWrapper",
            "GeneralizedPoissonResultsWrapper",
        ],
    },
    {
        "name": "StatsmodelsZIAdapter",
        "description": "statsmodels zero-inflated count models (ZIP, ZINB, ZIGP)",
        "hint_modules": ["statsmodels."],
        "hint_names": [
            "ZeroInflatedPoissonResultsWrapper",
            "ZeroInflatedNegativeBinomialResultsWrapper",
            "ZeroInflatedGeneralizedPoissonResultsWrapper",
        ],
    },
    {
        "name": "StatsmodelsRLMAdapter",
        "description": "statsmodels RLM (robust linear model)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["RLMResultsWrapper", "RLM"],
    },
    {
        "name": "StatsmodelsQuantRegAdapter",
        "description": "statsmodels QuantReg (quantile regression)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["QuantRegResults", "QuantReg"],
    },
    {
        "name": "StatsmodelsGEEAdapter",
        "description": "statsmodels GEE (generalized estimating equations)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["GEEResultsWrapper", "GEE"],
    },
    {
        "name": "StatsmodelsNominalGEEAdapter",
        "description": "statsmodels NominalGEE (multinomial GEE)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["NominalGEEResultsWrapper", "NominalGEE"],
    },
    {
        "name": "StatsmodelsOrdinalGEEAdapter",
        "description": "statsmodels OrdinalGEE (ordered GEE)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["OrdinalGEEResultsWrapper", "OrdinalGEE"],
    },
    {
        "name": "StatsmodelsMixedLMAdapter",
        "description": "statsmodels MixedLM (linear mixed effects)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["MixedLMResultsWrapper", "MixedLM"],
    },
    {
        "name": "StatsmodelsPHRegAdapter",
        "description": "statsmodels PHReg (Cox proportional hazards)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["PHRegResults", "PHReg"],
    },
    {
        "name": "LifelinesCoxPHAdapter",
        "description": "lifelines CoxPHFitter (Cox proportional hazards)",
        "hint_modules": ["lifelines."],
        "hint_names": ["CoxPHFitter"],
    },
    {
        "name": "LifelinesWeibullAFTAdapter",
        "description": "lifelines WeibullAFTFitter (Weibull AFT)",
        "hint_modules": ["lifelines."],
        "hint_names": ["WeibullAFTFitter"],
    },
    {
        "name": "LifelinesLogNormalAFTAdapter",
        "description": "lifelines LogNormalAFTFitter (LogNormal AFT)",
        "hint_modules": ["lifelines."],
        "hint_names": ["LogNormalAFTFitter"],
    },
    {
        "name": "LifelinesLogLogisticAFTAdapter",
        "description": "lifelines LogLogisticAFTFitter (LogLogistic AFT)",
        "hint_modules": ["lifelines."],
        "hint_names": ["LogLogisticAFTFitter"],
    },
    {
        "name": "LifelinesCoxPHSurvivalAdapter",
        "description": "lifelines CoxPHFitter survival-probability scale (bootstrap-only)",
        "hint_modules": ["lifelines."],
        "hint_names": ["CoxPHFitter"],
    },
    {
        "name": "StatsmodelsPHRegSurvivalAdapter",
        "description": "statsmodels PHReg survival-probability scale (bootstrap-only)",
        "hint_modules": ["statsmodels."],
        "hint_names": ["PHRegResults", "PHReg"],
    },
    {
        "name": "LifelinesGeneralizedGammaAdapter",
        "description": "lifelines GeneralizedGammaRegressionFitter (bootstrap-only)",
        "hint_modules": ["lifelines."],
        "hint_names": ["GeneralizedGammaRegressionFitter"],
    },
    {
        "name": "LifelinesPiecewiseExponentialAdapter",
        "description": "lifelines PiecewiseExponentialRegressionFitter (bootstrap-only)",
        "hint_modules": ["lifelines."],
        "hint_names": ["PiecewiseExponentialRegressionFitter"],
    },
    {
        "name": "LifelinesCRCSplineAdapter",
        "description": "lifelines CRCSplineFitter (bootstrap-only)",
        "hint_modules": ["lifelines."],
        "hint_names": ["CRCSplineFitter"],
    },
    {
        "name": "LifelinesCoxTimeVaryingAdapter",
        "description": "lifelines CoxTimeVaryingFitter (partial hazard)",
        "hint_modules": ["lifelines."],
        "hint_names": ["CoxTimeVaryingFitter"],
    },
    {
        "name": "LifelinesAalenAdditiveAdapter",
        "description": "lifelines AalenAdditiveFitter (bootstrap-only)",
        "hint_modules": ["lifelines."],
        "hint_names": ["AalenAdditiveFitter"],
    },
    {
        "name": "LinearmodelsPanelAdapter",
        "description": "linearmodels panel models (PanelOLS, PooledOLS, RandomEffects, FamaMacBeth, etc.)",
        "hint_modules": ["linearmodels."],
        "hint_names": [
            "PanelEffectsResults",
            "PanelResults",
            "RandomEffectsResults",
            "BetweenResults",
            "FirstDifferenceResults",
            "FamaMacBethResults",
        ],
    },
    {
        "name": "LinearmodelsIVAdapter",
        "description": "linearmodels IV models (IV2SLS, IVGMM, IVLIML) and OLS",
        "hint_modules": ["linearmodels."],
        "hint_names": ["IVResults", "IVGMMResults", "IVLIMLResults", "OLSResults"],
    },
    {
        "name": "LinearmodelsAbsorbingAdapter",
        "description": "linearmodels AbsorbingLS (high-dimensional fixed effects)",
        "hint_modules": ["linearmodels."],
        "hint_names": ["AbsorbingLSResults"],
    },
]


def register_adapter(
    adapter_class,
    *,
    predicate=None,
    hint_modules=None,
    hint_names=None,
    description=None,
):
    """Register an adapter class for auto-detection.

    This is the public hook for third-party packages (or user code) that
    want to support a new modelling framework without modifying pymargins
    core.  Registered adapters are checked *before* the built-in hardcoded
    dispatch, so they can override default behaviour when necessary.

    Parameters
    ----------
    adapter_class : type
        A concrete subclass of `pymargins.ModelAdapter`.
    predicate : callable, optional
        A function ``predicate(model) -> bool`` that returns ``True`` when
        ``adapter_class`` can wrap the supplied fitted model.  If not
        provided, ``hint_modules`` and ``hint_names`` are used to build a
        default predicate that inspects ``type(model).__module__`` and
        ``type(model).__name__``.
    hint_modules : list[str], optional
        Module-name prefixes used for error suggestions (e.g.
        ``["statsmodels."]``).
    hint_names : list[str], optional
        Class-name substrings used for error suggestions (e.g.
        ``["GLM"]``).
    description : str, optional
        Human-readable description shown in "Did you mean …?" error
        messages.

    Examples
    --------
    >>> from pymargins import register_adapter, ModelAdapter
    >>> class MyAdapter(ModelAdapter):
    ...     pass
    >>> register_adapter(
    ...     MyAdapter,
    ...     predicate=lambda m: type(m).__module__.startswith("mypackage."),
    ...     hint_modules=["mypackage."],
    ...     hint_names=["MyModel"],
    ...     description="mypackage MyModel",
    ... )

    Notes
    -----
    Order matters: adapters are checked in registration order.  Register
    more-specific predicates before broader ones.
    """
    if predicate is None:
        if hint_modules is None or hint_names is None:
            raise ValueError(
                "register_adapter requires either a `predicate` or both "
                "`hint_modules` and `hint_names`."
            )
        _mods = list(hint_modules)
        _names = list(hint_names)

        def predicate(model):
            module = type(model).__module__
            cls_name = type(model).__name__
            return (
                any(module.startswith(m) for m in _mods)
                and any(hint in cls_name for hint in _names)
            )

    _DETECTION_REGISTRY.append((predicate, adapter_class))

    # Also add to the suggestion list so the adapter appears in
    # "Did you mean …?" error messages.
    _REGISTERED_ADAPTERS.append(
        {
            "name": adapter_class.__name__,
            "description": description or adapter_class.__name__,
            "hint_modules": list(hint_modules or []),
            "hint_names": list(hint_names or []),
        }
    )


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
    # Third-party registry (checked first so external packages can override)
    for pred, adapter_class in _DETECTION_REGISTRY:
        if pred(model):
            return adapter_class

    cls_name = type(model).__name__
    module = type(model).__module__

    # statsmodels GLM
    if module.startswith("statsmodels.") and "GLM" in cls_name:
        from .statsmodels_glm import StatsmodelsGLMAdapter
        return StatsmodelsGLMAdapter

    # statsmodels QuantReg (also uses RegressionResultsWrapper, so check first)
    if module.startswith("statsmodels.") and cls_name == "RegressionResultsWrapper":
        if hasattr(model, "q") or getattr(getattr(model, "model", None), "__class__", None).__name__ == "QuantReg":
            from .statsmodels_quantreg import StatsmodelsQuantRegAdapter
            return StatsmodelsQuantRegAdapter

    # statsmodels OLS / WLS / GLS
    if module.startswith("statsmodels.") and cls_name in (
        "RegressionResultsWrapper",
    ):
        from .statsmodels_ols import StatsmodelsOLSAdapter
        return StatsmodelsOLSAdapter

    # statsmodels Logit / Probit (discrete binary models)
    # Modern statsmodels (>=0.14) wraps both in BinaryResultsWrapper;
    # older versions used LogitResultsWrapper / ProbitResultsWrapper.
    if module.startswith("statsmodels.") and cls_name in (
        "BinaryResultsWrapper",
        "LogitResultsWrapper",
        "ProbitResultsWrapper",
    ):
        # For BinaryResultsWrapper, distinguish Logit/Probit from other binary
        # models by inspecting the underlying model class.
        model_cls_name = getattr(getattr(model, "model", None), "__class__", None).__name__
        if model_cls_name in ("Logit", "Probit") or cls_name in (
            "LogitResultsWrapper",
            "ProbitResultsWrapper",
        ):
            from .statsmodels_discrete_binary import StatsmodelsDiscreteBinaryAdapter
            return StatsmodelsDiscreteBinaryAdapter

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

    # statsmodels discrete count models
    if module.startswith("statsmodels.") and cls_name in (
        "PoissonResultsWrapper",
        "NegativeBinomialResultsWrapper",
        "NegativeBinomialPResultsWrapper",
        "GeneralizedPoissonResultsWrapper",
    ):
        from .statsmodels_discrete_count import StatsmodelsDiscreteCountAdapter
        return StatsmodelsDiscreteCountAdapter

    # statsmodels zero-inflated count models
    if module.startswith("statsmodels.") and cls_name in (
        "ZeroInflatedPoissonResultsWrapper",
        "ZeroInflatedNegativeBinomialResultsWrapper",
        "ZeroInflatedGeneralizedPoissonResultsWrapper",
    ):
        from .statsmodels_zi import StatsmodelsZIAdapter
        return StatsmodelsZIAdapter

    # statsmodels RLM
    if module.startswith("statsmodels.") and cls_name in (
        "RLMResultsWrapper",
    ):
        from .statsmodels_rlm import StatsmodelsRLMAdapter
        return StatsmodelsRLMAdapter

    # statsmodels QuantReg
    if module.startswith("statsmodels.") and cls_name in (
        "QuantRegResults",
    ):
        from .statsmodels_quantreg import StatsmodelsQuantRegAdapter
        return StatsmodelsQuantRegAdapter

    # statsmodels PHReg (Cox proportional hazards)
    if module.startswith("statsmodels.") and cls_name in (
        "PHRegResults",
    ):
        from .statsmodels_phreg import StatsmodelsPHRegAdapter
        return StatsmodelsPHRegAdapter

    # statsmodels GEE
    if module.startswith("statsmodels.") and cls_name in (
        "GEEResultsWrapper",
    ):
        from .statsmodels_gee import StatsmodelsGEEAdapter
        return StatsmodelsGEEAdapter

    # statsmodels NominalGEE
    if module.startswith("statsmodels.") and cls_name in (
        "NominalGEEResultsWrapper",
    ):
        from .statsmodels_nominal_gee import StatsmodelsNominalGEEAdapter
        return StatsmodelsNominalGEEAdapter

    # statsmodels OrdinalGEE
    if module.startswith("statsmodels.") and cls_name in (
        "OrdinalGEEResultsWrapper",
    ):
        from .statsmodels_ordinal_gee import StatsmodelsOrdinalGEEAdapter
        return StatsmodelsOrdinalGEEAdapter

    # statsmodels MixedLM
    if module.startswith("statsmodels.") and cls_name in (
        "MixedLMResultsWrapper",
    ):
        from .statsmodels_mixedlm import StatsmodelsMixedLMAdapter
        return StatsmodelsMixedLMAdapter

    # lifelines CoxPHFitter
    if module.startswith("lifelines.") and cls_name in (
        "CoxPHFitter",
    ):
        from .lifelines_coxph import LifelinesCoxPHAdapter
        return LifelinesCoxPHAdapter

    # lifelines WeibullAFTFitter
    if module.startswith("lifelines.") and cls_name in (
        "WeibullAFTFitter",
    ):
        from .lifelines_weibull_aft import LifelinesWeibullAFTAdapter
        return LifelinesWeibullAFTAdapter

    # lifelines LogNormalAFTFitter
    if module.startswith("lifelines.") and cls_name in (
        "LogNormalAFTFitter",
    ):
        from .lifelines_lognormal_aft import LifelinesLogNormalAFTAdapter
        return LifelinesLogNormalAFTAdapter

    # lifelines LogLogisticAFTFitter
    if module.startswith("lifelines.") and cls_name in (
        "LogLogisticAFTFitter",
    ):
        from .lifelines_loglogistic_aft import LifelinesLogLogisticAFTAdapter
        return LifelinesLogLogisticAFTAdapter

    # lifelines GeneralizedGammaRegressionFitter
    if module.startswith("lifelines.") and cls_name in (
        "GeneralizedGammaRegressionFitter",
    ):
        from .lifelines_generalized_gamma import LifelinesGeneralizedGammaAdapter
        return LifelinesGeneralizedGammaAdapter

    # lifelines PiecewiseExponentialRegressionFitter
    if module.startswith("lifelines.") and cls_name in (
        "PiecewiseExponentialRegressionFitter",
    ):
        from .lifelines_piecewise_exponential import LifelinesPiecewiseExponentialAdapter
        return LifelinesPiecewiseExponentialAdapter

    # lifelines CRCSplineFitter
    if module.startswith("lifelines.") and cls_name in (
        "CRCSplineFitter",
    ):
        from .lifelines_crc_spline import LifelinesCRCSplineAdapter
        return LifelinesCRCSplineAdapter

    # lifelines CoxTimeVaryingFitter
    if module.startswith("lifelines.") and cls_name in (
        "CoxTimeVaryingFitter",
    ):
        from .lifelines_coxtimevarying import LifelinesCoxTimeVaryingAdapter
        return LifelinesCoxTimeVaryingAdapter

    # lifelines AalenAdditiveFitter
    if module.startswith("lifelines.") and cls_name in (
        "AalenAdditiveFitter",
    ):
        from .lifelines_aalen_additive import LifelinesAalenAdditiveAdapter
        return LifelinesAalenAdditiveAdapter

    # linearmodels panel models
    if module.startswith("linearmodels.") and cls_name in (
        "PanelEffectsResults",
        "PanelResults",
        "RandomEffectsResults",
        "BetweenResults",
        "FirstDifferenceResults",
        "FamaMacBethResults",
    ):
        from .linearmodels_panel import LinearmodelsPanelAdapter
        return LinearmodelsPanelAdapter

    # linearmodels IV models
    if module.startswith("linearmodels.") and cls_name in (
        "IVResults",
        "IVGMMResults",
        "IVLIMLResults",
        "OLSResults",
    ):
        from .linearmodels_iv import LinearmodelsIVAdapter
        return LinearmodelsIVAdapter

    # linearmodels AbsorbingLS
    if module.startswith("linearmodels.") and cls_name in (
        "AbsorbingLSResults",
    ):
        from .linearmodels_absorbing import LinearmodelsAbsorbingAdapter
        return LinearmodelsAbsorbingAdapter

    # Note: LifelinesCoxPHSurvivalAdapter and StatsmodelsPHRegSurvivalAdapter
    # are NOT auto-detected because they share the same result class as their
    # hazard-ratio counterparts. Users must construct them explicitly and pass
    # via adapter=.

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
