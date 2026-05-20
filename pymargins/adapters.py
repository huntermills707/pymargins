"""
pymargins.adapters — public access to concrete model adapters.

You rarely need anything here. ``Margins(model)`` auto-detects the right
adapter for standard statsmodels / linearmodels / lifelines results. Import
a concrete adapter only when:

1. You need a non-default scale that shares a result class with another
   adapter and therefore cannot be auto-detected — e.g.
   :class:`LifelinesCoxPHSurvivalAdapter` (survival probability) vs the
   auto-detected :class:`LifelinesCoxPHAdapter` (hazard ratio).
2. You are wrapping a framework with no native inference, such as
   scikit-learn via :class:`SklearnBootstrapAdapter`.
3. You want to construct an adapter explicitly to pass extra arguments
   (e.g. ``training_data=``) or to override auto-detection.

To *write* a custom adapter, subclass one of the base shapes exported from
the top-level package (``ModelAdapter``, ``GLMAdapter``,
``LinearPredictionAdapter``, ``WrappedFDAdapter``, ``BootstrapOnlyAdapter``)
and register it with :func:`pymargins.register_adapter`. See
``docs/howto/custom_adapter.md``.

Imports here are lazy (PEP 562): referencing an adapter triggers the import
of its module and the corresponding optional third-party dependency, so
``import pymargins`` stays cheap and does not require statsmodels,
lifelines, linearmodels, or scikit-learn to be installed.
"""

from __future__ import annotations

# Base classes and the registration hook are re-exported for convenience so
# that pymargins.adapters is a one-stop import for adapter work. These are
# cheap (no optional deps) and so are imported eagerly.
from ._adapter import (
    ModelAdapter,
    GLMAdapter,
    LinearPredictionAdapter,
    WrappedFDAdapter,
    BootstrapOnlyAdapter,
    VariableInfo,
    InferenceMethod,
)
from ._adapters import register_adapter

# Concrete adapter class name -> submodule under pymargins._adapters.
# Each submodule pulls in its optional third-party dependency, so resolution
# is deferred to module __getattr__ below.
_ADAPTER_MODULES = {
    # statsmodels
    "StatsmodelsGLMAdapter": "statsmodels_glm",
    "StatsmodelsOLSAdapter": "statsmodels_ols",
    "StatsmodelsRLMAdapter": "statsmodels_rlm",
    "StatsmodelsQuantRegAdapter": "statsmodels_quantreg",
    "StatsmodelsDiscreteBinaryAdapter": "statsmodels_discrete_binary",
    "StatsmodelsDiscreteCountAdapter": "statsmodels_discrete_count",
    "StatsmodelsZIAdapter": "statsmodels_zi",
    "StatsmodelsMNLogitAdapter": "statsmodels_mnlogit",
    "StatsmodelsOrderedAdapter": "statsmodels_ordered",
    "StatsmodelsGEEAdapter": "statsmodels_gee",
    "StatsmodelsNominalGEEAdapter": "statsmodels_nominal_gee",
    "StatsmodelsOrdinalGEEAdapter": "statsmodels_ordinal_gee",
    "StatsmodelsMixedLMAdapter": "statsmodels_mixedlm",
    "StatsmodelsPHRegAdapter": "statsmodels_phreg",
    "StatsmodelsPHRegSurvivalAdapter": "statsmodels_phreg_survival",
    # lifelines
    "LifelinesCoxPHAdapter": "lifelines_coxph",
    "LifelinesCoxPHSurvivalAdapter": "lifelines_coxph_survival",
    "LifelinesCoxTimeVaryingAdapter": "lifelines_coxtimevarying",
    "LifelinesCoxTimeVaryingSurvivalAdapter": "lifelines_cox_timevarying",
    "LifelinesWeibullAFTAdapter": "lifelines_weibull_aft",
    "LifelinesLogNormalAFTAdapter": "lifelines_lognormal_aft",
    "LifelinesLogLogisticAFTAdapter": "lifelines_loglogistic_aft",
    "LifelinesGeneralizedGammaAdapter": "lifelines_generalized_gamma",
    "LifelinesPiecewiseExponentialAdapter": "lifelines_piecewise_exponential",
    "LifelinesCRCSplineAdapter": "lifelines_crc_spline",
    "LifelinesCRCSplineHRAdapter": "lifelines_crc_spline_hr",
    "LifelinesAalenAdditiveAdapter": "lifelines_aalen_additive",
    # linearmodels
    "LinearmodelsPanelAdapter": "linearmodels_panel",
    "LinearmodelsIVAdapter": "linearmodels_iv",
    "LinearmodelsAbsorbingAdapter": "linearmodels_absorbing",
    # scikit-learn
    "SklearnBootstrapAdapter": "sklearn_bootstrap",
}

_BASE_EXPORTS = (
    "ModelAdapter",
    "GLMAdapter",
    "LinearPredictionAdapter",
    "WrappedFDAdapter",
    "BootstrapOnlyAdapter",
    "VariableInfo",
    "InferenceMethod",
    "register_adapter",
)

__all__ = [*_BASE_EXPORTS, *sorted(_ADAPTER_MODULES)]


def __getattr__(name: str):
    """Lazily resolve concrete adapters (PEP 562)."""
    submodule = _ADAPTER_MODULES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f"._adapters.{submodule}", __package__)
    obj = getattr(mod, name)
    # Cache on the module so subsequent lookups skip __getattr__.
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(__all__)
