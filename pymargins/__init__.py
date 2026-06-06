"""
pymargins — expert-mode marginal effects for Python.

Quick reference
---------------
Wrap a fitted model in a Margins session, declare your analytical posture
(scale, vcov, level, at), then compute predictions, slopes, and
contrasts via the session's methods::

    from pymargins import Margins
    m = Margins.log_scale(fitted_glm, vcov="HC3")
    rr = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": "treated"}},
            {"atexog": {"treatment": "control"}},
        ],
        contrasts=[+1, -1],
    )

Public API
----------
- ``Margins`` — main session class
- ``MarginsResult`` — output of predict/dydx/contrasts/evaluate
- ``TestResult`` — output of result.test() and result.joint_test()
- ``DiagnosticResult`` — output of m.diagnose()
- ``VariableInfo`` — per-variable metadata used by adapters
- ``register_adapter`` — register a custom adapter for auto-detection

Internal modules (prefixed with _) hold the numerical kernels and engine;
end users should not import from these directly.

Adapter authors writing a custom adapter subclass one of the base shapes
re-exported above (``ModelAdapter``, ``GLMAdapter``, etc.) plus the
``make_*_jvp_wrapper`` gradient helpers, and register it with
``register_adapter``. The concrete built-in adapters are available, lazily,
from the public ``pymargins.adapters`` module — you rarely need them since
``Margins(model)`` auto-detects the right one; import explicitly only to
override detection or select a non-default scale.
"""

from ._adapter import (
    BootstrapOnlyAdapter,
    GLMAdapter,
    InferenceMethod,
    LinearPredictionAdapter,
    ModelAdapter,
    VariableInfo,
    WrappedFDAdapter,
)
from ._adapters import register_adapter
from ._gradients import (
    GradientBackend,
    make_glm_jvp_wrapper,
    make_predict_with_fd_jvp,
)
from ._result import (
    AdjustedResults,
    DiagnosticResult,
    ImputationDiagnostic,
    MarginsResult,
    TestResult,
    adjust,
    pool_imputations,
)
from .margins import Margins
from ._transforms import drop_outliers, reimpute, trim
from .matching import PysmatchClient
from .scenarios import (
    all_pairwise,
    at_levels,
    did,
    diff,
    diff_matrix,
    grid,
    pairwise,
    reference,
)
from .survey import SurveyDesign

__all__ = [
    # Main entry point
    "Margins",
    # Result types
    "MarginsResult",
    "TestResult",
    "DiagnosticResult",
    "AdjustedResults",
    "adjust",
    "pool_imputations",
    "ImputationDiagnostic",
    # Adapter interface (for extension)
    "ModelAdapter",
    "GLMAdapter",
    "LinearPredictionAdapter",
    "WrappedFDAdapter",
    "BootstrapOnlyAdapter",
    "VariableInfo",
    "register_adapter",
    # Helpers for adapter implementers
    "make_predict_with_fd_jvp",
    "make_glm_jvp_wrapper",
    # Type aliases (for type hints in user code)
    "InferenceMethod",
    "GradientBackend",
    # Scenario/contrast helpers
    "pairwise",
    "reference",
    "at_levels",
    "grid",
    "did",
    "diff",
    "diff_matrix",
    "all_pairwise",
    # Matching support
    "PysmatchClient",
    # Survey design
    "SurveyDesign",
    # Transform pipeline stages
    "reimpute",
    "drop_outliers",
    "trim",
    # Package metadata
    "__version__",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("pymargins")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0+unknown"
