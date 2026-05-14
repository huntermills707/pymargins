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

Internal modules (prefixed with _) hold the numerical kernels and engine.
End users should not import from these directly. Adapter implementers
should import from _adapter, _gradients (for make_*_jvp_wrapper helpers),
and the type aliases in _adapter and _gradients.
"""

from .margins import Margins
from ._result import MarginsResult, TestResult, DiagnosticResult
from ._adapters import register_adapter
from ._adapter import (
    ModelAdapter,
    GLMAdapter,
    LinearPredictionAdapter,
    WrappedFDAdapter,
    BootstrapOnlyAdapter,
    VariableInfo,
    InferenceMethod,
)
from ._gradients import (
    make_predict_with_fd_jvp,
    make_glm_jvp_wrapper,
    GradientBackend,
)
from .scenarios import (
    pairwise,
    reference,
    at_levels,
    grid,
    did,
    diff,
    all_pairwise,
)
from .matching import PysmatchClient

__all__ = [
    # Main entry point
    "Margins",
    # Result types
    "MarginsResult",
    "TestResult",
    "DiagnosticResult",
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
    "all_pairwise",
    # Matching support
    "PysmatchClient",
]

__version__ = "0.0.1"
