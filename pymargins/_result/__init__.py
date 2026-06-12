"""pymargins._result package

Re-exports all result types for backward compatibility.
"""

from ._diagnostic import DiagnosticResult
from ._graphresult import GraphResult
from ._margins import (
    MarginsResult,
    _combine_results,
    _join_fallback_reasons,
    compose_results,
)
from ._pooling import ImputationDiagnostic, pool_imputations
from ._test import AdjustedResults, TestResult, adjust

__all__ = [
    "TestResult",
    "AdjustedResults",
    "adjust",
    "DiagnosticResult",
    "GraphResult",
    "MarginsResult",
    "ImputationDiagnostic",
    "pool_imputations",
    "_join_fallback_reasons",
    "_combine_results",
    "compose_results",
]
