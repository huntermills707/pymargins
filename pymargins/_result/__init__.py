"""pymargins._result package

Re-exports all result types for backward compatibility.
"""

from ._diagnostic import DiagnosticResult
from ._margins import (
    MarginsResult,
    _combine_results,
    _join_fallback_reasons,
    compose_results,
)
from ._test import AdjustedResults, TestResult, adjust

__all__ = [
    "TestResult",
    "AdjustedResults",
    "adjust",
    "DiagnosticResult",
    "MarginsResult",
    "_join_fallback_reasons",
    "_combine_results",
    "compose_results",
]
