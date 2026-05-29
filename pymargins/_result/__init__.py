"""pymargins._result package

Re-exports all result types for backward compatibility.
"""

from ._test import TestResult, AdjustedResults, adjust
from ._diagnostic import DiagnosticResult
from ._margins import MarginsResult, _join_fallback_reasons, _combine_results, compose_results

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
