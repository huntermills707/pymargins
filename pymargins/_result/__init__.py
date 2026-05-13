"""pymargins._result package

Re-exports all result types for backward compatibility.
"""

from ._test import TestResult
from ._diagnostic import DiagnosticResult
from ._margins import MarginsResult, _join_fallback_reasons, _combine_results

__all__ = [
    "TestResult",
    "DiagnosticResult",
    "MarginsResult",
    "_join_fallback_reasons",
    "_combine_results",
]
