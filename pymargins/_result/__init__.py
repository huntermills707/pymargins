"""pymargins._result package

Re-exports result types for the new engine.
"""

from ._graphresult import GraphResult
from ._pooling import ImputationDiagnostic, pool_imputations
from ._test import AdjustedResults, TestResult, adjust

__all__ = [
    "TestResult",
    "AdjustedResults",
    "adjust",
    "GraphResult",
    "ImputationDiagnostic",
    "pool_imputations",
]
