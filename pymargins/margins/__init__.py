"""pymargins.margins

Re-exports the public API from the split sub-package so that
``from pymargins import Margins`` and ``import pymargins.margins``
continue to work unchanged.
"""

from .._inference import run_inference
from ._session import Margins, _NOT_GIVEN

__all__ = ["Margins", "run_inference", "_NOT_GIVEN"]
