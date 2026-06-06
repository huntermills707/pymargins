"""Transform pipeline stages for bootstrap inference.

Stages are data transforms that the bootstrap re-derives on every
replicate.  They compose in user-given order and declare their contract
via the :class:`Stage` protocol.
"""

from __future__ import annotations

from ._base import IdentityStage, Stage
from ._filters import drop_outliers, trim
from ._reimpute import reimpute

__all__ = [
    "Stage",
    "IdentityStage",
    "reimpute",
    "drop_outliers",
    "trim",
]
