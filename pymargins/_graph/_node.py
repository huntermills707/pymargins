"""Computation graph nodes.

Implements the node contract from design §2.1 and req. §2.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from pymargins._tabular import fingerprint_frame

FanKind = Literal["imputation"]


def _fingerprint(value: Any) -> str:
    """Stable content fingerprint for a single value."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool:true" if value else "bool:false"
    if isinstance(value, (int, float, str)):
        return f"{type(value).__name__}:{value}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_fingerprint(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda x: str(x[0]))
        return (
            "{"
            + ",".join(f"{_fingerprint(k)}:{_fingerprint(v)}" for k, v in items)
            + "}"
        )
    if isinstance(value, np.ndarray):
        hasher = hashlib.sha256()
        hasher.update(str(value.shape).encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(value.tobytes())
        return hasher.hexdigest()
    # For pandas DataFrame / Series
    if hasattr(value, "shape") and hasattr(value, "columns"):
        # DataFrame fingerprint — shared with adapters/compiler
        return fingerprint_frame(value)
    if hasattr(value, "dtype") and hasattr(value, "tobytes"):
        # Series-like
        hasher = hashlib.sha256()
        hasher.update(str(value.shape).encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        arr = value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)
        if arr.dtype == object:
            for v in arr:
                hasher.update(str(v).encode("utf-8"))
        else:
            hasher.update(arr.tobytes())
        return hasher.hexdigest()
    # Callable / opaque object
    if callable(value):
        qual = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
        if qual:
            return f"callable:{qual}"
        src = getattr(value, "__code__", None)
        if src:
            return f"callable:code:{src.co_filename}:{src.co_firstlineno}"
        return "callable:unhashable"
    # SurveyDesign
    if type(value).__name__ == "SurveyDesign":
        hasher = hashlib.sha256()
        for attr in ("weights", "psu", "strata", "fpc", "fpc_is_fraction", "nest"):
            val = getattr(value, attr, None)
            if val is not None:
                hasher.update(str(attr).encode("utf-8"))
                hasher.update(_fingerprint(val).encode("utf-8"))
        return hasher.hexdigest()
    # Fallback: use class name + repr
    return f"repr:{type(value).__name__}:{repr(value)}"


@dataclass(frozen=True)
class Node:
    """Frozen, content-addressed graph node."""

    kind: str
    params: tuple[tuple[str, Any], ...] = ()
    inputs: tuple[Node, ...] = ()
    # Contract fields
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    alters_rows: bool = False
    emits_columns: tuple[str, ...] = ()
    fan: FanKind | None = None
    population_note: str | None = None
    # Payload is fingerprinted but not stored in the frozen fields
    _payload: Any = field(default=None, repr=False, compare=False, hash=False)
    _payload_fp: str = field(default="", repr=False, compare=True, hash=True)
    _hash: str = field(default="", repr=False, compare=False, hash=False)

    def __post_init__(self):
        # Ensure params are sorted for hash stability
        object.__setattr__(
            self, "params", tuple(sorted(self.params, key=lambda x: x[0]))
        )
        if not self._payload_fp and self._payload is not None:
            object.__setattr__(self, "_payload_fp", _fingerprint(self._payload))
        # Compute hash
        hasher = hashlib.sha256()
        hasher.update(self.kind.encode("utf-8"))
        for k, v in self.params:
            hasher.update(k.encode("utf-8"))
            hasher.update(_fingerprint(v).encode("utf-8"))
        for inp in self.inputs:
            hasher.update(inp.hash.encode("utf-8"))
        hasher.update(self._payload_fp.encode("utf-8"))
        object.__setattr__(self, "_hash", hasher.hexdigest())

    @property
    def hash(self) -> str:
        return self._hash

    def collect(self) -> Any:
        """Materialize the point-execution output of this node.

        For data-source nodes this returns the prepared DataFrame.
        For stage nodes this applies the stage to the collected inputs.
        Outputs are cached on the node instance so repeated calls are consistent
        (required for stochastic stages such as ``reimpute``). The cache is
        per-instance and disappears when the node is garbage collected.
        """
        if hasattr(self, "_collect_cache"):
            return self._collect_cache

        if self.kind == "input":
            result = self._payload
        elif self.kind == "trim":
            stage = self._payload
            parent = self.inputs[0].collect()
            result = stage.prepare(parent)
        elif self.kind == "drop_outliers":
            stage = self._payload
            parent = self.inputs[0].collect()
            result = stage.prepare(parent)
        elif self.kind == "match":
            matcher = self._payload
            parent = self.inputs[0].collect()
            result = matcher.matched_data
        elif self.kind == "reimpute":
            stage = self._payload
            parent = self.inputs[0].collect()
            result = stage.prepare(parent)
        else:
            raise NotImplementedError(
                f"Node.collect() not yet implemented for kind={self.kind!r}."
            )

        object.__setattr__(self, "_collect_cache", result)
        return result

    def with_payload(self, payload: Any) -> Node:
        """Return a new node with the given payload (used by step constructors)."""
        return Node(
            kind=self.kind,
            params=self.params,
            inputs=self.inputs,
            requires=self.requires,
            provides=self.provides,
            alters_rows=self.alters_rows,
            emits_columns=self.emits_columns,
            fan=self.fan,
            population_note=self.population_note,
            _payload=payload,
        )
