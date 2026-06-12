"""The Plan object.

Implements the pre-registration contract from design §4 and req. §4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Plan:
    """Immutable, fully-resolved analysis plan.

    Everything that defines *the analysis* is frozen here.  Execution knobs
    (``n_jobs``, ``progress_bar``, cache/retention) are excluded.
    """

    # Identifiers
    package_version: str = "0.4.0"
    plan_hash: str = ""

    # Graph topology
    node_kinds: tuple[str, ...] = ()
    node_hashes: tuple[str, ...] = ()

    # Estimand frame (declared + resolved)
    at: str | None = None
    scale: str | None = None
    method_declared: str = ""
    method_resolved: str = ""
    method_resolution_reason: str = ""
    vcov: Any | None = None
    ci: str = ""
    level: float = 0.95
    B: int = 0
    n_sim: int = 0
    seed: int | None = None

    # Data identity
    data_fingerprint: str = ""

    # Callable-hash status
    unhashable_callable: bool = False

    # Population metadata
    population_note: str | None = None

    # Constants overrides
    constants_overrides: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self):
        if not self.plan_hash:
            # Compute canonical hash
            hasher = hashlib.sha256()
            payload = {
                "version": self.package_version,
                "recipe": 1,
                "nodes": list(self.node_kinds),
                "node_hashes": list(self.node_hashes),
                "at": self.at,
                "scale": self.scale,
                "method_declared": self.method_declared,
                "method_resolved": self.method_resolved,
                "method_resolution_reason": self.method_resolution_reason,
                "vcov": str(self.vcov),
                "ci": self.ci,
                "level": self.level,
                "B": self.B,
                "n_sim": self.n_sim,
                "seed": self.seed,
                "data_fingerprint": self.data_fingerprint,
                "unhashable_callable": self.unhashable_callable,
                "population_note": self.population_note,
                "constants_overrides": list(self.constants_overrides),
            }
            hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
            object.__setattr__(self, "plan_hash", hasher.hexdigest())

    @property
    def hash(self) -> str:
        """Short display hash with recipe version suffix."""
        return f"{self.plan_hash[:7]}@1"

    def describe(self) -> str:
        """Human-readable plan summary."""
        lines = [
            f"Plan {self.hash}",
            f"  method: {self.method_resolved} (declared: {self.method_declared})",
        ]
        if self.method_resolution_reason:
            lines.append(f"  resolution reason: {self.method_resolution_reason}")
        lines.extend([
            f"  scale: {self.scale}",
            f"  at: {self.at}",
            f"  ci: {self.ci}",
            f"  level: {self.level}",
            f"  B: {self.B}",
            f"  n_sim: {self.n_sim}",
            f"  seed: {self.seed}",
            f"  data fingerprint: {self.data_fingerprint[:16]}..." if self.data_fingerprint else "  data fingerprint: (none)",
        ])
        if self.unhashable_callable:
            lines.append("  unhashable-callable: marked")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Canonical JSON serialization."""
        payload = {
            "package_version": self.package_version,
            "plan_hash": self.plan_hash,
            "node_kinds": list(self.node_kinds),
            "node_hashes": list(self.node_hashes),
            "at": self.at,
            "scale": self.scale,
            "method_declared": self.method_declared,
            "method_resolved": self.method_resolved,
            "method_resolution_reason": self.method_resolution_reason,
            "vcov": str(self.vcov),
            "ci": self.ci,
            "level": self.level,
            "B": self.B,
            "n_sim": self.n_sim,
            "seed": self.seed,
            "data_fingerprint": self.data_fingerprint,
            "unhashable_callable": self.unhashable_callable,
            "population_note": self.population_note,
            "constants_overrides": list(self.constants_overrides),
        }
        return json.dumps(payload, sort_keys=True, indent=2)
