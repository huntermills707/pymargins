"""Pure helper functions for atom enumeration and outcome slicing.

These are bound back onto the ``Margins`` class as methods (or
staticmethods) so they can be called with ``self.*`` syntax, but they
live here to keep ``_session.py`` focused on the public API and
construction logic.
"""

from __future__ import annotations
from typing import Optional, Union, Any

import jax.numpy as jnp
import numpy as np


def _enumerate_groups(
    session,
    scenario: dict,
    base_data,
    variable_metadata: dict,
):
    """Resolve ``scenario['over']`` into a list of (group_label, df) pairs.

    Returns a singleton ``[(None, base_data)]`` when no ``over`` is set,
    so downstream code has a uniform shape.
    """
    over_spec = scenario.get("over")
    if over_spec is None:
        return [(None, base_data)], None
    over_keys = [over_spec] if isinstance(over_spec, str) else list(over_spec)
    unknown = set(over_keys) - set(variable_metadata.keys())
    if unknown:
        raise ValueError(
            f"Unknown over variable(s): {sorted(unknown)}. "
            f"Known variables: {sorted(variable_metadata.keys())}."
        )
    if not hasattr(base_data, "groupby"):
        raise TypeError(
            f"over= requires base_data to support groupby, got {type(base_data).__name__}"
        )
    groups = [(g, gdf) for g, gdf in base_data.groupby(over_keys, sort=True)]
    if not groups:
        raise ValueError(
            f"over={over_keys!r} produced no groups; base data may be empty."
        )
    return groups, over_keys


def _format_atom_label(
    session,
    group_label,
    over_keys: Optional[list[str]],
    suffix: Optional[str],
) -> Optional[str]:
    """Build a stable label for one atom of a stacked estimand.

    Combines an over-group identifier (``"region=west"``) with an
    optional suffix (a grid index for atexog grids, a variable name
    for multi-variable slopes). Returns ``None`` when the atom is
    unique and unlabeled.
    """
    parts: list[str] = []
    if over_keys is not None:
        gl = group_label if isinstance(group_label, tuple) else (group_label,)
        parts.extend(f"{k}={v}" for k, v in zip(over_keys, gl))
    if suffix is not None:
        parts.append(suffix)
    return ", ".join(parts) if parts else None


def _finalize_atoms(
    session,
    atoms: list[tuple[Optional[str], Callable]],
) -> tuple[Callable, Optional[list[str]]]:
    """Reduce a list of (label, h_atom) pairs to (h, labels).

    Single atom: return its h directly with no labels. Multiple atoms:
    stack into a vector estimand and return the labels list.

    .. note:: Performance
       The current implementation uses a Python list comprehension
       inside ``jnp.stack``. For many atoms (>50), a ``jax.vmap`` over
       a single parametrized function would be faster because it avoids
       per-atom Python overhead and enables XLA fusion. This is a known
       optimization opportunity tracked in CODE_AUDIT §3.4.
    """
    if len(atoms) == 1:
        label = atoms[0][0]
        return atoms[0][1], ([label] if label is not None else None)
    individual_h = [h for _, h in atoms]
    labels = [lab for lab, _ in atoms]
    def h_vector(beta):
        return jnp.stack([hi(beta) for hi in individual_h])
    return h_vector, labels


def _slice_by_outcome(
    session,
    result_data: dict,
    outcome: Union[int, list[int]],
) -> dict:
    """Slice result arrays to the requested outcome indices.

    For multi-outcome models (MNLogit, OrderedModel), the inference
    engine returns estimates/SEs/CIs for all outcomes. This helper
    subsets them and updates labels accordingly.
    """
    n_outcomes = session.adapter.n_outcomes
    labels = session.adapter.outcome_labels or [str(i) for i in range(n_outcomes)]

    keys = [outcome] if isinstance(outcome, int) else list(outcome)
    for k in keys:
        if not (0 <= k < n_outcomes):
            raise ValueError(
                f"Outcome index {k} is out of range for model with "
                f"{n_outcomes} outcomes (valid: 0..{n_outcomes - 1})."
            )
    idx = np.asarray(keys, dtype=int)

    def _slice(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        squeeze = len(idx) == 1
        if arr.ndim == 1:
            out = arr[idx]
            return out.item() if squeeze and out.ndim == 0 else out
        elif arr.ndim == 2:
            # (n_atoms, n_outcomes) or (n_outcomes, n_params)
            # We want to slice along the outcome axis. Heuristic:
            # if the last dim equals n_outcomes, slice last axis.
            if arr.shape[-1] == n_outcomes:
                out = arr[..., idx]
                return out.squeeze(-1) if squeeze else out
            elif arr.shape[0] == n_outcomes:
                out = arr[idx]
                return out.squeeze(0) if squeeze else out
            else:
                return arr  # Can't determine outcome axis
        elif arr.ndim == 3:
            # (n_atoms, n_outcomes, n_params) or (n_sim, n_atoms, n_outcomes)
            if arr.shape[-1] == n_outcomes:
                out = arr[..., idx]
                return out.squeeze(-1) if squeeze else out
            elif arr.shape[1] == n_outcomes:
                out = arr[:, idx]
                return out.squeeze(1) if squeeze else out
            else:
                return arr
        return arr

    result = dict(result_data)
    for key in ("estimate", "std_error", "conf_int_lower", "conf_int_upper",
                "gradient", "draws", "kappa"):
        if key in result:
            result[key] = _slice(result[key])

    # Update labels
    meta = result.get("estimand_metadata", {})
    old_labels = meta.get("labels")
    if old_labels is not None:
        new_labels = []
        for lab in old_labels:
            for k in idx:
                suffix = labels[k]
                new_labels.append(f"{lab} ({suffix})" if lab else suffix)
        meta = dict(meta)
        meta["labels"] = new_labels
        meta["outcome_sliced"] = True
        result["estimand_metadata"] = meta

    return result
