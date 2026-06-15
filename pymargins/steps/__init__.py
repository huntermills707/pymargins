"""Wiring verbs for the computation graph.

Implements the step surface from design §4.1 and req. §2.
Each function returns a :class:`pymargins._graph.Node`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pymargins._graph._node import Node

# Re-export scenario helpers so users can write
#   from pymargins import steps
#   steps.at_levels(...)
# while the canonical location remains pymargins.scenarios.
from pymargins.scenarios import at_levels, did, grid, pairwise, reference

__all__ = [
    "input",
    "match",
    "reimpute",
    "trim",
    "drop_outliers",
    "impute",
    "imputed",
    "propensity",
    # scenario re-exports
    "at_levels",
    "did",
    "grid",
    "pairwise",
    "reference",
]


def input(
    df: pd.DataFrame,
    *,
    design: Any | None = None,
    cluster: Any | None = None,
    block: Any | None = None,
    block_type: str = "moving",
) -> Node:
    """Resampling root node.

    Parameters
    ----------
    df : pandas.DataFrame
        The training / analysis data.
    design : SurveyDesign, optional
        Complex survey design declaration.
    cluster : array-like, optional
        Exchangeable cluster identifiers.
    block : int, optional
        Block-bootstrap block length.
    block_type : str, default "moving"
        Block type for block bootstrap ("moving", "circular", "nonoverlapping").

    Returns
    -------
    Node
        A node of kind ``"input"`` carrying *df* as payload.
    """
    params = []
    if design is not None:
        params.append(("design", design))
    if cluster is not None:
        params.append(("cluster", cluster))
    if block is not None:
        params.append(("block", block))
    if block_type != "moving":
        params.append(("block_type", block_type))
    return Node(
        kind="input",
        params=tuple(params),
        _payload=df,
    )


def match(node: Node, matcher: Any) -> Node:
    """Matching step.

    Wraps today's matcher protocol (``matched_data``, ``cluster_ids``,
    ``rematch``).

    Parameters
    ----------
    node : Node
        Input data node.
    matcher : object
        Any object exposing ``matched_data``, ``cluster_ids``, and
        ``rematch(data)``.

    Returns
    -------
    Node
        A node with ``alters_rows=True`` and a population note.
    """
    pop_note = getattr(matcher, "population_note", None)
    if pop_note is None:
        pop_note = "matched sample"
    return Node(
        kind="match",
        inputs=(node,),
        alters_rows=True,
        population_note=pop_note,
        _payload=matcher,
    )


def reimpute(node: Node, imputer: Any) -> Node:
    """Bootstrap-scoped re-imputation step.

    On every bootstrap replicate the *imputer* is called fresh on the
    resampled data.  This is the v0.3.0 ``reimpute`` stage, now exposed
    as a graph node.

    Parameters
    ----------
    node : Node
        Input data node (should carry the incomplete frame as payload).
    imputer : callable
        ``imputer(frame) -> frame``.

    Returns
    -------
    Node
        A node with ``requires_resampling=True`` (implicit via the stage
        payload).
    """
    from pymargins._transforms._reimpute import _ReimputeStage

    # The incomplete frame is the collected output of the parent node.
    # For eager construction we collect here; the plan hash records the
    # fingerprint so template mismatches are caught at compile time.
    incomplete = node.collect() if hasattr(node, "collect") else None
    if incomplete is None:
        raise ValueError(
            "reimpute requires a data node whose collect() returns a DataFrame."
        )
    stage = _ReimputeStage(imputer, incomplete)
    return Node(
        kind="reimpute",
        inputs=(node,),
        alters_rows=False,
        _payload=stage,
    )


def trim(node: Node, *, lower=None, upper=None, columns=None) -> Node:
    """Trim rows where specified columns fall outside bounds.

    Parameters
    ----------
    node : Node
        Input data node.
    lower : float, optional
        Lower bound (inclusive).
    upper : float, optional
        Upper bound (inclusive).
    columns : list of str, optional
        Columns to check.  If None, all numeric columns are checked.

    Returns
    -------
    Node
        A node with ``alters_rows=True``.
    """
    from pymargins._transforms._filters import _TrimStage

    stage = _TrimStage(lower=lower, upper=upper, columns=columns)
    return Node(
        kind="trim",
        inputs=(node,),
        params=(("lower", lower), ("upper", upper), ("columns", columns)),
        alters_rows=True,
        _payload=stage,
    )


def drop_outliers(node: Node, rule: Any) -> Node:
    """Drop rows that satisfy a rule.

    Parameters
    ----------
    node : Node
        Input data node.
    rule : callable
        ``rule(frame) -> boolean mask``.  Rows where True are dropped.

    Returns
    -------
    Node
        A node with ``alters_rows=True``.
    """
    from pymargins._transforms._filters import _DropOutliersStage

    stage = _DropOutliersStage(rule)
    return Node(
        kind="drop_outliers",
        inputs=(node,),
        alters_rows=True,
        _payload=stage,
    )


def impute(node: Node, imputer: Any, m: int) -> Node:
    """Fan node for multiple imputation.

    Raises
    ------
    NotImplementedError
        This step lands in 0.5.0.
    """
    raise NotImplementedError("lands in 0.5.0/0.6.0")


def imputed(datasets: list[pd.DataFrame]) -> Node:
    """Precomputed fan node from external imputations.

    Raises
    ------
    NotImplementedError
        This step lands in 0.5.0.
    """
    raise NotImplementedError("lands in 0.5.0/0.6.0")


def propensity(node: Node, spec: str) -> Node:
    """Propensity-score estimation step.

    Raises
    ------
    NotImplementedError
        This step lands in 0.6.0.
    """
    raise NotImplementedError("lands in 0.5.0/0.6.0")
