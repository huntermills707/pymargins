"""
pymargins.scenarios — Helper functions for building scenario lists and contrast weights.

These are pure factories that return data structures accepted by
`Margins.contrasts()`, `Margins.evaluate()`, and `Margins.predict()`.
They keep common contrast patterns (pairwise, reference-level, diff-in-diff,
grid predictions) explicit and less error-prone.

Examples
--------
>>> from pymargins.scenarios import pairwise, reference, did, grid, at_levels
>>>
>>> # Simple pairwise risk difference
>>> scenarios, contrasts = pairwise("treatment", [1, 0])
>>> m.contrasts(scenarios=scenarios, contrasts=contrasts)
>>>
>>> # Reference-level contrasts (each vs baseline)
>>> scenarios, contrasts = reference("region", ["north", "south", "east"], ref_level="north")
>>> m.contrasts(scenarios=scenarios, contrasts=contrasts)
>>>
>>> # Diff-in-diff
>>> scenarios, contrasts = did("treatment", "post")
>>> m.contrasts(scenarios=scenarios, contrasts=contrasts)
>>>
>>> # Predict at every level of a categorical variable
>>> scenarios = at_levels("region", levels=["north", "south", "east", "west"])
>>> m.predict(scenarios=scenarios)
>>>
>>> # Grid of counterfactuals
>>> scenarios = grid(age=[30, 50, 70], treatment=[0, 1])
>>> m.predict(scenarios=scenarios)
"""

from __future__ import annotations
from typing import Optional, Union
import itertools

import numpy as np


# ---------------------------------------------------------------------------
# Single-variable helpers
# ---------------------------------------------------------------------------

def pairwise(
    variable: str,
    values,
    *,
    label_fmt: Optional[str] = None,
    **fixed,
):
    """Build two scenarios and a [+1, -1] contrast for a pairwise comparison.

    Parameters
    ----------
    variable : str
        Variable name to contrast.
    values : list or tuple of length 2
        The two levels to compare (e.g. ``[1, 0]`` or ``["treated", "control"]``).
    label_fmt : str, optional
        Format string for auto-generated labels.  If None, defaults to
        ``"{var}={val}"``.
    **fixed
        Additional variables held constant across both scenarios
        (e.g. ``female=0`` for MER-style contrasts).

    Returns
    -------
    scenarios : list[dict]
    contrasts : list[float]

    Examples
    --------
    >>> scenarios, contrasts = pairwise("treatment", [1, 0])
    >>> scenarios
    [
        {"atexog": {"treatment": 1}, "label": "treatment=1"},
        {"atexog": {"treatment": 0}, "label": "treatment=0"},
    ]
    >>> contrasts
    [1, -1]

    >>> # With fixed covariates (MER)
    >>> scenarios, contrasts = pairwise("black", [1, 0], female=0, age=40)
    """
    if len(values) != 2:
        raise ValueError(f"pairwise() expects exactly 2 values, got {len(values)}")

    if label_fmt is None:
        label_fmt = "{var}={val}"

    scenarios = []
    for val in values:
        atexog = dict(fixed)
        atexog[variable] = val
        label = label_fmt.format(var=variable, val=val)
        scenarios.append({"atexog": atexog, "label": label})

    return scenarios, [+1, -1]


def reference(
    variable: str,
    levels,
    *,
    ref_level=None,
    label_fmt: Optional[str] = None,
    **fixed,
):
    """Build reference-level contrasts: each level vs a common baseline.

    Parameters
    ----------
    variable : str
        Variable name.
    levels : list or tuple
        All levels of the variable.  The first element is used as the
        reference unless ``ref_level`` is provided.
    ref_level : any, optional
        Which level to treat as the reference/baseline.  Defaults to the
        first element of ``levels``.
    label_fmt : str, optional
        Format string for labels.  Defaults to ``"{var}={val}"``.
    **fixed
        Additional variables held constant.

    Returns
    -------
    scenarios : list[dict]
        One scenario per level (reference level first).
    contrasts : dict[str, list[float]]
        Named contrast vectors, one per non-reference level.

    Examples
    --------
    >>> scenarios, contrasts = reference("region", ["north", "south", "east"])
    >>> contrasts
    {
        "south_vs_north": [ -1, +1,  0],
        "east_vs_north" : [ -1,  0, +1],
    }
    """
    if not levels:
        raise ValueError("reference() requires at least one level")

    levels = list(levels)
    if ref_level is None:
        ref_level = levels[0]

    if ref_level not in levels:
        raise ValueError(
            f"ref_level {ref_level!r} not found in levels {levels}"
        )

    if label_fmt is None:
        label_fmt = "{var}={val}"

    # Reorder so ref_level is first
    ordered = [ref_level] + [lv for lv in levels if lv != ref_level]

    scenarios = []
    for val in ordered:
        atexog = dict(fixed)
        atexog[variable] = val
        label = label_fmt.format(var=variable, val=val)
        scenarios.append({"atexog": atexog, "label": label})

    n = len(scenarios)
    contrasts = {}
    ref_label = label_fmt.format(var=variable, val=ref_level)
    for i, val in enumerate(ordered[1:], start=1):
        name = f"{val}_vs_{ref_level}"
        # Clean up name for use as a dict key
        name = str(name).replace(" ", "_").replace("-", "_")
        w = [0.0] * n
        w[0] = -1.0
        w[i] = +1.0
        contrasts[name] = w

    return scenarios, contrasts


def at_levels(
    variable: str,
    *,
    levels,
    label_fmt: Optional[str] = None,
    **fixed,
):
    """Build one scenario per level of a categorical/binary variable.

    Useful for passing to ``predict()`` to get predictions at every level,
    or as input to custom contrast matrices.

    Parameters
    ----------
    variable : str
        Variable name.
    levels : list or tuple
        Levels to generate scenarios for.
    label_fmt : str, optional
        Format string for labels.  Defaults to ``"{var}={val}"``.
    **fixed
        Additional variables held constant.

    Returns
    -------
    scenarios : list[dict]

    Examples
    --------
    >>> scenarios = at_levels("region", levels=["north", "south", "east"])
    >>> m.predict(scenarios=scenarios)
    """
    if not levels:
        raise ValueError("at_levels() requires at least one level")

    if label_fmt is None:
        label_fmt = "{var}={val}"

    scenarios = []
    for val in levels:
        atexog = dict(fixed)
        atexog[variable] = val
        label = label_fmt.format(var=variable, val=val)
        scenarios.append({"atexog": atexog, "label": label})

    return scenarios


# ---------------------------------------------------------------------------
# Multi-variable / factorial helpers
# ---------------------------------------------------------------------------

def grid(*, label_fmt: Optional[str] = None, **variables):
    """Cartesian product of variable values into a scenario list.

    Parameters
    ----------
    label_fmt : str, optional
        Format string for labels.  If None, defaults to a comma-joined
        ``"var=val"`` pattern.
    **variables
        Keyword arguments where each key is a variable name and each value
        is a list of levels.

    Returns
    -------
    scenarios : list[dict]

    Examples
    --------
    >>> scenarios = grid(age=[30, 50, 70], treatment=[0, 1])
    >>> len(scenarios)
    6
    >>> scenarios[0]
    {"atexog": {"age": 30, "treatment": 0}, "label": "age=30, treatment=0"}
    """
    if not variables:
        raise ValueError("grid() requires at least one variable")

    names = list(variables.keys())
    value_lists = [variables[name] for name in names]

    scenarios = []
    for combo in itertools.product(*value_lists):
        atexog = dict(zip(names, combo))
        if label_fmt is None:
            label = ", ".join(f"{n}={v}" for n, v in zip(names, combo))
        else:
            label = label_fmt.format(**atexog)
        scenarios.append({"atexog": atexog, "label": label})

    return scenarios


def did(
    treatment: str,
    time: str,
    *,
    treated_level=1,
    control_level=0,
    post_level=1,
    pre_level=0,
    label_fmt: Optional[str] = None,
    **fixed,
):
    """Build scenarios and contrast weights for a difference-in-differences design.

    The classic 2×2 DID contrast is:

        (Treated_post − Treated_pre) − (Control_post − Control_pre)

    which corresponds to weights ``[+1, -1, -1, +1]`` over the four
    scenarios ordered as:

        0. Treated_post
        1. Treated_pre
        2. Control_post
        3. Control_pre

    Parameters
    ----------
    treatment : str
        Name of the treatment indicator variable.
    time : str
        Name of the time/post indicator variable.
    treated_level, control_level : any, default 1, 0
        Values of ``treatment`` for treated and control units.
    post_level, pre_level : any, default 1, 0
        Values of ``time`` for post and pre periods.
    label_fmt : str, optional
        Format string for labels.  Defaults to ``"{treatment}={tval}, {time}={pval}"``.
    **fixed
        Additional variables held constant.

    Returns
    -------
    scenarios : list[dict]
    contrasts : dict[str, list[float]]
        Contains a single entry ``"did"`` with the contrast weights.

    Examples
    --------
    >>> scenarios, contrasts = did("treat", "post")
    >>> m.contrasts(scenarios=scenarios, contrasts=contrasts)
    """
    if label_fmt is None:
        label_fmt = "{treatment}={tval}, {time}={pval}"

    combos = [
        (treated_level, post_level),
        (treated_level, pre_level),
        (control_level, post_level),
        (control_level, pre_level),
    ]

    scenarios = []
    for tval, pval in combos:
        atexog = dict(fixed)
        atexog[treatment] = tval
        atexog[time] = pval
        label = label_fmt.format(treatment=treatment, time=time, tval=tval, pval=pval)
        scenarios.append({"atexog": atexog, "label": label})

    contrasts = {"did": [+1, -1, -1, +1]}
    return scenarios, contrasts


# ---------------------------------------------------------------------------
# Contrast-weight utilities
# ---------------------------------------------------------------------------

def diff(n: int):
    """Return a standard pairwise difference contrast vector.

    Parameters
    ----------
    n : int
        Number of scenarios.  Must be >= 2.

    Returns
    -------
    weights : list[float]
        ``[+1, -1, 0, 0, ...]`` — first scenario minus second.

    Examples
    --------
    >>> diff(2)
    [1, -1]
    >>> diff(4)
    [1, -1, 0, 0]
    """
    if n < 2:
        raise ValueError("diff() requires n >= 2")
    return [+1, -1] + [0.0] * (n - 2)


def diff_matrix(k: int, kind: str = "reference") -> np.ndarray:
    """Build a contrast matrix for a k-level factor.

    Parameters
    ----------
    k : int
        Number of levels (must be >= 2).
    kind : {"reference", "pairwise"}, default "reference"
        * ``"reference"`` — returns a ``(k-1, k)`` matrix where each row
          is ``level[i] - level[0]``.
        * ``"pairwise"`` — returns a ``(k*(k-1)/2, k)`` matrix with all
          pairwise differences.

    Returns
    -------
    ndarray

    Examples
    --------
    >>> diff_matrix(3, kind="reference")
    array([[-1.,  1.,  0.],
           [-1.,  0.,  1.]])
    >>> diff_matrix(3, kind="pairwise")
    array([[-1.,  1.,  0.],
           [-1.,  0.,  1.],
           [ 0., -1.,  1.]])
    """
    if k < 2:
        raise ValueError("diff_matrix requires k >= 2")
    if kind not in ("reference", "pairwise"):
        raise ValueError(f"kind must be 'reference' or 'pairwise', got {kind!r}")

    if kind == "reference":
        C = np.zeros((k - 1, k), dtype=float)
        for i in range(1, k):
            C[i - 1, 0] = -1.0
            C[i - 1, i] = +1.0
        return C

    # pairwise
    n_contrasts = k * (k - 1) // 2
    C = np.zeros((n_contrasts, k), dtype=float)
    row = 0
    for i in range(k):
        for j in range(i + 1, k):
            C[row, i] = -1.0
            C[row, j] = +1.0
            row += 1
    return C


def all_pairwise(
    variables,
    values_list,
    *,
    label_fmt: Optional[str] = None,
    **fixed,
):
    """Build all scenario combinations for a factorial design and return
    named pairwise contrasts for each stratum.

    For a single variable you may pass the name directly and the levels
    as a flat list (``all_pairwise("x", [0, 1])``); for multiple
    variables pass lists (``all_pairwise(["x", "y"], [[0, 1], ["A", "B"]])``).
    

    Parameters
    ----------
    variables : list[str]
        Variable names (e.g. ``["treatment", "sex"]``).
    values_list : list[list]
        Levels for each variable (e.g. ``[[1, 0], ["M", "F"]]``).
    label_fmt : str, optional
        Format string for labels.  Defaults to comma-joined ``"var=val"``.
    **fixed
        Additional variables held constant.

    Returns
    -------
    scenarios : list[dict]
    contrasts : dict[str, list[float]]
        Named pairwise contrasts, one per non-baseline combination.
        The first combination (first level of each variable) is the
        reference.

    Examples
    --------
    >>> scenarios, contrasts = all_pairwise(
    ...     ["treatment", "sex"],
    ...     [[1, 0], ["M", "F"]],
    ... )
    >>> len(scenarios)
    4
    >>> contrasts
    {
        "treatment=0, sex=M_vs_treatment=1, sex=M": [...],
        ...
    }
    """
    # Normalise single-variable shorthand to list form
    if isinstance(variables, str):
        variables = [variables]
        values_list = [values_list]

    if len(variables) != len(values_list):
        raise ValueError(
            "variables and values_list must have the same length"
        )

    combos = list(itertools.product(*values_list))
    if not combos:
        raise ValueError("all_pairwise() requires at least one combination")

    scenarios = []
    for combo in combos:
        atexog = dict(fixed)
        atexog.update(zip(variables, combo))
        if label_fmt is None:
            label = ", ".join(f"{v}={c}" for v, c in zip(variables, combo))
        else:
            label = label_fmt.format(**atexog)
        scenarios.append({"atexog": atexog, "label": label})

    n = len(scenarios)
    contrasts = {}
    ref_label = scenarios[0]["label"]
    for i in range(1, n):
        name = f"{scenarios[i]['label']}_vs_{ref_label}"
        name = name.replace(" ", "_").replace("-", "_")
        w = [0.0] * n
        w[0] = -1.0
        w[i] = +1.0
        contrasts[name] = w

    return scenarios, contrasts
