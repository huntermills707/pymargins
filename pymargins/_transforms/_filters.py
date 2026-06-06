"""Row-filtering stages: drop_outliers and trim."""

from __future__ import annotations

import pandas as pd


def drop_outliers(rule):
    """Create a ``drop_outliers`` pipeline stage.

    Drops rows that satisfy the user-supplied *rule*.  Under bootstrap
    the rule is re-applied to every resample, capturing detection
    variability.

    Parameters
    ----------
    rule : callable
        ``rule(frame) -> boolean Series`` or ``rule(frame) -> boolean mask``.
        Rows where the rule is *True* are dropped.

    Returns
    -------
    Stage
        A stage with ``alters_rows=True`` and ``requires_resampling=False``.
    """
    return _DropOutliersStage(rule)


class _DropOutliersStage:
    requires_resampling = False
    alters_rows = True
    emits_columns = ()
    source_data = None

    def __init__(self, rule):
        self._rule = rule

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        mask = self._rule(data)
        return data[~mask].reset_index(drop=True)

    def prepare_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        mask = self._rule(data)
        return data[~mask].reset_index(drop=True)


def trim(*, lower=None, upper=None, columns=None):
    """Create a ``trim`` pipeline stage.

    Drops rows where any of the specified *columns* falls outside the
    ``[lower, upper]`` bounds.  Under bootstrap the bounds are re-applied
    to every resample.

    Parameters
    ----------
    lower : float, optional
        Lower bound (inclusive).
    upper : float, optional
        Upper bound (inclusive).
    columns : list of str, optional
        Columns to check.  If None, all numeric columns are checked.

    Returns
    -------
    Stage
        A stage with ``alters_rows=True`` and ``requires_resampling=False``.
    """
    return _TrimStage(lower=lower, upper=upper, columns=columns)


class _TrimStage:
    requires_resampling = False
    alters_rows = True
    emits_columns = ()
    source_data = None

    def __init__(self, *, lower=None, upper=None, columns=None):
        self._lower = lower
        self._upper = upper
        self._columns = columns

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(data)

    def prepare_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(data)

    def _apply(self, data: pd.DataFrame) -> pd.DataFrame:
        cols = self._columns
        if cols is None:
            cols = list(data.select_dtypes(include="number").columns)
        mask = pd.Series(True, index=data.index)
        for c in cols:
            if c not in data.columns:
                continue
            if self._lower is not None:
                mask &= data[c] >= self._lower
            if self._upper is not None:
                mask &= data[c] <= self._upper
        return data[mask].reset_index(drop=True)
