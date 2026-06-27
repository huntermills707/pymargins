"""Matching-library adapters for pymargins.

This module provides thin wrappers that translate popular matching libraries
into the ``MatchingClient`` protocol expected by ``GComputation(..., matching=...)``.

Reference implementation
------------------------
- :class:`PysmatchClient` — wraps `pysmatch <https://pypi.org/project/pysmatch/>`_,
  the active successor to the well-cited ``pymatch`` library.

Users of other matching libraries (custom code, sklearn NearestNeighbors,
etc.) can write their own wrapper by exposing the same three attributes:
``matched_data``, ``cluster_ids``, and ``rematch(data)``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ._tabular import to_pandas_if_needed


class PysmatchClient:
    """Wrap a fitted ``pysmatch.Matcher`` for use with ``GComputation``.

    This adapter satisfies the ``MatchingClient`` protocol consumed by
    ``GComputation(..., matching=...)``. It extracts the matched subset and
    matched-set labels from a fitted ``pysmatch.Matcher``, and implements
    ``rematch()`` so that bootstrap inference can re-run matching on each
    resampled replicate.

    Parameters
    ----------
    matcher : pysmatch.Matcher
        A **fitted** ``pysmatch.Matcher`` that has already had
        ``fit_scores()``, ``predict_scores()``, and ``match()`` called.
    treatment_col : str
        Name of the binary treatment indicator column in the data. Used to
        reconstruct test/control groups during ``rematch()``.
    fit_scores_kwds : dict, optional
        Keyword arguments passed to ``Matcher.fit_scores()`` on each
        ``rematch()`` call. If omitted, sensible defaults are used
        (``balance=True, model_type='linear', nmodels=3, n_jobs=1``).
    match_kwds : dict, optional
        Keyword arguments passed to ``Matcher.match()`` on each
        ``rematch()`` call. If omitted, sensible defaults are used
        (``method='min', nmatches=1, threshold=0.001, replacement=False``).

    Attributes
    ----------
    matched_data : pandas.DataFrame
        The matched subset (``matcher.matched_data``).
    cluster_ids : numpy.ndarray
        Matched-set labels extracted from ``matched_data["match_id"]``.

    Examples
    --------
    >>> from pysmatch.Matcher import Matcher
    >>> from pymargins.matching import PysmatchClient
    >>> matcher = Matcher(test, control, yvar="treated", exclude=["y"])
    >>> matcher.fit_scores(balance=True, model_type="linear")
    >>> matcher.predict_scores()
    >>> matcher.match(method="min", nmatches=1, threshold=0.001)
    >>> client = PysmatchClient(matcher, treatment_col="treated")
    >>> m = GComputation(fitted_model, matching=client)
    """

    def __init__(
        self,
        matcher: Any,
        treatment_col: str,
        fit_scores_kwds: dict | None = None,
        match_kwds: dict | None = None,
    ):
        try:
            import pysmatch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PysmatchClient requires 'pysmatch'. "
                "Install it with: pip install pysmatch"
            ) from exc

        if not hasattr(matcher, "matched_data"):
            raise ValueError(
                "matcher must be a fitted pysmatch.Matcher with a "
                "'matched_data' attribute. Call fit_scores(), predict_scores(), "
                "and match() before wrapping."
            )

        self._matcher = matcher
        self._treatment_col = treatment_col

        # Store kwds for rematch; use defaults if not provided.
        self._fit_scores_kwds = {
            "balance": True,
            "model_type": "linear",
            "nmodels": 3,
            "n_jobs": 1,
            "max_iter": 100,
            "use_optuna": False,
            "n_trials": 10,
            "balance_strategy": "over",
        }
        if fit_scores_kwds is not None:
            self._fit_scores_kwds.update(fit_scores_kwds)

        self._match_kwds = {
            "method": "min",
            "nmatches": 1,
            "threshold": 0.001,
            "replacement": False,
            "exhaustive_matching": None,
        }
        if match_kwds is not None:
            self._match_kwds.update(match_kwds)

        # Expose protocol attributes
        self.matched_data = matcher.matched_data.copy()
        if "match_id" not in self.matched_data.columns:
            raise ValueError(
                "matcher.matched_data is missing the 'match_id' column. "
                "Ensure match() was called before wrapping."
            )
        self.cluster_ids = self.matched_data["match_id"].values

    def rematch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Re-run pysmatch matching on a bootstrap resample.

        Parameters
        ----------
        data : pandas.DataFrame
            A resample of the original matched data (may contain duplicate
            rows). Must include the ``treatment_col`` column.

        Returns
        -------
        pandas.DataFrame
            The new matched subset from pysmatch. Length may differ from
            ``data`` because some rows may fail to find matches.

        Raises
        ------
        ValueError
            If the resampled data has no samples in one of the treatment
            groups (propagated from ``pysmatch``).
        """
        from pysmatch.Matcher import Matcher

        data = to_pandas_if_needed(data)
        # Reconstruct test / control from the treatment indicator
        mask_treat = data[self._treatment_col] == 1
        test = data[mask_treat].copy()
        control = data[~mask_treat].copy()

        # If one side is empty, pysmatch will raise. Let it propagate so the
        # bootstrap engine can count it as a failed replicate.
        new_matcher = Matcher(
            test=test,
            control=control,
            yvar=self._treatment_col,
            exclude=getattr(self._matcher, "exclude", []),
        )

        # Suppress verbose pysmatch logging during bootstrap rematching
        pysmatch_logger = logging.getLogger("pysmatch")
        old_level = pysmatch_logger.level
        pysmatch_logger.setLevel(logging.WARNING)
        try:
            new_matcher.fit_scores(**self._fit_scores_kwds)
            new_matcher.predict_scores()
            new_matcher.match(**self._match_kwds)
        finally:
            pysmatch_logger.setLevel(old_level)

        return new_matcher.matched_data
