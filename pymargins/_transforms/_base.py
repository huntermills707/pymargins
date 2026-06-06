"""Stage protocol and base types for the transform pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Stage(Protocol):
    """A data-transform stage in the bootstrap pipeline.

    Stages apply ``frame → frame`` transforms.  The engine reads the
    declared contract attributes (``requires_resampling``, ``alters_rows``,
    ``emits_columns``, ``source_data``) to decide resampling strategy and
    index handling; it calls ``prepare`` once at point-estimate time and
    ``prepare_resample`` on every bootstrap replicate.
    """

    # --- declared contract (read by the engine, not called) ---
    requires_resampling: bool
    """True ⇒ invalid under delta/simulation; forces bootstrap."""

    alters_rows: bool
    """True ⇒ row set changes ⇒ refit index=None."""

    emits_columns: tuple[str, ...]
    """Aux columns this stage writes (reserved in v1)."""

    source_data: pd.DataFrame | None
    """If not None, the frame the bootstrap resamples instead of
    ``adapter.training_data``."""

    # --- behaviour ---
    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run once, at point-estimate time.

        Fit-and-apply: fit any internal model on *data* and return the
        transformed frame.

        .. note::
            In v1 the engine never calls ``prepare``; the user is expected
            to fit their model on the already-prepared data (matching
            precedent).  Stage authors should not rely on the engine
            invoking this method.
        """

    def prepare_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Re-derive the transform on a bootstrap resample.

        MUST re-fit anything data-dependent from scratch (re-derive, don't
        reuse).  This is the direct analog of ``matching.rematch``.
        """


class IdentityStage:
    """No-op stage for testing and plumbing verification.

    Returns input unchanged; all contract flags are False/empty defaults.
    """

    requires_resampling: bool = False
    alters_rows: bool = False
    emits_columns: tuple[str, ...] = ()
    source_data: pd.DataFrame | None = None

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def prepare_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        return data
