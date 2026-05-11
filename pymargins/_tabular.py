"""
pymargins._tabular

TabularData protocol and concrete backends. Decouples the scenario engine
(_scenarios.py, _estimands.py, margins.py) from pandas.DataFrame.

Phase 1 (current): PandasTabular — wraps pd.DataFrame, zero user-visible change.
Phase 2 (current): PolarsTabular — wraps polars.DataFrame, faster scenario plumbing.
"""

from __future__ import annotations
from typing import Protocol, Iterable, Any, Union, runtime_checkable
import numpy as np

ArrayLike = Union[np.ndarray, Any]  # Any = jnp.ndarray when jax is available


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TabularData(Protocol):
    """Minimal tabular interface for the pymargins scenario engine."""

    # --- introspection ---
    @property
    def columns(self) -> list[str]: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    def dtypes(self) -> dict[str, type]: ...

    # --- column access ---
    def __getitem__(self, key: str) -> ArrayLike: ...

    def with_column(self, name: str, values: ArrayLike) -> "TabularData": ...

    # --- row slicing ---
    def iloc(self, idx: Any) -> "TabularData": ...

    # --- grouping ---
    def groupby(self, keys: list[str]) -> Iterable[tuple[Any, "TabularData"]]: ...

    # --- combination ---
    @staticmethod
    def concat(tables: list["TabularData"]) -> "TabularData": ...

    # --- conversion ---
    def to_pandas(self) -> "pd.DataFrame": ...

    def to_jax_dict(self) -> dict[str, Any]: ...  # dict[str, jnp.ndarray]


# ---------------------------------------------------------------------------
# Pandas backend (Phase 1)
# ---------------------------------------------------------------------------

class PandasTabular:
    """TabularData backend backed by pandas. Zero user-visible change."""

    def __init__(self, df: "pd.DataFrame"):
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"PandasTabular expects pd.DataFrame, got {type(df).__name__}")
        self._df = df.reset_index(drop=True)

    # --- introspection ---
    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def dtypes(self) -> dict[str, type]:
        return {c: self._df[c].dtype for c in self._df.columns}

    # --- column access ---
    def __getitem__(self, key: str) -> np.ndarray:
        return self._df[key].values

    def with_column(self, name: str, values: ArrayLike) -> "PandasTabular":
        import pandas as pd

        df = self._df.copy()
        df[name] = values
        return PandasTabular(df)

    # --- row slicing ---
    def iloc(self, idx: Any) -> "PandasTabular":
        import pandas as pd

        if hasattr(idx, "dtype") and idx.dtype == bool:
            return PandasTabular(self._df.loc[idx].reset_index(drop=True))
        return PandasTabular(self._df.iloc[idx].reset_index(drop=True))

    # --- grouping ---
    def groupby(self, keys: list[str]) -> Iterable[tuple[Any, "PandasTabular"]]:
        for g, gdf in self._df.groupby(keys, sort=True):
            yield (g, PandasTabular(gdf.reset_index(drop=True)))

    # --- combination ---
    @staticmethod
    def concat(tables: list["PandasTabular"]) -> "PandasTabular":
        import pandas as pd

        dfs = [t._df for t in tables]
        return PandasTabular(pd.concat(dfs, ignore_index=True))

    # --- conversion ---
    def to_pandas(self) -> "pd.DataFrame":
        return self._df.copy()

    def to_jax_dict(self) -> dict[str, Any]:
        import jax.numpy as jnp

        return {c: jnp.array(self._df[c].values) for c in self._df.columns}

    # --- pandas-specific escape hatches ---
    def copy(self) -> "PandasTabular":
        return PandasTabular(self._df.copy())

    def head(self, n: int = 5) -> "PandasTabular":
        return PandasTabular(self._df.head(n).copy())

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"PandasTabular(shape={self.shape}, columns={self.columns})"


# ---------------------------------------------------------------------------
# Polars backend (Phase 2)
# ---------------------------------------------------------------------------

class PolarsTabular:
    """TabularData backend backed by Polars.

    Accelerates scenario plumbing (groupby, concat, with_column) and
    converts to pandas at the adapter boundary for patsy/formulaic
    compatibility.
    """

    def __init__(self, df: "pl.DataFrame"):
        import polars as pl

        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"PolarsTabular expects pl.DataFrame, got {type(df).__name__}")
        self._df = df

    # --- introspection ---
    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def dtypes(self) -> dict[str, type]:
        return {c: self._df[c].dtype for c in self._df.columns}

    # --- column access ---
    def __getitem__(self, key: str) -> np.ndarray:
        return self._df[key].to_numpy()

    def with_column(self, name: str, values: ArrayLike) -> "PolarsTabular":
        import polars as pl

        # Normalize values to a polars Series
        if hasattr(values, "__len__") and not isinstance(values, (str, bytes)):
            # array-like: numpy, jax, list, etc.
            series = pl.Series(name, np.asarray(values))
        else:
            # scalar
            series = pl.lit(values).alias(name)
        return PolarsTabular(self._df.with_columns(series))

    # --- row slicing ---
    def iloc(self, idx: Any) -> "PolarsTabular":
        import polars as pl

        if hasattr(idx, "dtype") and idx.dtype == bool:
            # boolean mask
            mask = pl.Series("mask", np.asarray(idx))
            return PolarsTabular(self._df.filter(mask))
        # integer index (single or list)
        if np.ndim(idx) == 0:
            idx = [idx]
        return PolarsTabular(self._df[np.asarray(idx).tolist()])

    # --- grouping ---
    def groupby(self, keys: list[str]) -> Iterable[tuple[Any, "PolarsTabular"]]:
        for g, gdf in self._df.group_by(keys, maintain_order=True):
            # Polars returns tuple keys even for single key; unwrap for pandas compat
            if len(keys) == 1:
                g = g[0]
            yield (g, PolarsTabular(gdf))

    # --- combination ---
    @staticmethod
    def concat(tables: list["PolarsTabular"]) -> "PolarsTabular":
        import polars as pl

        dfs = [t._df for t in tables]
        return PolarsTabular(pl.concat(dfs))

    # --- conversion ---
    def to_pandas(self) -> "pd.DataFrame":
        # Standard numpy-backed conversion for patsy/formulaic safety.
        # PyArrow-backed (use_pyarrow_extension_array=True) is faster but
        # patsy does not handle Arrow extension dtypes correctly.
        return self._df.to_pandas()

    def to_jax_dict(self) -> dict[str, Any]:
        import jax.numpy as jnp

        return {c: jnp.array(self._df[c].to_numpy()) for c in self._df.columns}

    # --- escape hatches ---
    def copy(self) -> "PolarsTabular":
        return PolarsTabular(self._df.clone())

    def head(self, n: int = 5) -> "PolarsTabular":
        return PolarsTabular(self._df.head(n))

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"PolarsTabular(shape={self.shape}, columns={self.columns})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def as_tabular(data) -> Union[PandasTabular, PolarsTabular]:
    """Convert data to a TabularData backend.

    Supports pd.DataFrame, polars.DataFrame, and existing TabularData wrappers.
    """
    import pandas as pd

    if isinstance(data, (PandasTabular, PolarsTabular)):
        return data
    if isinstance(data, pd.DataFrame):
        return PandasTabular(data)

    # Lazy import polars — it's an optional dependency
    try:
        import polars as pl
        if isinstance(data, pl.DataFrame):
            return PolarsTabular(data)
    except ImportError:
        pass

    raise TypeError(
        f"Cannot convert {type(data).__name__} to TabularData. "
        "Expected pd.DataFrame, polars.DataFrame, or a TabularData wrapper."
    )


def concat_tables(tables: list) -> Union[PandasTabular, PolarsTabular]:
    """Concatenate a list of TabularData tables into one.

    Delegates to the dominant backend. Mixed lists are converted to pandas.
    """
    if not tables:
        raise ValueError("concat_tables requires at least one table")

    # Detect dominant backend
    has_polars = any(isinstance(t, PolarsTabular) for t in tables)
    has_pandas = any(isinstance(t, PandasTabular) for t in tables)

    if has_polars and not has_pandas:
        return PolarsTabular.concat(tables)  # type: ignore[arg-type]
    if has_pandas and not has_polars:
        return PandasTabular.concat(tables)  # type: ignore[arg-type]

    # Mixed: convert all to pandas and concat
    pd_tables = [as_tabular(t.to_pandas()) for t in tables]
    return PandasTabular.concat(pd_tables)  # type: ignore[arg-type]


def to_pandas_if_needed(data):
    """Return a pandas DataFrame, converting from TabularData if necessary."""
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return data
