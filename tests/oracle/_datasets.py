"""Deterministic oracle datasets.

Design §4.2, req §7. Added in 0.4.0 (R1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_oracle_main(seed: int = 20260611, n: int = 400) -> pd.DataFrame:
    """Return the main oracle frame.

    Columns support OLS, logit, probit, Poisson, and survey-logit cases.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treat = rng.binomial(1, 0.5, size=n).astype(float)
    eta_b = -0.3 + 0.8 * treat + 0.5 * x1 - 0.4 * x2
    y_bin = rng.binomial(1, 1 / (1 + np.exp(-eta_b))).astype(float)
    y_count = rng.poisson(np.exp(0.1 + 0.4 * treat + 0.3 * x1)).astype(float)
    y_cont = 1.0 + 2.0 * treat + 1.5 * x1 - 1.0 * x2 + rng.normal(size=n)
    g = np.repeat(np.arange(25), n // 25)
    strata = np.repeat(np.arange(4), n // 4)
    obs_per_psu = n // 4 // 5
    psu = strata * 100 + np.tile(np.repeat(np.arange(5), obs_per_psu), 4)
    w = np.exp(rng.normal(0.0, 0.3, size=n))
    return pd.DataFrame(
        {
            "y_cont": y_cont,
            "y_bin": y_bin,
            "y_count": y_count,
            "treat": treat,
            "x1": x1,
            "x2": x2,
            "g": g,
            "strata": strata,
            "psu": psu,
            "w": w,
        }
    )


def write_data(path: Path | str | None = None) -> Path:
    """Write the main oracle frame to CSV and return the path."""
    if path is None:
        path = Path(__file__).parent / "data" / "oracle_main.csv"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_oracle_main().to_csv(path, index=False)
    return path


if __name__ == "__main__":
    out = write_data()
    print("wrote", out)
