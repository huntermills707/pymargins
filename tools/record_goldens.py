"""Record regression goldens from the new engine.

Usage:
    python tools/record_goldens.py [--force]

Writes one NPZ per anchor-matrix cell to tests/golden/ and a manifest.json.
Overwrite is refused unless --force is passed; --force prints the ledger
reminder.

Design §7.4 / R7.4.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jaxlib
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import steps
from pymargins.estimators import GComputation

REPO_ROOT = Path(__file__).parent.parent
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
PACKAGE_VERSION = "0.4.0"
SEED = 12345


def _env_versions() -> dict:
    """Float-environment fingerprint. Layer-4 goldens are byte-exact, so
    transcendental (expit) and reduction-order bits drift across jaxlib/XLA
    and numpy upgrades. Pinning these makes a future byte mismatch
    diagnosable as environment drift rather than a logic regression."""
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "statsmodels": sm.__version__,
    }


def _make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.3, size=n).astype(float),
            "y_cont": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "treat": rng.binomial(1, 0.5, size=n).astype(float),
            "cluster": np.repeat(np.arange(40), 5),
        }
    )


def _fit_ols(df: pd.DataFrame):
    return smf.ols("y_cont ~ treat + x1 + x2", data=df).fit()


def _fit_glm(df: pd.DataFrame):
    return smf.glm("y ~ treat + x1 + x2", data=df, family=sm.families.Binomial()).fit()


def _save(cell_id: str, r, constructor: str) -> dict:
    payload = {
        "estimate": np.asarray(r.estimate, dtype=np.float64),
        "std_error": np.asarray(r.std_error, dtype=np.float64),
        "conf_int_lower": np.asarray(r.conf_int_lower, dtype=np.float64),
        "conf_int_upper": np.asarray(r.conf_int_upper, dtype=np.float64),
    }
    if hasattr(r, "draws") and r.draws is not None:
        payload["draws"] = np.asarray(r.draws, dtype=np.float64)
    path = GOLDEN_DIR / f"{cell_id}.npz"
    np.savez(path, **payload)
    return {
        "cell_id": cell_id,
        "file": str(path.relative_to(REPO_ROOT)),
        "constructor": constructor,
        "package_version": PACKAGE_VERSION,
        "recorded": datetime.now(timezone.utc).isoformat(),
    }


def _run_predict(est: GComputation, cell_id: str, constructor: str):
    return _save(cell_id, est.predict(), constructor)


def _run_dydx(est: GComputation, cell_id: str, constructor: str):
    return _save(cell_id, est.dydx("x1"), constructor)


def _run_contrasts(est: GComputation, cell_id: str, constructor: str):
    scenarios = [{"atexog": {"treat": 1}}, {"atexog": {"treat": 0}}]
    return _save(
        cell_id,
        est.contrasts(scenarios=scenarios, contrasts=[1, -1]),
        constructor,
    )


def _run_evaluate(est: GComputation, cell_id: str, constructor: str):
    scenarios = [{"atexog": {"treat": 1}}, {"atexog": {"treat": 0}}]
    return _save(
        cell_id,
        est.evaluate(scenarios=scenarios, compose=lambda x: x[1] - x[0]),
        constructor,
    )


def _run_weights(est: GComputation, cell_id: str, constructor: str):
    return _save(cell_id, est.predict(), constructor)


def _run_elasticity(est: GComputation, cell_id: str, constructor: str, query: str):
    return _save(cell_id, getattr(est, query)("x1"), constructor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing golden files (requires a ledger entry).",
    )
    args = parser.parse_args(argv)

    jax.config.update("jax_enable_x64", True)

    existing = list(GOLDEN_DIR.glob("*.npz"))
    if existing and not args.force:
        print(
            "Refusing to overwrite existing goldens:",
            [p.name for p in existing],
            file=sys.stderr,
        )
        print("Pass --force to overwrite (and append a ledger entry).", file=sys.stderr)
        return 1

    if args.force and existing:
        print("WARNING: overwriting regression goldens — append a ledger entry.")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    df = _make_df()
    fit_ols = _fit_ols(df)
    fit_glm = _fit_glm(df)

    manifest_entries = []

    # Anchor matrix: models × methods × queries
    for fit_name, fit in [("ols", fit_ols), ("glm", fit_glm)]:
        for method in ("delta", "simulation"):
            est = GComputation(fit, at="overall", method=method, seed=SEED)
            for query, runner in [("predict", _run_predict), ("dydx", _run_dydx)]:
                cell_id = f"{fit_name}_{method}_{query}"
                manifest_entries.append(
                    runner(
                        est,
                        cell_id,
                        f"GComputation(fit_{fit_name}, at='overall', method='{method}', seed={SEED})",
                    )
                )

    # Bootstrap cells
    for fit_name, fit in [("ols", fit_ols), ("glm", fit_glm)]:
        est = GComputation(fit, at="overall", method="bootstrap", B=200, seed=SEED)
        for query, runner in [("predict", _run_predict), ("dydx", _run_dydx)]:
            cell_id = f"{fit_name}_bootstrap_{query}"
            manifest_entries.append(
                runner(
                    est,
                    cell_id,
                    f"GComputation(fit_{fit_name}, at='overall', method='bootstrap', B=200, seed={SEED})",
                )
            )

    # Cluster-bootstrap cells
    cluster_ids = df["cluster"].values
    for fit_name, fit in [("ols", fit_ols), ("glm", fit_glm)]:
        est = GComputation(
            steps.input(df, cluster=cluster_ids),
            outcome=fit,
            at="overall",
            method="bootstrap",
            B=200,
            seed=SEED,
        )
        for query, runner in [("predict", _run_predict), ("dydx", _run_dydx)]:
            cell_id = f"{fit_name}_cluster_bootstrap_{query}"
            manifest_entries.append(
                runner(
                    est,
                    cell_id,
                    f"GComputation(steps.input(df, cluster=cluster_ids), outcome=fit_{fit_name}, at='overall', method='bootstrap', B=200, seed={SEED})",
                )
            )

    # Contrasts / evaluate / weights
    est_ols_delta = GComputation(fit_ols, at="overall", method="delta", seed=SEED)
    manifest_entries.append(
        _run_contrasts(
            est_ols_delta,
            "ols_delta_contrasts",
            "GComputation(fit_ols, at='overall', method='delta', seed=SEED)",
        )
    )
    manifest_entries.append(
        _run_evaluate(
            est_ols_delta,
            "ols_delta_evaluate",
            "GComputation(fit_ols, at='overall', method='delta', seed=SEED)",
        )
    )

    rng = np.random.default_rng(7)
    w = rng.uniform(0.5, 1.5, size=len(df))
    est_ols_w = GComputation(
        fit_ols, at="overall", method="delta", weights=w, seed=SEED
    )
    manifest_entries.append(
        _run_weights(
            est_ols_w,
            "ols_delta_weights_predict",
            "GComputation(fit_ols, at='overall', method='delta', weights=w, seed=SEED)",
        )
    )

    # Elasticity cells
    for query in ("eyex", "eydx", "dyex"):
        cell_id = f"ols_delta_elasticity_{query}"
        manifest_entries.append(
            _run_elasticity(
                est_ols_delta,
                cell_id,
                "GComputation(fit_ols, at='overall', method='delta', seed=SEED)",
                query,
            )
        )

    manifest = {
        "package_version": PACKAGE_VERSION,
        "recorded": datetime.now(timezone.utc).isoformat(),
        "environment": _env_versions(),
        "cells": manifest_entries,
    }
    manifest_path = GOLDEN_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"Recorded {len(manifest_entries)} golden cells to {GOLDEN_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
