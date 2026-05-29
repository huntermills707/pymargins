"""
Reference implementation using StatsModels only — scales and contrasts.

Replicates the ratio/lift analyses from williams_2012_demo_scales.py
using manual calculations.  These values serve as "ground truth" for
correctness tests of pymargins.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def make_data(n=5000, seed=42):
    """Synthetic health-survey data mimicking NHANES II structure."""
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.52, size=n)
    black = rng.binomial(1, 0.11, size=n)
    age = rng.integers(20, 75, size=n)
    agegrp = pd.cut(
        age, bins=[19, 29, 39, 49, 59, 69, 100], labels=[1, 2, 3, 4, 5, 6]
    ).astype(int)
    bmi = 22 + 0.15 * age + 1.5 * female + rng.normal(0, 4, size=n)
    bmi = np.clip(bmi, 15, 50)
    lp = (
        -4.0
        + 0.55 * black
        + 0.10 * female
        + 0.06 * age
        + 0.03 * bmi
        + 0.5 * (agegrp == 2).astype(float)
        + 0.9 * (agegrp == 3).astype(float)
        + 1.4 * (agegrp == 4).astype(float)
        + 2.0 * (agegrp == 5).astype(float)
        + 2.6 * (agegrp == 6).astype(float)
    )
    diabetes = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    bp = (
        110
        + 0.4 * age
        + 2.5 * black
        + 1.2 * female
        + 0.5 * bmi
        + rng.normal(0, 8, size=n)
    )
    return pd.DataFrame(
        {
            "diabetes": diabetes,
            "bp": bp,
            "black": black,
            "female": female,
            "age": age,
            "agegrp": agegrp,
            "bmi": bmi,
        }
    )


def _print_section(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


def _fmt(val, prec=4):
    return f"{val:.{prec}f}"


if __name__ == "__main__":
    df = make_data(n=5000, seed=42)

    fit_logit = smf.glm(
        "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # ------------------------------------------------------------------
    # 1. Risk Ratio / Direct Ratio (manual)
    # ------------------------------------------------------------------
    _print_section("1. RISK RATIO & DIRECT RATIO (manual)")

    tmp_b1 = df.copy()
    tmp_b1["black"] = 1
    tmp_b0 = df.copy()
    tmp_b0["black"] = 0
    p_black1 = fit_logit.predict(tmp_b1).mean()
    p_black0 = fit_logit.predict(tmp_b0).mean()
    rr_black = p_black1 / p_black0
    print(f"  black:  RR = {_fmt(rr_black)}")

    tmp_f1 = df.copy()
    tmp_f1["female"] = 1
    tmp_f0 = df.copy()
    tmp_f0["female"] = 0
    p_female1 = fit_logit.predict(tmp_f1).mean()
    p_female0 = fit_logit.predict(tmp_f0).mean()
    rr_female = p_female1 / p_female0
    print(f"  female: RR = {_fmt(rr_female)}")
    print()

    # ------------------------------------------------------------------
    # 2. True Lift (RR - 1)
    # ------------------------------------------------------------------
    _print_section("2. TRUE LIFT = RR - 1 (manual)")
    lift_black = rr_black - 1.0
    lift_female = rr_female - 1.0
    print(f"  black:  lift = {_fmt(lift_black)}")
    print(f"  female: lift = {_fmt(lift_female)}")
    print()

    # ------------------------------------------------------------------
    # Compact reference table
    # ------------------------------------------------------------------
    _print_section("REFERENCE TABLE — Point Estimates")
    print(f"{'Scale':<30} {'black':>12} {'female':>12}")
    print("-" * 56)
    print(f"{'RR / Direct ratio':<30} {_fmt(rr_black):>12} {_fmt(rr_female):>12}")
    print(f"{'True lift (RR - 1)':<30} {_fmt(lift_black):>12} {_fmt(lift_female):>12}")
    print()
    print("StatsModels reference complete.")
