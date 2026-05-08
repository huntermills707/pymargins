"""
Demo: Scales and contrasts — RR, direct ratio, and lift.

Uses the same synthetic NHANES-like data as williams_2012_demo.py.
Demonstrates three ways to quantify the effect of a binary covariate:

  1. Risk Ratio (RR)      — log_scale:  exp(log(p1) - log(p0))
  2. Direct Ratio         — linear_scale + evaluate:  p1 / p0
  3. Lift (RR - 1)        — log_scale result minus 1
  4. pymargins lift_scale — (1+p1)/(1+p0) - 1  (documented for completeness)

Reference implementations:
  - StatsModels: manual point estimates
  - R marginaleffects: delta-method SEs and CIs
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins


def make_data(n=5000, seed=42):
    """Synthetic health-survey data mimicking NHANES II structure."""
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.52, size=n)
    black = rng.binomial(1, 0.11, size=n)
    age = rng.integers(20, 75, size=n)
    agegrp = pd.cut(age, bins=[19, 29, 39, 49, 59, 69, 100], labels=[1, 2, 3, 4, 5, 6]).astype(int)
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
    bp = 110 + 0.4 * age + 2.5 * black + 1.2 * female + 0.5 * bmi + rng.normal(0, 8, size=n)
    return pd.DataFrame({
        "diabetes": diabetes,
        "bp": bp,
        "black": black,
        "female": female,
        "age": age,
        "agegrp": agegrp,
        "bmi": bmi,
    })


def _print_section(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


if __name__ == "__main__":
    df = make_data(n=5000, seed=42)

    fit_logit = smf.glm(
        "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()

    # =====================================================================
    # 1. Risk Ratio via log_scale
    # =====================================================================
    _print_section("1. RISK RATIO — log_scale")
    print("   Inference: log(p1) - log(p0)   |   Reported: exp(...)")
    print()

    m_rr = Margins.log_scale(fit_logit, at="overall", kappa_threshold=float("inf"))
    rr_black = m_rr.contrasts(
        scenarios=[
            {"atexog": {"black": 1}, "label": "black=1"},
            {"atexog": {"black": 0}, "label": "black=0"},
        ],
        contrasts=[+1, -1],
    )
    print(rr_black.summary(stars=True))
    print()

    rr_female = m_rr.contrasts(
        scenarios=[
            {"atexog": {"female": 1}, "label": "female=1"},
            {"atexog": {"female": 0}, "label": "female=0"},
        ],
        contrasts=[+1, -1],
    )
    print(rr_female.summary(stars=True))
    print()

    # =====================================================================
    # 2. Direct Ratio via linear_scale + evaluate()
    # =====================================================================
    _print_section("2. DIRECT RATIO — linear_scale + evaluate(p[0]/p[1])")
    print("   Inference: p1 / p0 directly   |   Delta-method SE on ratio scale")
    print()

    m_ratio = Margins.linear_scale(fit_logit, at="overall", kappa_threshold=float("inf"))
    ratio_black = m_ratio.evaluate(
        scenarios=[
            {"atexog": {"black": 1}, "label": "black=1"},
            {"atexog": {"black": 0}, "label": "black=0"},
        ],
        compose=lambda p: p[0] / p[1],
    )
    print(ratio_black.summary(stars=True))
    print()

    ratio_female = m_ratio.evaluate(
        scenarios=[
            {"atexog": {"female": 1}, "label": "female=1"},
            {"atexog": {"female": 0}, "label": "female=0"},
        ],
        compose=lambda p: p[0] / p[1],
    )
    print(ratio_female.summary(stars=True))
    print()

    # =====================================================================
    # 3. True Lift = RR - 1
    # =====================================================================
    _print_section("3. TRUE LIFT (RR - 1)")
    print("   Computed from log_scale RR:  lift = RR - 1")
    print()

    lift_black_est = float(rr_black.estimate) - 1.0
    lift_black_ci = (float(rr_black.conf_int_lower) - 1.0, float(rr_black.conf_int_upper) - 1.0)
    print(f"black:  lift = {lift_black_est:.4f}   CI = ({lift_black_ci[0]:.4f}, {lift_black_ci[1]:.4f})")

    lift_female_est = float(rr_female.estimate) - 1.0
    lift_female_ci = (float(rr_female.conf_int_lower) - 1.0, float(rr_female.conf_int_upper) - 1.0)
    print(f"female: lift = {lift_female_est:.4f}   CI = ({lift_female_ci[0]:.4f}, {lift_female_ci[1]:.4f})")
    print()

    # =====================================================================
    # 4. pymargins lift_scale (non-standard lift)
    # =====================================================================
    _print_section("4. pymargins lift_scale — (1+p1)/(1+p0) - 1")
    print("   NOTE: This is NOT standard marketing lift (RR - 1).")
    print("   The lift_scale applies log1p to predictions and expm1 to CIs,")
    print("   so the reported contrast is (1+p1)/(1+p0) - 1.")
    print()

    m_lift = Margins.lift_scale(fit_logit, at="overall", kappa_threshold=float("inf"))
    lift_black = m_lift.contrasts(
        scenarios=[
            {"atexog": {"black": 1}, "label": "black=1"},
            {"atexog": {"black": 0}, "label": "black=0"},
        ],
        contrasts=[+1, -1],
    )
    print(lift_black.summary(stars=True))
    print()

    lift_female = m_lift.contrasts(
        scenarios=[
            {"atexog": {"female": 1}, "label": "female=1"},
            {"atexog": {"female": 0}, "label": "female=0"},
        ],
        contrasts=[+1, -1],
    )
    print(lift_female.summary(stars=True))
    print()

    # =====================================================================
    # Compact reference table
    # =====================================================================
    _print_section("REFERENCE TABLE — All Scales")
    print(f"{'Scale':<20} {'black':>12} {'female':>12}")
    print("-" * 46)
    print(f"{'RR (log_scale)':<20} {float(rr_black.estimate):>12.4f} {float(rr_female.estimate):>12.4f}")
    print(f"{'Direct ratio':<20} {float(ratio_black.estimate):>12.4f} {float(ratio_female.estimate):>12.4f}")
    print(f"{'True lift (RR-1)':<20} {lift_black_est:>12.4f} {lift_female_est:>12.4f}")
    print(f"{'lift_scale':<20} {float(lift_black.estimate):>12.4f} {float(lift_female.estimate):>12.4f}")
    print()
    print("Demo complete.")
