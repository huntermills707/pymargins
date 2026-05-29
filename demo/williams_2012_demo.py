"""
Demo: Replicating analyses from Richard Williams (2012),
"Using the Margins Command to Estimate and Interpret Adjusted Predictions
and Marginal Effects", Stata Journal 12(2): 308-331.

DOI: 10.1177/1536867X1201200209

This script generates synthetic NHANES-like data and demonstrates the
core analyses from the paper using statsmodels + pymargins.

Analyses covered:
  1. Logit model with factor variables
  2. Adjusted Predictions at the Means (APM)
  3. Average Adjusted Predictions (AAP)
  4. Predictions at Representative Values
  5. Marginal Effects at the Means (MEM) — continuous
  6. Average Marginal Effects (AME) — continuous
  7. Discrete changes for dummy variables
  8. Marginal Effects at Representative Values (MER)
  9. OLS model for comparison
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins

# ---------------------------------------------------------------------------
# 0. Generate synthetic NHANES-like data
# ---------------------------------------------------------------------------


def make_data(n=5000, seed=42):
    """Synthetic health-survey data mimicking NHANES II structure."""
    rng = np.random.default_rng(seed)

    # Demographics
    female = rng.binomial(1, 0.52, size=n)
    black = rng.binomial(1, 0.11, size=n)
    age = rng.integers(20, 75, size=n)

    # Age groups (1=20-29, 2=30-39, ..., 6=70+)
    agegrp = pd.cut(
        age, bins=[19, 29, 39, 49, 59, 69, 100], labels=[1, 2, 3, 4, 5, 6]
    ).astype(int)

    # BMI (correlated with age and sex)
    bmi = 22 + 0.15 * age + 1.5 * female + rng.normal(0, 4, size=n)
    bmi = np.clip(bmi, 15, 50)

    # Diabetes risk (logit link)
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

    # Continuous outcome (e.g., blood pressure) for OLS demo
    bp = (
        110
        + 0.4 * age
        + 2.5 * black
        + 1.2 * female
        + 0.5 * bmi
        + rng.normal(0, 8, size=n)
    )

    df = pd.DataFrame(
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
    return df


if __name__ == "__main__":
    df = make_data(n=5000, seed=42)
    print("=" * 70)
    print(f"Synthetic NHANES-like data (n = {len(df):,})")
    print("=" * 70)
    print(df.describe())
    print()

    # =====================================================================
    # 1. Logit model with factor variables (Williams §2)
    # =====================================================================
    print("=" * 70)
    print("1. LOGIT MODEL: diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age")
    print("=" * 70)

    fit_logit = smf.glm(
        "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(fit_logit.summary())
    print()

    # =====================================================================
    # 2. Adjusted Predictions at the Means (APM)  (Williams §3.1)
    # =====================================================================
    # In Stata: margins C(agegrp), atmeans
    # All covariates held at their typical values (continuous = median,
    # categorical = mode).  For models with factor variables this is the
    # safest pymargins equivalent because patsy requires integer category
    # codes; setting a dummy to its mean proportion (0.107) is not a valid
    # level for patsy even though it is mathematically correct.
    print("=" * 70)
    print("2. ADJUSTED PREDICTIONS AT THE MEANS (APM)")
    print("   Stata equivalent: margins agegrp, atmeans")
    print("=" * 70)

    m_apm = Margins.log_scale(fit_logit, at="typical")
    apm = m_apm.predict(
        atexog={"agegrp": list(range(1, 7))},
    )
    print(apm.summary(stars=True))
    print()

    # =====================================================================
    # 3. Average Adjusted Predictions (AAP)  (Williams §3.2)
    # =====================================================================
    # In Stata: margins C(agegrp)
    # Predictions averaged over the observed distribution of other covariates.
    print("=" * 70)
    print("3. AVERAGE ADJUSTED PREDICTIONS (AAP)")
    print("   Stata equivalent: margins agegrp")
    print("=" * 70)

    m_aap = Margins.log_scale(fit_logit, at="overall")
    aap = m_aap.predict(
        atexog={"agegrp": list(range(1, 7))},
    )
    print(aap.summary(stars=True))
    print()

    # =====================================================================
    # 4. Predictions at Representative Values  (Williams §3.3)
    # =====================================================================
    # In Stata: margins, at(age=(20 50 70)) atmeans
    # Evaluate predictions for specific ages, holding others at means.
    print("=" * 70)
    print("4. PREDICTIONS AT REPRESENTATIVE VALUES")
    print("   Stata equivalent: margins, at(age=(20 50 70)) atmeans")
    print("=" * 70)

    m_repr = Margins.log_scale(fit_logit, at="typical")
    repr_pred = m_repr.predict(
        atexog={"age": [20, 50, 70]},
    )
    print(repr_pred.summary(stars=True))
    print()

    # =====================================================================
    # 5. Marginal Effects at the Means (MEM) — continuous  (Williams §4.1)
    # =====================================================================
    # In Stata: margins, dydx(age) atmeans
    print("=" * 70)
    print("5. MARGINAL EFFECTS AT THE MEANS (MEM) — age")
    print("   Stata equivalent: margins, dydx(age) atmeans")
    print("=" * 70)

    m_mem = Margins.log_scale(fit_logit, at="typical")
    mem_age = m_mem.dydx("age")
    print(mem_age.summary(stars=True))
    print()

    # =====================================================================
    # 6. Average Marginal Effects (AME) — continuous  (Williams §4.2)
    # =====================================================================
    # In Stata: margins, dydx(age)
    print("=" * 70)
    print("6. AVERAGE MARGINAL EFFECTS (AME) — age")
    print("   Stata equivalent: margins, dydx(age)")
    print("=" * 70)

    m_ame = Margins.log_scale(fit_logit, at="overall")
    ame_age = m_ame.dydx("age")
    print(ame_age.summary(stars=True))
    print()

    # =====================================================================
    # 7. Discrete changes for dummy variables  (Williams §4.3)
    # =====================================================================
    # For binary/discrete variables, Stata margins computes the discrete
    # change (difference in predicted probability when flipping the dummy
    # from 0 to 1), not a derivative.
    #
    # In Stata: margins, dydx(black)
    # In pymargins: use contrasts() with two scenarios.
    print("=" * 70)
    print("7. DISCRETE CHANGE (MARGINAL EFFECT) — black")
    print("   Stata equivalent: margins, dydx(black)")
    print("=" * 70)

    m_disc = Margins.log_scale(fit_logit, at="overall")
    disc_black = m_disc.contrasts(
        scenarios=[
            {"atexog": {"black": 1}, "label": "black=1"},
            {"atexog": {"black": 0}, "label": "black=0"},
        ],
        contrasts=[+1, -1],
    )
    print(disc_black.summary(stars=True))
    print()

    # Discrete change at the means
    print("-" * 70)
    print("7b. DISCRETE CHANGE AT THE MEANS — female")
    print("    Stata equivalent: margins, dydx(female) atmeans")
    print("-" * 70)

    m_disc_mem = Margins.log_scale(fit_logit, at="typical")
    disc_female = m_disc_mem.contrasts(
        scenarios=[
            {"atexog": {"female": 1}, "label": "female=1"},
            {"atexog": {"female": 0}, "label": "female=0"},
        ],
        contrasts=[+1, -1],
    )
    print(disc_female.summary(stars=True))
    print()

    # =====================================================================
    # 8. Marginal Effects at Representative Values (MER)  (Williams §4.4)
    # =====================================================================
    # In Stata: margins, dydx(black) at(female=(0 1))
    # Compute the discrete change for black, separately for females and males.
    print("=" * 70)
    print("8. MARGINAL EFFECTS AT REPRESENTATIVE VALUES (MER)")
    print("   Stata equivalent: margins, dydx(black) at(female=(0 1))")
    print("=" * 70)

    m_mer = Margins.log_scale(fit_logit, at="overall")
    mer_female0 = m_mer.contrasts(
        scenarios=[
            {"atexog": {"black": 1, "female": 0}, "label": "black=1, female=0"},
            {"atexog": {"black": 0, "female": 0}, "label": "black=0, female=0"},
        ],
        contrasts=[+1, -1],
    )
    mer_female1 = m_mer.contrasts(
        scenarios=[
            {"atexog": {"black": 1, "female": 1}, "label": "black=1, female=1"},
            {"atexog": {"black": 0, "female": 1}, "label": "black=0, female=1"},
        ],
        contrasts=[+1, -1],
    )
    print("Effect of black for MALES (female=0):")
    print(mer_female0.summary(stars=True))
    print()
    print("Effect of black for FEMALES (female=1):")
    print(mer_female1.summary(stars=True))
    print()

    # =====================================================================
    # 9. OLS model — compare with logit  (Williams §5)
    # =====================================================================
    print("=" * 70)
    print("9. OLS MODEL: bp ~ C(black) + C(female) + age + bmi")
    print("=" * 70)

    fit_ols = smf.ols(
        "bp ~ C(black) + C(female) + age + bmi",
        data=df,
    ).fit()
    print(fit_ols.summary())
    print()

    # MEM for OLS (linear model: MEM = AME = coefficient)
    m_ols = Margins.linear_scale(fit_ols, at="typical")
    mem_ols_age = m_ols.dydx("age")
    print("MEM of age in OLS model (should match coefficient):")
    print(mem_ols_age.summary(stars=True))
    print()

    # AME for OLS (same as MEM in linear models)
    m_ols_ame = Margins.linear_scale(fit_ols, at="overall")
    ame_ols_age = m_ols_ame.dydx("age")
    print("AME of age in OLS model:")
    print(ame_ols_age.summary(stars=True))
    print()

    # =====================================================================
    # 10. LaTeX / HTML export demos
    # =====================================================================
    print("=" * 70)
    print("10. EXPORT EXAMPLES")
    print("=" * 70)

    print("--- LaTeX output for AME of age ---")
    print(ame_age.to_latex(stars=True, caption="AME of Age on Diabetes Risk"))
    print()

    print("--- HTML output for AME of age ---")
    print(ame_age.to_html(stars=True, caption="AME of Age on Diabetes Risk"))
    print()

    print("Demo complete.")
