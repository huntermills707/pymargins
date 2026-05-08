"""
Reference implementation using StatsModels only.

Replicates the analyses from williams_2012_demo.py using native
statsmodels functionality and manual calculations.  These values serve as
"ground truth" for correctness tests of pymargins.

Notes on scale
--------------
- The original pymargins demo uses Margins.log_scale(), which reports
  *contrasts* as risk ratios (p1/p0) with asymmetric CIs.
- StatsModels get_margeff() always computes contrasts as probability
  differences (p1 - p0) with symmetric CIs.
- Therefore this script reports BOTH the probability difference (from
  get_margeff) and the risk ratio (computed manually) for discrete-change
  estimands so that tests can verify whichever quantity pymargins is
  configured to return.
"""

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


def _print_section(title, subtitle=""):
    print("=" * 70)
    print(title)
    if subtitle:
        print("   " + subtitle)
    print("=" * 70)


def _fmt(val, prec=4):
    return f"{val:.{prec}f}"


if __name__ == "__main__":
    df = make_data(n=5000, seed=42)

    # =====================================================================
    # 1. Logit model
    # =====================================================================
    _print_section("1. LOGIT MODEL")
    fit_logit = smf.glm(
        "diabetes ~ C(black) + C(female) + C(agegrp) + bmi + age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(fit_logit.summary())
    print()

    median_vals = df.median(numeric_only=True)
    mode_vals = df.mode().iloc[0]

    # Helper: build a 1-row DataFrame at typical (median) values
    def typical_row(overrides=None):
        overrides = overrides or {}
        return pd.DataFrame({
            "black": [overrides.get("black", median_vals["black"])],
            "female": [overrides.get("female", median_vals["female"])],
            "agegrp": [overrides.get("agegrp", median_vals["agegrp"])],
            "bmi": [overrides.get("bmi", median_vals["bmi"])],
            "age": [overrides.get("age", median_vals["age"])],
        })

    # =====================================================================
    # 2. Adjusted Predictions at the Means (APM)
    # =====================================================================
    _print_section(
        "2. ADJUSTED PREDICTIONS AT THE MEANS (APM)",
        "Stata: margins agegrp, atmeans  (pymargins uses median = 'typical')"
    )
    apm = []
    for g in range(1, 7):
        pred = fit_logit.predict(typical_row({"agegrp": g}))[0]
        apm.append({"agegrp": g, "pred": pred})
    for row in apm:
        print(f"  agegrp={row['agegrp']}: {_fmt(row['pred'])}")
    print()

    # =====================================================================
    # 3. Average Adjusted Predictions (AAP)
    # =====================================================================
    _print_section(
        "3. AVERAGE ADJUSTED PREDICTIONS (AAP)",
        "Stata: margins agegrp"
    )
    aap = []
    for g in range(1, 7):
        tmp = df.copy()
        tmp["agegrp"] = g
        pred = fit_logit.predict(tmp).mean()
        aap.append({"agegrp": g, "pred": pred})
    for row in aap:
        print(f"  agegrp={row['agegrp']}: {_fmt(row['pred'])}")
    print()

    # =====================================================================
    # 4. Predictions at Representative Values
    # =====================================================================
    _print_section(
        "4. PREDICTIONS AT REPRESENTATIVE VALUES",
        "Stata: margins, at(age=(20 50 70)) atmeans"
    )
    repr_preds = []
    for a in [20, 50, 70]:
        pred = fit_logit.predict(typical_row({"age": a}))[0]
        repr_preds.append({"age": a, "pred": pred})
    for row in repr_preds:
        print(f"  age={row['age']}: {_fmt(row['pred'])}")
    print()

    # =====================================================================
    # 5. Marginal Effects at the Means (MEM) — continuous
    # =====================================================================
    _print_section(
        "5. MARGINAL EFFECTS AT THE MEANS (MEM) — age",
        "Stata: margins, dydx(age) atmeans"
    )
    mem = fit_logit.get_margeff(at="mean")
    mem_summary = mem.summary_frame()
    print(mem_summary.loc["age"])
    print()

    # =====================================================================
    # 6. Average Marginal Effects (AME) — continuous
    # =====================================================================
    _print_section(
        "6. AVERAGE MARGINAL EFFECTS (AME) — age",
        "Stata: margins, dydx(age)"
    )
    ame = fit_logit.get_margeff(at="overall")
    ame_summary = ame.summary_frame()
    print(ame_summary.loc["age"])
    print()

    # =====================================================================
    # 7. Discrete changes for dummy variables
    # =====================================================================
    _print_section(
        "7. DISCRETE CHANGE (MARGINAL EFFECT) — black",
        "Stata: margins, dydx(black)"
    )
    # get_margeff with dummy=True gives the probability difference
    disc = fit_logit.get_margeff(at="overall", dummy=True)
    disc_summary = disc.summary_frame()
    print("Probability difference (get_margeff dummy=True):")
    print(disc_summary.loc["C(black)[T.1]"])

    # Manual risk ratio for log-scale comparison
    tmp_black1 = df.copy()
    tmp_black1["black"] = 1
    tmp_black0 = df.copy()
    tmp_black0["black"] = 0
    p_black1 = fit_logit.predict(tmp_black1).mean()
    p_black0 = fit_logit.predict(tmp_black0).mean()
    rr_black = p_black1 / p_black0
    print(f"\nRisk ratio (manual): {_fmt(rr_black)}")
    print()

    # ------------------------------------------------------------------
    print("-" * 70)
    print("7b. DISCRETE CHANGE AT THE MEANS — female")
    print("    Stata: margins, dydx(female) atmeans")
    print("-" * 70)
    disc_mem = fit_logit.get_margeff(at="mean", dummy=True)
    disc_mem_summary = disc_mem.summary_frame()
    print("Probability difference:")
    print(disc_mem_summary.loc["C(female)[T.1]"])

    # Manual risk ratio at means
    row_f1 = typical_row({"female": 1})
    row_f0 = typical_row({"female": 0})
    p_f1 = fit_logit.predict(row_f1)[0]
    p_f0 = fit_logit.predict(row_f0)[0]
    rr_female = p_f1 / p_f0
    print(f"\nRisk ratio (manual): {_fmt(rr_female)}")
    print()

    # =====================================================================
    # 8. Marginal Effects at Representative Values (MER)
    # =====================================================================
    _print_section(
        "8. MARGINAL EFFECTS AT REPRESENTATIVE VALUES (MER)",
        "Stata: margins, dydx(black) at(female=(0 1))"
    )

    # Effect of black for females = 0 (males)
    tmp_m = df.copy()
    tmp_m["female"] = 0
    tmp_m_b1 = tmp_m.copy()
    tmp_m_b1["black"] = 1
    tmp_m_b0 = tmp_m.copy()
    tmp_m_b0["black"] = 0
    p_m_b1 = fit_logit.predict(tmp_m_b1).mean()
    p_m_b0 = fit_logit.predict(tmp_m_b0).mean()
    rd_m = p_m_b1 - p_m_b0
    rr_m = p_m_b1 / p_m_b0
    print("MALES (female=0):")
    print(f"  Prob diff: {_fmt(rd_m)}")
    print(f"  Risk ratio: {_fmt(rr_m)}")
    print()

    # Effect of black for females = 1
    tmp_f = df.copy()
    tmp_f["female"] = 1
    tmp_f_b1 = tmp_f.copy()
    tmp_f_b1["black"] = 1
    tmp_f_b0 = tmp_f.copy()
    tmp_f_b0["black"] = 0
    p_f_b1 = fit_logit.predict(tmp_f_b1).mean()
    p_f_b0 = fit_logit.predict(tmp_f_b0).mean()
    rd_f = p_f_b1 - p_f_b0
    rr_f = p_f_b1 / p_f_b0
    print("FEMALES (female=1):")
    print(f"  Prob diff: {_fmt(rd_f)}")
    print(f"  Risk ratio: {_fmt(rr_f)}")
    print()

    # =====================================================================
    # 9. OLS model
    # =====================================================================
    _print_section("9. OLS MODEL")
    fit_ols = smf.ols(
        "bp ~ C(black) + C(female) + age + bmi",
        data=df,
    ).fit()
    print(fit_ols.summary())
    print()

    print("MEM / AME of age in OLS model:")
    print(f"  Coefficient (identical to MEM and AME): {_fmt(fit_ols.params['age'])}")
    print()

    # =====================================================================
    # Compact reference table
    # =====================================================================
    _print_section("REFERENCE TABLE — Point Estimates")
    print(f"{'Estimand':<40} {'Value':>12}")
    print("-" * 54)
    for row in apm:
        print(f"APM agegrp={row['agegrp']:<33} {_fmt(row['pred']):>12}")
    for row in aap:
        print(f"AAP agegrp={row['agegrp']:<33} {_fmt(row['pred']):>12}")
    for row in repr_preds:
        print(f"Repr age={row['age']:<35} {_fmt(row['pred']):>12}")
    print(f"MEM age{'':<39} {_fmt(mem_summary.loc['age', 'dy/dx']):>12}")
    print(f"AME age{'':<39} {_fmt(ame_summary.loc['age', 'dy/dx']):>12}")
    print(f"Discrete black (prob diff){'':<24} {_fmt(disc_summary.loc['C(black)[T.1]', 'dy/dx']):>12}")
    print(f"Discrete black (risk ratio){'':<23} {_fmt(rr_black):>12}")
    print(f"Discrete female atmeans (prob diff){'':<15} {_fmt(disc_mem_summary.loc['C(female)[T.1]', 'dy/dx']):>12}")
    print(f"Discrete female atmeans (risk ratio){'':<14} {_fmt(rr_female):>12}")
    print(f"MER black | female=0 (prob diff){'':<18} {_fmt(rd_m):>12}")
    print(f"MER black | female=0 (risk ratio){'':<17} {_fmt(rr_m):>12}")
    print(f"MER black | female=1 (prob diff){'':<18} {_fmt(rd_f):>12}")
    print(f"MER black | female=1 (risk ratio){'':<17} {_fmt(rr_f):>12}")
    print(f"OLS age coefficient{'':<31} {_fmt(fit_ols.params['age']):>12}")
    print()
    print("StatsModels reference complete.")
