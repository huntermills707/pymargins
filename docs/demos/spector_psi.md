---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Spector & Mazzeo (1980) — Did PSI work?

Thirty-two economics students. Three covariates: incoming GPA, a
diagnostic test score (`TUCE`), and a binary indicator `PSI` for
whether the student was taught via Personalized System of
Instruction. Outcome `GRADE` is 1 if the student's grade improved.

The dataset is tiny — small enough that the *uncertainty* dominates
the analysis. That makes it perfect for demonstrating how
`pymargins` lets you cross-check delta-method intervals against a
simulation or bootstrap with one keyword change, and how the κ
diagnostic flags when those methods disagree.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import Margins, pairwise

data = sm.datasets.spector.load_pandas()
df = data.exog.copy()
df["GRADE"] = data.endog.astype(int)
print(df.describe().round(2))
```

## 1. Fit the logit

```{code-cell} python
fit = smf.glm(
    "GRADE ~ GPA + TUCE + PSI",
    data=df,
    family=sm.families.Binomial(),
).fit()
print(fit.summary().tables[1])
```

## 2. The treatment effect three ways

The substantive question is "what does PSI do to the probability of
grade improvement, holding GPA and TUCE fixed?" That's a contrast
between PSI=1 and PSI=0:

```{code-cell} python
scen, w = pairwise("PSI", [1, 0])

m_delta = Margins.linear_scale(fit, vcov="HC1", at="overall")
print("Delta method:")
print(m_delta.contrasts(scenarios=scen, contrasts=w).summary())
```

With only 32 observations the delta-method CI is wide. A Krinsky–Robb
simulation is the cleanest cross-check that does not require refits:

```{code-cell} python
m_sim = Margins.linear_scale(
    fit, vcov="HC1", at="overall",
    method="simulation", n_sim=2000, rng_seed=0,
)
print("Krinsky–Robb simulation:")
print(m_sim.contrasts(scenarios=scen, contrasts=w).summary())
```

A paired bootstrap is the more general non-parametric check, but
with n=32 and three covariates the resampled datasets are frequently
perfectly separable — the logit refit fails to converge and the
bootstrap is not the right tool for this size of problem. The κ
diagnostic gives the analytical reason the delta and simulation
intervals agree:

```{code-cell} python
print(m_delta.diagnose().summary())
```

A small κ confirms the response surface is essentially linear in the
neighborhood being asked about, so the delta method is the cheapest
correct answer. A large κ would point you straight at the simulation
column.

## 3. Heterogeneous treatment effect — does PSI help everyone equally?

Compute the PSI vs no-PSI contrast at three representative GPA
levels. This is `at` doing real work — each row of the result is the
treatment effect at a fixed GPA, with the other covariate (`TUCE`)
held at its sample average:

```{code-cell} python
gpas = [2.5, 3.0, 3.5]
k = len(gpas)
scenarios = [
    {"atexog": {"PSI": 1, "GPA": g}, "label": f"PSI=1, GPA={g}"} for g in gpas
] + [
    {"atexog": {"PSI": 0, "GPA": g}, "label": f"PSI=0, GPA={g}"} for g in gpas
]
W = np.zeros((k, 2 * k))
for i in range(k):
    W[i, i] = +1.0
    W[i, k + i] = -1.0
het = m_delta.contrasts(scenarios=scenarios, contrasts=W)
print(het.summary())
```

If the joint effect is statistically distinguishable from zero but
the individual GPA-stratified effects are not, that's a signal that
the dataset has just enough power for an *average* claim and not
enough for a *subgroup* claim. With n=32 this happens a lot.

## 4. Predicted probability curves

```{code-cell} python
import matplotlib.pyplot as plt

gpa_grid = np.linspace(df["GPA"].min(), df["GPA"].max(), 40)
res = m_delta.predict(atexog={"GPA": gpa_grid, "PSI": [0, 1]})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
for psi, sub in df_plot.groupby("PSI"):
    label = "PSI" if psi == 1 else "Control"
    ax.plot(sub["GPA"], sub["estimate"], label=label)
    ax.fill_between(
        sub["GPA"], sub["ci_lower"], sub["ci_upper"], alpha=0.2
    )
ax.set(xlabel="Entering GPA", ylabel="P(grade improvement)")
ax.legend()
```

The plot makes the small-sample story honest: the PSI curve sits
above the control curve at every GPA, but the CIs overlap heavily,
which the contrast table above already told you in scalar form.

## Where to next

- [](../tutorials/inference_methods.md) — delta vs simulation vs
  bootstrap, side by side, on a single estimand.
- [](../howto/bootstrap.md) — bootstrap recipes.
- [](../explanations/kappa_diagnostic.md) — what κ measures and when
  it triggers a fallback.
