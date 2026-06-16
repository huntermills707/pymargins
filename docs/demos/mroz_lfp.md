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

# Mroz (1987) — Female labor force participation
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

The Mroz data is the canonical textbook example for a binary-choice
labor model: 753 married women observed in 1975, with a binary
indicator `inlf` for whether they participated in the labor force.
The covariates are family non-wife income, education, experience
(plus its square), age, and counts of young (`kidslt6`) and older
(`kidsge6`) children.

This demo runs a complete `pymargins` analysis end-to-end:

1. Fit a logit.
2. Compare AMEs vs MEMs — the canonical case where they materially
   differ.
3. Discrete-change effect of an additional young child.
4. Predicted participation curve over education, by number of young
   children.
5. A bootstrap confidence interval on the discrete-change effect for
   comparison with the delta-method result.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.datasets import mroz

from pymargins import GComputation, pairwise  # 0.4.0: Margins -> GComputation

cols = ["inlf", "nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
df = mroz.load()[cols].dropna().copy()
df["expersq"] = df["exper"] ** 2
print(df.describe().round(2))
```

## 1. Fit the participation logit

```{code-cell} python
fit = smf.glm(
    "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6",
    data=df,
    family=sm.families.Binomial(),
).fit()
print(fit.summary().tables[1])
```

## 2. AME vs MEM — why the choice matters

The marginal effect of education on P(inlf=1) is *nonlinear* in the
linear predictor, so the average of per-row effects (AME) is not the
same as the effect evaluated at the sample mean profile (MEM). For a
participation rate near 0.5 the two are usually close; for rates near
0 or 1 they can disagree by an order of magnitude.

```{code-cell} python
m_ame = Margins.linear_scale(fit, vcov="HC3", at="overall")
m_mem = Margins.linear_scale(fit, vcov="HC3", at="typical")

print("AME (averaged over sample):")
print(m_ame.dydx("educ").summary())
print("\nMEM (at the typical profile):")
print(m_mem.dydx("educ").summary())
```

The AME for `educ` here is the right number to report in a paper;
the MEM exists for replication of Stata `atmeans` output.

## 3. Discrete change — one additional child under six

`kidslt6` is a count regressor, but for policy interpretation the
quantity of interest is *the effect of having one more young child*,
not the partial derivative. That's a contrast, not a slope:

```{code-cell} python
scen, w = pairwise("kidslt6", [1, 0])
delta = m_ame.contrasts(scenarios=scen, contrasts=w)
print(delta.summary())
```

The estimate is the population-average drop in participation
probability associated with moving from zero to one young child,
holding every other covariate at its observed value. Compare to the
naive `dydx("kidslt6")`:

```{code-cell} python
print(m_ame.dydx("kidslt6").summary())
```

The two answer different questions. The derivative assumes a smooth
treatment intensity that doesn't exist; the contrast is what an
audience usually wants.

## 4. Education profile by family composition

```{code-cell} python
import matplotlib.pyplot as plt

educ_grid = list(range(5, 18))
res = m_ame.predict(atexog={"educ": educ_grid, "kidslt6": [0, 1, 2]})
df_plot = res.to_frame()

fig, ax = plt.subplots(figsize=(6, 4))
for k, sub in df_plot.groupby("kidslt6"):
    ax.plot(sub["educ"], sub["estimate"], label=f"{k} young child(ren)")
    ax.fill_between(
        sub["educ"], sub["ci_lower"], sub["ci_upper"], alpha=0.15
    )
ax.set(xlabel="Years of education",
       ylabel="P(in labor force)",
       title="Predicted participation, averaged over other covariates")
ax.legend()
```

Two things show up immediately on this plot that you cannot read off
the coefficient table: the *slope* of education flattens as kids
under six accumulate, and the gap between the kid-curves is widest
at intermediate education — both implications of the logit link, not
of any new parameter.

## 5. Bootstrap cross-check on the discrete change

The session API lets you swap inference methods without re-stating
the estimand. Compare the delta-method CI from above to a paired
bootstrap on the same contrast:

```{code-cell} python
m_boot = Margins.linear_scale(
    fit, vcov="HC3", at="overall",
    method="bootstrap", n_boot=400, rng_seed=0,
)
print(m_boot.contrasts(scenarios=scen, contrasts=w).summary())
```

If the bootstrap and delta-method intervals agree, you have evidence
the linearization is well-behaved at the estimand surface; if they
disagree materially, the [κ
diagnostic](../explanations/kappa_diagnostic.md) will usually have
flagged the surface as curved already. For a logit at a mid-range
participation probability the two methods agree closely.

## Where to next

- [](fair_affairs.md) — same dataset family (cross-sectional binary
  choice), but with two competing model families (logit and Poisson)
  and a scale comparison.
- [](../tutorials/glm_logit.md) — the underlying logit tutorial.
- [](../howto/discrete_changes.md) — when to use contrasts vs slopes
  for count regressors.