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

# Bayesian marginal effects — posterior draws via `from_posterior`
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

Every inference path in `pymargins` ultimately needs a distribution
over the parameter vector β. The delta method assumes that
distribution is normal with covariance `vcov`; Krinsky–Robb
*simulates* draws from that same normal; the bootstrap resamples it.
But if you fit the model in a Bayesian framework — PyMC, NumPyro,
Stan, `bambi` — you already have the posterior as a bank of MCMC
draws, and there is no reason to throw it away and re-approximate it
as normal.

`Margins.from_posterior` takes those draws directly. It runs the
*simulation* inference path, but instead of drawing β from
`N(β̂, V̂)` it uses your draws as-is. The marginal effect is computed
once per draw, and the spread of those values **is** the posterior of
the marginal effect — a genuine credible interval on the outcome
scale, with no delta-method linearization anywhere.

This demo uses the Mroz female-labor-force data and a logit. To keep
the documentation build dependency-free, we generate the posterior
draws with a **Laplace approximation** — a multivariate normal centred
at the MLE with the model's estimated covariance. Real MCMC draws
from PyMC or NumPyro are an array of exactly the same shape
`(n_draws, n_params)` and plug into the same call; the only thing that
changes is where the draws come from. See the closing note for the
drop-in PyMC/NumPyro snippet.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.datasets import mroz

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

cols = ["inlf", "nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]
df = mroz.load()[cols].dropna().copy()
df["expersq"] = df["exper"] ** 2

fit = smf.glm(
    "inlf ~ nwifeinc + educ + exper + expersq + age + kidslt6 + kidsge6",
    data=df,
    family=sm.families.Binomial(),
).fit()
print(fit.summary().tables[1])
```

## 1. Build the posterior draw bank

The Laplace posterior is `N(β̂, V̂)`. We draw 4 000 samples; the
columns are in the same order as `fit.params`, which is exactly the
order the adapter expects.

```{code-cell} python
beta_hat = np.asarray(fit.params)
V_hat = np.asarray(fit.cov_params())

rng = np.random.default_rng(0)
posterior_draws = rng.multivariate_normal(beta_hat, V_hat, size=4000)
print("draw bank shape:", posterior_draws.shape)  # (n_draws, n_params)
```

## 2. Open a posterior session

`from_posterior` defaults `method="simulation"` and `n_sim` to the
number of draws. The point estimate defaults to the posterior mean;
pass `point_estimate=` to report a different summary (e.g. the MLE or
the posterior median).

```{code-cell} python
m_bayes = Margins.from_posterior(fit, posterior_draws, at="overall")
print("method:", m_bayes.method, " draws:", m_bayes.n_sim)
```

## 3. AME credible interval vs delta-method interval

Ask the posterior session for the average marginal effect of
education on participation. The interval it returns is a **95 %
credible interval** — the 2.5th and 97.5th percentiles of the AME
across the posterior draws. We put it side by side with the ordinary
delta-method confidence interval from a standard session.

```{code-cell} python
m_delta = Margins.linear_scale(fit, at="overall")

ame_bayes = m_bayes.dydx("educ")
ame_delta = m_delta.dydx("educ")

print("Posterior (credible interval):")
print(ame_bayes.summary())
print("\nDelta method (confidence interval):")
print(ame_delta.summary())
```

Under a flat prior the Laplace posterior is, by construction, the same
normal the delta method assumes — so the two intervals here agree to
within Monte-Carlo error. That agreement is the point: it shows the
posterior path is calibrated against the analytic one in the case
where they *should* coincide. The value of `from_posterior` shows up
when they should **not** coincide — an informative or skewed prior, a
small sample where the posterior is non-normal, or a nonlinear
estimand whose posterior is asymmetric. There the credible interval
follows the true posterior shape and the delta interval cannot.

## 4. The full posterior of an effect, not just an interval

Because every draw produces a complete marginal effect, you are not
limited to a symmetric interval — you can report any posterior
summary. Here is the AME of an additional young child (a discrete
contrast) as a posterior, with its probability of being negative.

```{code-cell} python
from pymargins import pairwise

scen, w = pairwise("kidslt6", [1, 0])
child_effect = m_bayes.contrasts(scenarios=scen, contrasts=w)
print(child_effect.summary())

draws = np.asarray(child_effect.draws)
print(f"\nPosterior mean : {draws.mean():.4f}")
print(f"P(effect < 0)  : {(draws < 0).mean():.3f}")
print(f"90% credible   : [{np.percentile(draws, 5):.4f}, "
      f"{np.percentile(draws, 95):.4f}]")
```

`P(effect < 0)` is a direct posterior probability — the kind of
statement a Bayesian analysis is supposed to deliver and that a
p-value cannot. It comes for free once the estimand is evaluated on
each draw.

## Plugging in real MCMC draws

Nothing above is specific to the Laplace approximation. If you fit the
same logit in PyMC or NumPyro, you obtain a posterior array of shape
`(n_draws, n_params)` and pass it to the *same* constructor:

```python
# PyMC
import pymc as pm
with pm.Model() as model:
    # ... priors and likelihood for the same design matrix ...
    idata = pm.sample()
# stack chains × draws into (n_draws, n_params), columns in fit.params order
posterior_draws = idata.posterior[coef_names].to_array() \
    .transpose("chain", "draw", "variable") \
    .values.reshape(-1, len(coef_names))

m_bayes = Margins.from_posterior(fit, posterior_draws, at="overall")
m_bayes.dydx("educ")   # AME with a real posterior credible interval
```

The only requirements are that the draw columns line up with the
adapter's coefficient order (the order of `fit.params`) and that
`fit` is the same specification you sampled. Everything downstream —
`predict`, `dydx`, `contrasts`, `evaluate`, subgroups, plotting —
behaves identically to any other session.

## Where to next

- [](../explanations/delta_sim_bootstrap.md) — how the simulation
  path relates to the delta method and the bootstrap, and why
  `from_posterior` slots into the same machinery.
- [](mroz_lfp.md) — the same dataset analysed with the standard
  delta-method and bootstrap paths.
- [](../tutorials/inference_methods.md) — switching inference methods
  on a single session.