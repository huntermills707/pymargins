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

# Bayesian marginal effects — posterior draws
> **Migration note (0.4.0):** direct posterior-draw inference via
> `Margins.from_posterior` has been removed from the 0.4.x surface. It is
> planned for re-introduction in 0.5.0. This demo is kept as a placeholder.

Every inference path in `pymargins` ultimately needs a distribution
over the parameter vector β. The delta method assumes that
distribution is normal with covariance `vcov`; Krinsky–Robb
*simulates* draws from that same normal; the bootstrap resamples it.

If you fit the model in a Bayesian framework — PyMC, NumPyro,
Stan, `bambi` — you already have the posterior as a bank of MCMC
draws. In 0.4.0 you cannot pass those draws directly; use
`method="bootstrap"` or `method="simulation"` instead.

```python
from pymargins import GComputation

# Delta-method interval
m_delta = GComputation(fit, at="overall", method="delta", scale="identity")
print(m_delta.dydx("educ").summary())

# Bootstrap interval
m_boot = GComputation(fit, at="overall", method="bootstrap", B=1000, scale="identity")
print(m_boot.dydx("educ").summary())
```

When 0.5.0 lands, a posterior-draw path will plug the same draw
matrix directly into the simulation machinery, giving a genuine
outcome-scale credible interval without delta-method linearization.
