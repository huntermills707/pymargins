# pymargins

[![PyPI](https://img.shields.io/pypi/v/pymargins.svg)](https://pypi.org/project/pymargins/)
[![Python versions](https://img.shields.io/pypi/pyversions/pymargins.svg)](https://pypi.org/project/pymargins/)
[![Documentation](https://readthedocs.org/projects/pymargins/badge/?version=latest)](https://pymargins.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Expert-mode marginal effects for Python: adjusted predictions, slopes,
contrasts, difference-in-differences, and arbitrary differentiable
estimands from fitted statistical models — with **session-level
analytical pre-commitment**, **JAX-native autodiff**, and a
**κ-driven simulation fallback** when the delta method is unsafe.

The design targets [Stata's `margins`](https://www.stata.com/manuals/rmargins.pdf)
and R's [marginaleffects](https://marginaleffects.com/), with one
substantive difference: a `Margins` object is a *session*. Once
constructed it commits to an inference scale, variance estimator,
confidence level, default evaluation point, and inference method.
Every subsequent computation inherits those commitments — so a
reviewer sees the entire methodological posture in one constructor
call, and posture changes show up as new sessions in the audit trail.

## Why marginal effects?

A fitted nonlinear model answers questions on the wrong scale. A
logistic regression of `diabetes` on `bmi` returns a coefficient in
log-odds: `β = 0.087`. No stakeholder, and few analysts, can act on
"a one-unit BMI increase adds 0.087 to the log-odds." The quantity
people actually need — *how much does the probability of diabetes
move?* — is a marginal effect, and in a nonlinear model it is not any
coefficient. It is a derivative (or a discrete contrast) that depends
on where in covariate space you evaluate it, combines several
coefficients whenever the model has interactions or polynomials, and
carries its own standard error.

pymargins computes that quantity and its uncertainty in the units the
decision is made in:

```python
m = Margins.log_scale(fit, vcov="HC3")

m.dydx("bmi")               # AME: avg change in P(diabetes) per unit BMI
m.predict(atexog={"bmi": [25, 30, 35]})   # adjusted P at representative profiles
m.contrasts(scenarios=[                   # treated-vs-control risk difference
    {"atexog": {"treatment": 1}, "label": "treated"},
    {"atexog": {"treatment": 0}, "label": "control"},
], contrasts=[+1, -1])
```

Reach for this when:

- **The model is nonlinear** — GLM, logit/probit, Poisson, Cox, AFT:
  coefficients are not effects on the outcome scale.
- **Effects are conditional** — interactions, polynomials, or splines
  mean the effect of *X* is a combination of terms, not one of them.
- **The audience needs outcome units** — percentage points, expected
  counts, predicted survival — not log-odds or hazard ratios.
- **Heterogeneity is the question** — the effect by age, by region,
  by treatment arm, evaluated over a grid.
- **You need a counterfactual contrast** — "what if everyone were
  treated?" as an average contrast, not a coefficient.

These map onto the three estimands: `predict` (adjusted predictions),
`dydx` (slopes), and `contrasts` / `evaluate` (differences and
arbitrary differentiable combinations). Picking *whose* effect — the
sample average (AME), a typical unit (MEM), or a representative
profile (MER) — is the aggregation axis described in the docs.

## Installation

```bash
pip install pymargins
```

Requires Python ≥3.10. Core dependencies (`jax`, `numpy`, `scipy`,
`pandas`) install automatically. Modeling backends are detected at
runtime and pull in only what you have; install the extras you need:

```bash
pip install "pymargins[statsmodels]"     # statsmodels GLM/OLS/GEE/...
pip install "pymargins[linearmodels]"    # IV, panel, absorbing, Fama–MacBeth
pip install "pymargins[lifelines]"       # Cox, AFT, spline survival models
pip install "pymargins[sklearn]"         # scikit-learn estimators (bootstrap)
pip install "pymargins[polars]"          # Polars input frames
pip install "pymargins[matching]"        # pysmatch propensity matching
```

## Quick example

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pymargins import Margins

fit = smf.glm(
    "outcome ~ treatment + age + sex",
    data=df,
    family=sm.families.Binomial(),
).fit()

# Open a session, committing to log-scale analysis with HC3 SEs
m = Margins.log_scale(fit, vcov="HC3", level=0.95)
print(m.summary())          # methods-section paragraph

# Pre-flight: is the delta method reliable here?
print(m.diagnose().summary())

# Relative risk: a contrast of two counterfactual scenarios
rr = m.contrasts(
    scenarios=[
        {"atexog": {"treatment": 1}, "label": "treated"},
        {"atexog": {"treatment": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
)
print(rr.summary())         # estimate, asymmetric CI, κ, etc.
```

Session constructors declare the inference scale up front:
`Margins.linear_scale`, `Margins.log_scale`, `Margins.logit_scale`,
`Margins.correlation_scale`, or `Margins.from_formula` to fit and wrap
in one call.

## Supported model backends

Adapters are auto-detected from the fitted object. Currently shipping:

- **statsmodels** — OLS/WLS/GLS, GLM, discrete (Logit/Probit/Poisson/
  NegBin/zero-inflated), MNLogit, ordered, GEE (nominal/ordinal),
  MixedLM, RLM, QuantReg, PHReg
- **linearmodels** — IV/2SLS, panel (fixed/random effects), absorbing
  regression, Fama–MacBeth
- **lifelines** — CoxPH, time-varying Cox, Weibull/LogNormal/LogLogistic
  AFT, generalized gamma, piecewise exponential, cubic-spline (CRC),
  with survival-curve estimands
- **scikit-learn** — any estimator, via bootstrap inference
- **custom** — register your own with `register_adapter`

## Inference methods

- **Delta method** — JAX-native exact gradients and Hessians
- **Krinsky–Robb simulation** — parametric Monte Carlo on the
  coefficient distribution
- **Bootstrap** — nonparametric, plus cluster and block variants;
  parallelizable, with percentile/BCa/normal CIs

A **κ (kappa) diagnostic** quantifies local nonlinearity and
automatically recommends — or falls back to — simulation when the
delta-method linearization is untrustworthy.

## Documentation

Full documentation, tutorials, and theory at
**[pymargins.readthedocs.io](https://pymargins.readthedocs.io)** —
including per-backend tutorials (logit, Poisson, OLS, MNLogit, Cox,
AFT, IV/2SLS, panel FE, GEE, mixed effects, sklearn), task-focused
how-to guides (robust/clustered SEs, bootstrap variants, simultaneous
CIs, DiD, elasticities, custom adapters, matching, plotting,
exporting), and design explanations.

## Performance notes

- **Bootstrap with `n_jobs > 1`**: parallel bootstrap uses a
  `ThreadPoolExecutor` for model refitting, but JAX evaluation is
  always serial in the main thread to avoid XLA compilation races.
  BLAS threads are pinned to 1 per worker to prevent oversubscription.

- **Large scenario grids**: `expand_scenario` materializes one block of
  rows per grid point. A 10-point grid over a 1M-row dataset is 10M
  rows. Use representative samples (`at="typical"`) or explicit `data=`
  overrides when exploring high-dimensional counterfactuals.

- **Memory retention**: `MarginsResult` objects hold references to the
  parent session, design matrices, and gradients. Call
  `result.materialize()` promptly on results you intend to keep; it
  drops the heavy machinery while preserving estimates, standard
  errors, and confidence intervals.

## Status

Alpha. APIs may change before 1.0. Bug reports and feedback welcome at
the [issue tracker](https://github.com/huntermills707/pymargins/issues).

## License

[MIT](LICENSE)
