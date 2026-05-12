# pymargins

Expert-mode marginal effects for Python. Session-level analytical
pre-commitment, JAX-native autodiff, κ-driven simulation fallback.

## Status

Alpha. End-to-end usable with statsmodels GLM and OLS/WLS/GLS through
auto-detection. Additional adapters (sklearn, linearmodels, mixed
models), cluster/block bootstrap, and reporting polish remain on the
roadmap.

## Quick example

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pymargins import Margins

# Fit a model
fit = smf.glm(
    "outcome ~ treatment + age + sex",
    data=df,
    family=sm.families.Binomial(),
).fit()

# Wrap in a session, committing to log-scale analysis
m = Margins.log_scale(fit, vcov="HC3", level=0.95)
print(m.summary())  # methods-section paragraph

# Pre-flight diagnostic: is delta reliable here?
print(m.diagnose().summary())

# Compute a relative risk contrast
rr = m.contrasts(
    scenarios=[
        {"atexog": {"treatment": 1}, "label": "treated"},
        {"atexog": {"treatment": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
)
print(rr.summary())  # estimate, asymmetric CI, κ, etc.
```

## Performance notes

- **Bootstrap with `n_jobs > 1`**: Parallel bootstrap uses a
  `ThreadPoolExecutor` for model refitting, but JAX evaluation is
  always serial in the main thread. This avoids XLA compilation race
  conditions. BLAS threads are limited to 1 per worker to prevent
  oversubscription.

- **Large scenario grids**: `expand_scenario` creates one block of rows
  per grid point. For a 10-point grid over a 1M-row dataset this
  materialises 10M rows. Use smaller representative samples (e.g.
  `at="typical"`) or pass explicit `data=` overrides when exploring
  high-dimensional counterfactuals.

- **Memory retention**: `MarginsResult` objects hold references to the
  parent session, design matrices, and gradients. Call
  `result.materialize()` promptly on results you intend to store
  long-term; this drops the heavy machinery while preserving estimates,
  standard errors, and confidence intervals.

## License

MIT
