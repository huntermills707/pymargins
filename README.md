# pymargins

Expert-mode marginal effects for Python. Session-level analytical
pre-commitment, JAX-native autodiff, κ-driven simulation fallback.

## Status

Pre-alpha. The library is currently a complete API specification with
implementation stubs and a working numerical core. Not yet usable
end-to-end without the implementation work outlined in
`IMPLEMENTATION_GUIDE.md`.

## Quick orientation

For implementers and contributors:

1. **`PRIMER.md`** — architectural philosophy and design rationale.
   Read this first.
2. **`IMPLEMENTATION_GUIDE.md`** — what's done, what's stubbed, what
   needs filling in, in priority order.
3. **`pymargins/`** — the package itself. Read in the order listed at
   the end of `PRIMER.md`.

## Quick example (once implemented)

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

## License

TBD.
