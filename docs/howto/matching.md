# Matching support

`pymargins.PysmatchClient` integrates with the
[`pysmatch`](https://pypi.org/project/pysmatch/) propensity-score
matcher. The client wraps a fitted matcher, exposes the matched
sample, and offers a `rematch` method for sensitivity analysis.

## Basic workflow

1. Fit a propensity-score model and run matching outside `pymargins`.
2. Wrap the matcher in `PysmatchClient`.
3. Call `rematch(df)` to pull the matched sample deterministically.
4. Fit the outcome model on the matched sample.
5. Open a `Margins` session on the matched outcome fit.

```python
import pandas as pd
import statsmodels.formula.api as smf
from pymargins import Margins, PysmatchClient

# matcher is a fitted pysmatch.Match object from your upstream pipeline
client = PysmatchClient(matcher)
matched = client.rematch(df)             # deterministic re-pull

fit = smf.glm("y ~ treatment + x1 + x2", data=matched,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, vcov="HC3", at="overall")
m.contrasts(
    scenarios=[
        {"atexog": {"treatment": 1}, "label": "matched-treated"},
        {"atexog": {"treatment": 0}, "label": "matched-control"},
    ],
    contrasts=[+1, -1],
).summary()
```

## Why `rematch`?

Matching algorithms are stochastic (random starts, caliper tie-breaking).
`rematch` fixes the random seed internally so that every call returns
the *same* matched sample given the same input data.  This makes the
analysis reproducible and reviewable: the reviewer can re-run the
script and get identical matched rows.

## Sensitivity analysis

`rematch` accepts a `caliper_multiplier` argument.  By varying the
caliper you can assess how sensitive the marginal effect is to the
strictness of the matching:

```python
for mult in [0.5, 1.0, 2.0]:
    matched = client.rematch(df, caliper_multiplier=mult)
    fit = smf.glm("y ~ treatment + x1 + x2", data=matched,
                  family=sm.families.Binomial()).fit()
    m = Margins.log_scale(fit, vcov="HC3", at="overall")
    res = m.contrasts(
        scenarios=[
            {"atexog": {"treatment": 1}},
            {"atexog": {"treatment": 0}},
        ],
        contrasts=[+1, -1],
    )
    print(f"caliper={mult}: RR = {res.estimate.item():.3f}")
```

## Inference after matching

Matching changes the effective sample size and induces correlation
between matched pairs.  Two common strategies:

1. **Analytic cluster-robust SEs** — treat the matched pair as a
   cluster and pass `vcov={"type": "cluster", "groups": pair_id}`.
2. **Cluster bootstrap** — resample matched pairs; see
   [](cluster_block_bootstrap.md).

Both are valid; the bootstrap is more robust to small numbers of
clusters or misspecified propensity models.
