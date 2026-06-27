# Matching support
`pymargins` integrates with propensity-score and other matching libraries
through a small protocol.  The matching object is attached via
`steps.match(...)` and governs all downstream behavior: which rows enter
the estimand, how `vcov` is derived, and how bootstrap resamples are
drawn.

The reference implementation wraps
[`pysmatch`](https://pypi.org/project/pysmatch/) (the active successor
to `pymatch`).  Users of other libraries can write a thin wrapper that
exposes the same three attributes/methods.

## The `MatchingClient` protocol

A matching object consumed by `steps.match(node, matcher)` must provide:

| Attribute / method | Description |
|--------------------|-------------|
| `matched_data` | `pd.DataFrame` — the matched subset the model was fit on |
| `cluster_ids` | `np.ndarray` — matched-set label for every row in `matched_data` |
| `rematch(data)` | callable that re-runs matching on a resampled DataFrame and returns the new matched subset |

`rematch()` is **only required for `method="bootstrap"`**.  Delta and
Krinsky–Robb ignore it.

## Basic workflow with `pysmatch`

1. Fit a propensity-score model and run matching outside `pymargins`.
2. Fit the **outcome model on the matched sample only**.
3. Wrap the fitted matcher in `PysmatchClient`.
4. Build a `GComputation` estimator with `steps.match(steps.input(df), client)`.

```python
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation, PysmatchClient, steps  # 0.4.0: Margins -> GComputation
from pysmatch.Matcher import Matcher

# 1. Run matching
test = df[df["treated"] == 1].copy()
control = df[df["treated"] == 0].copy()
matcher = Matcher(test, control, yvar="treated", exclude=["y"])
matcher.fit_scores(balance=True, model_type="linear")
matcher.predict_scores()
matcher.match(method="min", nmatches=1, threshold=0.001)

# 2. Fit outcome model on matched sample ONLY
matched = matcher.matched_data
fit = smf.glm(
    "y ~ x1 + x2 + treated",
    data=matched,
    family=sm.families.Binomial(),
).fit(cov_type="cluster", cov_kwds={"groups": matched["match_id"]})

# 3. Wrap the matcher
client = PysmatchClient(matcher, treatment_col="treated")

# 4. Build estimator — vcov auto-derives to cluster-robust
m = GComputation(steps.match(steps.input(matched), client), outcome=fit)
rd = m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}, "label": "treated"},
        {"atexog": {"treated": 0}, "label": "control"},
    ],
    contrasts=[+1, -1],
)
print(rd.summary())
```

:::{important}
The outcome model **must** be fit on the matched sample, not the full
sample.  `pymargins` validates that `len(matched_data) ==
len(training_data)` and raises an error if they differ.
:::

## Why `rematch`?

Matching algorithms are stochastic (random starts, caliper tie-breaking).
`rematch` fixes the pipeline internally so that every call returns the
*same* matched sample given the same input data.  More importantly, for
bootstrap inference `rematch` is called on every resampled replicate,
capturing both the uncertainty from the outcome model **and** the
uncertainty introduced by the matching process itself.

## Sensitivity analysis

`PysmatchClient.rematch()` accepts a `caliper_multiplier` argument (when
called directly).  By varying the caliper you can assess how sensitive
the marginal effect is to the strictness of the matching:

```python
for mult in [0.5, 1.0, 2.0]:
    matched = client.rematch(df, caliper_multiplier=mult)
    fit = smf.glm("y ~ x1 + x2 + treated", data=matched,
                  family=sm.families.Binomial()).fit()
    m = GComputation(fit, vcov="HC3", at="overall")
    res = m.contrasts(
        scenarios=[
            {"atexog": {"treated": 1}},
            {"atexog": {"treated": 0}},
        ],
        contrasts=[+1, -1],
    )
    print(f"caliper={mult}: RD = {res.estimate.item():.3f}")
```

## Inference after matching

Matching changes the effective sample size and induces correlation
between matched pairs.  `pymargins` handles this automatically:

| Inference | What happens |
|-----------|--------------|
| **Delta / K–R** (default) | The estimand averages over `matched_data` instead of the full sample. `vcov` auto-derives to cluster-robust at the matched-set level. |
| **Bootstrap** | Resamples **matched sets** with replacement, calls `rematch()` on each replicate, then refits the model. This is more robust to small numbers of clusters or misspecified propensity models. |

### Bootstrap with rematching

```python
m_boot = GComputation(
    steps.match(steps.input(matched), client),
    outcome=fit,
    method="bootstrap",
    B=500,
    seed=42,
)
rd_boot = m_boot.contrasts(
    scenarios=[
        {"atexog": {"treated": 1}},
        {"atexog": {"treated": 0}},
    ],
    contrasts=[+1, -1],
)
print(rd_boot.summary())
```

:::{note}
Bootstrap SEs should generally be **wider** than delta SEs because the
bootstrap captures additional uncertainty from the matching algorithm.
If they are narrower, inspect the propensity model for near-perfect
prediction or very small calipers.
:::

## Custom matching wrappers

Users of `sklearn.neighbors.NearestNeighbors`, custom propensity-score
code, or other libraries can write a wrapper that exposes the same
protocol:

```python
class MyMatcher:
    def __init__(self, matched_data, cluster_ids, matcher_fn):
        self.matched_data = matched_data
        self.cluster_ids = cluster_ids
        self._matcher_fn = matcher_fn

    def rematch(self, data):
        return self._matcher_fn(data)

# Usage
m = GComputation(
    steps.match(steps.input(matched_df), MyMatcher(matched_df, pair_ids, my_match_fn)),
    outcome=fit,
)
```

## Pickling and parallelism

`PysmatchClient` must be picklable for process-based parallel bootstrap
(`n_jobs > 1` on the estimator).  If the underlying matcher stores
non-picklable objects, set `n_jobs=1` on the estimator.

## Vcov behavior summary

| User provides | Behavior |
|---------------|----------|
| `steps.match(...)` only | Auto-derives `vcov = {"type": "cluster", "groups": matching.cluster_ids}` |
| `steps.match(...)` + `vcov="cluster"` | Uses user-supplied cluster specification |
| `steps.match(...)` + `vcov="HC3"` | User override; warning emitted because matched-set dependence is ignored |
| `steps.match(...)` + `vcov=ndarray` | User-supplied Σ̂ used directly; no validation against cluster structure |