# Simultaneous confidence intervals and multiple-testing correction
When you ask for `m` margins in one call, the per-margin CIs are
*pointwise* by default. Family-wise coverage requires a wider
critical value.

## Correlated margins: simultaneous CIs

For a **single vector result** where the components are correlated
(e.g. predictions at adjacent ages, or slopes over several regions),
use `conf_int(simultaneous=True)`. This is correlation-aware and
strictly tighter than Bonferroni:

```python
res = m.predict(atexog={"age": [25, 35, 45, 55, 65]})

res.conf_int()                                # pointwise
res.conf_int(simultaneous=True)               # default simultaneous
res.conf_int(method="bonferroni")             # Bonferroni, any method
res.conf_int(method="sup-t")                  # requires draws (sim/bootstrap)
```

### Choosing a method

| Method       | Assumption                              | When to use |
|--------------|-----------------------------------------|-------------|
| `pointwise`  | None                                    | Single pre-registered comparison |
| `bonferroni` | None                                    | Conservative fallback; always valid but often wide |
| `sidak`      | Independent tests (or positively dependent) | Slightly less conservative than Bonferroni; good for large families |
| `sup-t`      | Draws available (simulation or bootstrap) | **Recommended** when tests are correlated — e.g. predictions at adjacent ages |

`sup-t` reads the family-wise critical value off the empirical
distribution of `max_j |t_j|` from the simulation or bootstrap draw
matrix. It is typically narrower than Bonferroni and Šidák when the
margins in the family are correlated — exactly the case for
predictions evaluated at neighbouring covariate profiles.

### Example: all-pairs comparisons with simultaneous CIs

```python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)

# sup-t uses the joint bootstrap/simulation draws
res.conf_int(method="sup-t")
```

## Independent results: multiple-comparison adjustment

When you have **independent results** from separate calls (e.g.
estimates from different cohorts or subgroups that do not share a
joint covariance), use `pymargins.adjust()`:

```python
from pymargins import adjust

results = {
    "cohort_a": m.predict(atexog={"tx": 1, "cohort": "a"}),
    "cohort_b": m.predict(atexog={"tx": 1, "cohort": "b"}),
    "cohort_c": m.predict(atexog={"tx": 1, "cohort": "c"}),
}

adj = adjust(results, method="holm")
print(adj.summary())
print(adj.to_frame())
```

`adjust` wraps `statsmodels.stats.multitest.multipletests` and accepts
any of its method names (`"bonferroni"`, `"holm"`, `"sidak"`,
`"fdr_bh"`, etc.). Pass a single `GraphResult`, a list, or a dict.

## Subgroup equality tests via pairwise contrasts

When you compute an estimand within subgroups (`over="region"`), the
natural follow-up is to test whether the subgroup estimates are equal.
Use `pairwise_contrasts()` to form all pairwise differences with proper
joint inference, then `joint_test()` or `adjust()`:

```python
from pymargins import diff_matrix, adjust

# Slopes within each region
slopes = m.dydx("age", over="region")

# All pairwise differences (k*(k-1)/2 contrasts)
diffs = slopes.pairwise_contrasts()
print(diffs.summary())

# Joint Wald test: H0 — all slopes are equal
print(diffs.joint_test().summary())

# Or correct for multiple comparisons across the pairs
adj = adjust(diffs, method="holm")
print(adj.summary())
```

### Building contrast matrices manually

`diff_matrix(k, kind="reference")` returns a `(k-1, k)` matrix of
each level versus the first level. Use it directly with `contrast()`
on any vector result:

```python
C = diff_matrix(4, kind="reference")   # 3 rows: [2-1, 3-1, 4-1]
ref_diffs = slopes.contrast(C, labels=["S-N", "E-N", "W-N"])
print(ref_diffs.summary())
```

For all pairwise differences, `kind="pairwise"` returns a
`(k*(k-1)/2, k)` matrix. `pairwise_contrasts()` is simply this matrix
applied automatically.

### When to use which

* **`conf_int(simultaneous=True)`** — one vector result with correlated
  components. Uses the exact joint covariance or empirical correlation
  from draws. Tighter and more powerful than generic p-value adjustment.

* **`pairwise_contrasts()`** — you already have a vector result and want
  to test all pairwise equalities. The contrasts share Σ̂, so joint
  inference is exact.

* **`adjust(...)`** — several independent results that do not share a
  estimator or Σ̂. The only option when results come from different models
  or non-overlapping samples.