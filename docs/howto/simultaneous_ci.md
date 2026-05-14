# Simultaneous confidence intervals

When you ask for `m` margins in one call, the per-margin CIs are
*pointwise* by default. Family-wise coverage requires a wider
critical value.

```python
res = m.predict(atexog={"age": [25, 35, 45, 55, 65]})

res.conf_int(level=0.95)                       # pointwise
res.conf_int(level=0.95, method="bonferroni")  # works with any method
res.conf_int(level=0.95, method="sidak")
res.conf_int(level=0.95, method="sup-t")       # requires draws (sim/bootstrap)
```

## Choosing a method

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

## Example: all-pairs comparisons with simultaneous CIs

```python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)

# sup-t uses the joint bootstrap/simulation draws
res.conf_int(level=0.95, method="sup-t")
```
