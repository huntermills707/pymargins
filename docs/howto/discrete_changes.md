# Discrete changes for binary / categorical regressors

`dydx` on a binary regressor is a derivative — it evaluates the slope
at the midpoint of the 0→1 jump (or averages midpoints over the
sample).  That is almost never the quantity you want.  The correct
quantity is the **discrete change**: the predicted difference between
the two levels, holding everything else constant.

## Binary regressor (0 → 1)

Use `contrasts` with the `pairwise` helper:

```python
from pymargins import Margins, pairwise

m = Margins.linear_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
m.contrasts(scenarios=scen, contrasts=w).summary()
```

## Multi-level factor (each level vs baseline)

```python
from pymargins import reference

scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
m.contrasts(scenarios=scen, contrasts=W).summary()
```

## All-pairs comparisons

All-pairs are returned in one result with a joint covariance, so
simultaneous CIs are available:

```python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)
res.conf_int(method="sup-t")
```
