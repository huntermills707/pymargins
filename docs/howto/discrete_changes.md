# Discrete changes for binary / categorical regressors

`dydx` on a binary regressor is a derivative — almost never what you
want. For a 0→1 discrete change, use `contrasts` with the
`pairwise` helper:

```python
from pymargins import Margins, pairwise

m = Margins.linear_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
m.contrasts(scenarios=scen, contrasts=w).summary()
```

For a multi-level factor, choose a baseline and contrast every other
level against it:

```python
from pymargins import reference

scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
m.contrasts(scenarios=scen, contrasts=W).summary()
```

All-pairs comparisons (returned in one result with a joint
covariance, so simultaneous CIs are available):

```python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)
res.conf_int(method="sup-t")
```
