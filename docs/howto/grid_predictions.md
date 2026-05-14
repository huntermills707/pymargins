# Grid predictions

For a Cartesian product of counterfactual values, use `grid` (or pass
a list to `atexog`):

```python
from pymargins import grid

# Equivalent ways to write the same 6-row grid
m.predict(scenarios=grid(age=[25, 45, 65], treatment=[0, 1]))
m.predict(atexog={"age": [25, 45, 65], "treatment": [0, 1]})
```

Variables not mentioned in `atexog` / `grid` follow the session's
`at=` rule (`"overall"` averages over the sample, `"typical"` /
`"mean"` hold them at a representative profile).

## Memory and large grids

`expand_scenario` materialises one block of rows per grid point. For
a 10-point grid over a 1M-row dataset that is 10M rows. Strategies:

- Use a smaller representative sample at session construction
  (`at="typical"`).
- Pass explicit `data=` overrides on the call to override the source
  rows.
- Call `result.materialize()` promptly on results you intend to keep
  long-term; this drops the heavy machinery (gradients, design
  matrices, session refs).
