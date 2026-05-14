# Scenario helpers

The `pymargins.scenarios` module ships factories for common contrast
patterns so you don't have to hand-build dicts and weight vectors.

```python
from pymargins import pairwise, reference, at_levels, grid, did, diff, all_pairwise
```

## `pairwise(var, [a, b])` — two scenarios, `[+1, -1]` contrast

```python
scen, w = pairwise("treatment", [1, 0])
m.contrasts(scenarios=scen, contrasts=w)
```

## `reference(var, levels, ref_level=...)` — each level vs baseline

```python
scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
m.contrasts(scenarios=scen, contrasts=W)
```

## `all_pairwise(var, levels)` — every level pair

```python
scen, W = all_pairwise("region", ["N", "S", "E", "W"])
```

## `at_levels(var, levels=[...])` — predict at each level

```python
m.predict(scenarios=at_levels("region", levels=["N", "S", "E", "W"]))
```

## `grid(**vars)` — Cartesian product of counterfactual values

```python
m.predict(scenarios=grid(age=[25, 45, 65], treatment=[0, 1]))
```

## `did(g, c, group_levels=..., condition_levels=...)` — 2×2 DiD

```python
scen, w = did("group", "preexist",
              group_levels=["A", "B"], condition_levels=[0, 1])
m.contrasts(scenarios=scen, contrasts=w)
```

## `diff(n)` — `[-1, 0, …, 0, +1]` for an ordered grid

```python
scen = at_levels("age", levels=[25, 45, 65])
m.contrasts(scenarios=scen, contrasts=diff(3))   # age=65 minus age=25
```
