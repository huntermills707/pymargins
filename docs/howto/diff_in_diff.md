# Difference-in-differences on the response scale

For a 2×2 DiD on a nonlinear model, evaluate the four cells on the
response scale and difference them. The interaction coefficient
itself is on the link scale and does *not* answer the question
(Ai & Norton, 2003).

```python
from pymargins import Margins, did

m = Margins.linear_scale(fit, vcov="HC3", at="overall")

scen, w = did(
    "group", "preexist",
    group_levels=["A", "B"],
    condition_levels=[0, 1],
)
m.contrasts(scenarios=scen, contrasts=w).summary()
```

At a single representative patient profile:

```python
m.contrasts(
    scenarios=did(
        "group", "preexist",
        group_levels=["A", "B"], condition_levels=[0, 1],
        age=60, female=0,
    )[0],
    contrasts=[+1, -1, -1, +1],
)
```

The four cell predictions and the two simple effects share the same
joint covariance, so the DiD's standard error is exact under the
delta method.
