# Reading and controlling the κ fallback

Every result records the inference path actually used. When the
session's κ threshold is exceeded, delta auto-falls-back to
simulation, and the summary annotates the fallback reason.

```python
m = Margins.log_scale(fit, kappa_threshold=0.3, method="delta", n_sim=4000)
res = m.predict(atexog={"x": [-3, 0, 3]})
print(res.summary())     # notes 'fallback: simulation (κ=0.42 > 0.30)'
```

## Pre-flight diagnostic

```python
print(m.diagnose().summary())
```

`diagnose()` samples the estimand surface and reports the κ
distribution; if it is uniformly large, change the inference scale
(see [](../tutorials/scales_and_kappa.md)) before paying for
simulation on every call.

## Disabling the fallback

Set `kappa_threshold=float("inf")` to force the chosen `method=` to
run regardless of curvature.

```python
m = Margins.log_scale(fit, kappa_threshold=float("inf"))
```

## Disabling diagnostics altogether

In tight loops, set `diagnostics=False` to skip the κ computation:

```python
m = Margins.log_scale(fit, diagnostics=False)
```
