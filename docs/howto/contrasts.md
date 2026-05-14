# Linear contrasts with `contrasts`

`Margins.contrasts` forms a **weighted sum of scenario predictions on the
inference scale**.  It is the workhorse for risk differences, risk
ratios, odds ratios, lift, reference-level comparisons, and
difference-in-differences — any estimand that is a linear combination
of predictions.

For nonlinear compositions (ratios on the raw scale, NNT, reciprocals,
custom utility functions) use `evaluate`.  See
[](../howto/contrasts_vs_evaluate.md) for the decision rule.

## How `contrasts` works

`contrasts` takes a list of **scenarios** and a **weight vector** (or
matrix).  The engine:

1. Computes the inference-scale prediction for each scenario:
   `hᵢ = φ⁻¹( mean_predict(beta, scenario_i) )`.
2. Forms the weighted sum: `Σᵢ wᵢ · hᵢ`.
3. Runs delta-method inference (or simulation/bootstrap if κ is high).
4. Back-transforms CI endpoints with `phi` for reporting.

Mathematically:

```
result = φ( Σᵢ wᵢ · φ⁻¹(pᵢ) )
```

where `pᵢ` is the aggregated response-scale prediction for scenario `i`.

Because the inference is on a **linear combination** of `φ⁻¹(pᵢ)`, the
delta method is exact to the extent that the individual `hᵢ` are
locally linear.  This is why simple contrasts usually have smaller κ
than `evaluate` calls.

## Risk difference (linear scale)

The simplest contrast: the arithmetic difference between two predicted
probabilities.

```python
from pymargins import Margins, pairwise

m = Margins.linear_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
res = m.contrasts(scenarios=scen, contrasts=w)
res.summary()
```

On the linear scale the contrast is `p₁ − p₀`.  The CI is symmetric on
the probability scale.

## Risk ratio (log scale)

A ratio is a difference on the log scale.  The back-transform turns the
log-ratio into a ratio with an asymmetric CI.

```python
m = Margins.log_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
res = m.contrasts(scenarios=scen, contrasts=w)
res.summary()
```

The point estimate is `exp(log(p₁) − log(p₀)) = p₁ / p₀`.  Because the
inference is on the log scale, the delta method is exact and κ is
small.

## Odds ratio (logit scale)

For probabilities near 0 or 1, the logit scale keeps the CI inside
(0, 1) for each arm before forming the odds ratio.

```python
m = Margins.logit_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
res = m.contrasts(scenarios=scen, contrasts=w)
res.summary()
```

The back-transform is `expit(logit(p₁) − logit(p₀))`, which simplifies
to the odds ratio `(p₁/(1−p₁)) / (p₀/(1−p₀))`.

## Lift (RR − 1)

Lift is a risk ratio minus one.  The easiest path is `log_scale` for
the ratio, then subtract one from the estimate and CI endpoints:

```python
m = Margins.log_scale(fit, at="overall")
res = m.contrasts(scenarios=scen, contrasts=w)

lift_est = float(res.estimate) - 1.0
lift_ci = (float(res.conf_int_lower) - 1.0, float(res.conf_int_upper) - 1.0)
```

Alternatively, `Margins.lift_scale` computes `(1+p₁)/(1+p₀) − 1` on
the `log1p`/`expm1` scale.  This is a different estimand; use it only
when your field convention requires it.

## Reference-level contrasts

Compare every level of a factor against a common baseline.  The weight
matrix has one row per comparison.

```python
from pymargins import reference

scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
res = m.contrasts(scenarios=scen, contrasts=W)
res.summary()
```

## All-pairs comparisons

Compare every level against every other level.  The result carries a
joint covariance, so simultaneous CIs are available.

```python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)

# Family-wise coverage
res.conf_int(level=0.95, method="sup-t")
```

## Testing a non-zero null

```python
# Test whether the risk ratio exceeds 1.5
m.log_scale(fit, at="overall").contrasts(scenarios=scen, contrasts=w).test(null=1.5)
```

The `null` value is interpreted on the **reporting scale** and lifted
onto the inference scale via `phi_inv` automatically.

## Contrasts over a grid

Evaluate the same contrast at several values of a moderator.  The
result is a vector estimand with one component per grid point.

```python
scen, w = pairwise("treated", [1, 0])
res = m.contrasts(
    scenarios=scen,
    contrasts=w,
    over="age",                     # or atexog={"age": [25, 45, 65]}
)
res.summary()
```

## 2×2 Difference-in-differences

DiD is a contrast across four cells with weights `[+1, −1, −1, +1]`.
See [](../howto/diff_in_diff.md) for the full derivation and
response-scale motivation (Ai & Norton, 2003).

```python
from pymargins import did

scen, w = did("group", "preexist",
              group_levels=["A", "B"], condition_levels=[0, 1])
m.contrasts(scenarios=scen, contrasts=w).summary()
```

## See also

- [](../howto/scenarios_helpers.md) for `pairwise`, `reference`, `grid`, etc.
- [](../howto/diff_in_diff.md) for DiD theory and examples.
- [](../howto/contrasts_vs_evaluate.md) for choosing between `contrasts`
  and `evaluate`.
- [](../math.rst) for the delta-method derivation.
