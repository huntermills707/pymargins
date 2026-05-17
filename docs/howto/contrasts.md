---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Linear contrasts with `contrasts`


```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import Margins

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "age": rng.integers(20, 75, n),
    "female": rng.binomial(1, 0.52, n),
    "treated": rng.binomial(1, 0.40, n),
    "region": rng.choice(["N", "S", "E", "W"], n),
    "group": rng.choice(["A", "B"], n),
    "preexist": rng.binomial(1, 0.4, n),
})
lp = (-1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
      + 0.2 * (df["region"] == "S") + 0.4 * (df["region"] == "E")
      - 0.1 * (df["region"] == "W")
      + 0.6 * (df["group"] == "B") + 0.4 * df["preexist"])
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + female + treated + C(region) + C(group) + preexist", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```


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

```{code-cell} python
from pymargins import Margins, pairwise

m = Margins.linear_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
res = m.contrasts(scenarios=scen, contrasts=w)
print(res.summary())
```

On the linear scale the contrast is `p₁ − p₀`.  The CI is symmetric on
the probability scale.

## Risk ratio (log scale)

A ratio is a difference on the log scale.  The back-transform turns the
log-ratio into a ratio with an asymmetric CI.

```{code-cell} python
m = Margins.log_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
res = m.contrasts(scenarios=scen, contrasts=w)
print(res.summary())
```

The point estimate is `exp(log(p₁) − log(p₀)) = p₁ / p₀`.  Because the
inference is on the log scale, the delta method is exact and κ is
small.

## Odds ratio (logit scale)

For probabilities near 0 or 1, the logit scale keeps the CI inside
(0, 1) for each arm before forming the odds ratio.

```{code-cell} python
m = Margins.logit_scale(fit, at="overall")

scen, w = pairwise("treated", [1, 0])
res = m.contrasts(scenarios=scen, contrasts=w)
print(res.summary())
```

The back-transform is `expit(logit(p₁) − logit(p₀))`, which simplifies
to the odds ratio `(p₁/(1−p₁)) / (p₀/(1−p₀))`.

## Lift (RR − 1)

Lift is a risk ratio minus one.  The easiest path is `log_scale` for
the ratio, then subtract one from the estimate and CI endpoints:

```{code-cell} python
m = Margins.log_scale(fit, at="overall")
res = m.contrasts(scenarios=scen, contrasts=w)

lift_est = float(res.estimate) - 1.0
lift_ci = (float(res.conf_int_lower) - 1.0, float(res.conf_int_upper) - 1.0)
```

## Reference-level contrasts

Compare every level of a factor against a common baseline.  The weight
matrix has one row per comparison.

```{code-cell} python
from pymargins import reference

scen, W = reference("region", ["N", "S", "E", "W"], ref_level="N")
res = m.contrasts(scenarios=scen, contrasts=W)
print(res.summary())
```

## All-pairs comparisons

Compare every level against every other level.  The result carries a
joint covariance, so simultaneous CIs are available.

```{code-cell} python
from pymargins import all_pairwise

scen, W = all_pairwise("region", ["N", "S", "E", "W"])
res = m.contrasts(scenarios=scen, contrasts=W)

# Joint covariance is stored in the result for post-hoc combination
print(res.summary())
```

## Testing a non-zero null

```{code-cell} python
# Test whether the risk ratio exceeds 1.5
scen_test, w_test = pairwise("treated", [1, 0])
print(m.log_scale(fit, at="overall").contrasts(scenarios=scen_test, contrasts=w_test).test(value=1.5).summary())
```

The `null` value is interpreted on the **reporting scale** and lifted
onto the inference scale via `phi_inv` automatically.

## Predictions over a grid

Evaluate predictions at several values of a moderator:

```{code-cell} python
res = m.predict(atexog={"age": [25, 45, 65], "treated": [0, 1]})
print(res.summary())
```

For a contrast at a specific moderator value, build the scenarios
explicitly:

```{code-cell} python
scen, w = pairwise("treated", [1, 0])
res = m.contrasts(
    scenarios=[
        {"atexog": {"treated": 1, "age": 45}, "label": "treated@45"},
        {"atexog": {"treated": 0, "age": 45}, "label": "control@45"},
    ],
    contrasts=w,
)
print(res.summary())
```

## 2×2 Difference-in-differences

DiD is a contrast across four cells with weights `[+1, −1, −1, +1]`.
See [](../howto/diff_in_diff.md) for the full derivation and
response-scale motivation (Ai & Norton, 2003).

```{code-cell} python
from pymargins import did

scen, w = did("group", "preexist",
              treated_level="B", control_level="A",
              post_level=1, pre_level=0)
print(m.contrasts(scenarios=scen, contrasts=w).summary())
```

## See also

- [](../howto/scenarios_helpers.md) for `pairwise`, `reference`, `grid`, etc.
- [](../howto/diff_in_diff.md) for DiD theory and examples.
- [](../howto/contrasts_vs_evaluate.md) for choosing between `contrasts`
  and `evaluate`.
- [](../math.rst) for the delta-method derivation.
