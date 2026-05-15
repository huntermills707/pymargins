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
})
lp = -1.5 + 0.04 * df["age"] - 0.3 * df["female"] + 0.8 * df["treated"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + female + treated", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```

# The scenarios model — `at` vs `atexog`

Two knobs control *where in covariate space* a margin is evaluated.
They live at different levels of the API.

## `at` — session-level aggregation rule

`at=` is set at session construction. It controls the default
evaluation rule for variables *not* otherwise pinned:

| Value          | Per-variable behavior                                     |
|----------------|-----------------------------------------------------------|
| `"overall"`    | take the observed value on every row, then average        |
| `"typical"`    | median for continuous, mode for discrete                  |
| `"mean"`       | mean for all (errors on non-numeric)                      |
| `"median"`     | median for all                                            |
| `"mode"`       | mode for all (errors on continuous)                       |
| dict           | per-variable override, with `_default` as a fallback rule |
| callable       | `(data) -> 1-row DataFrame`, fully bespoke                |

`"overall"` corresponds to Stata's bare `margins` and gives AAP / AME.
`"typical"` corresponds to `margins, atmeans` for mixed factor /
continuous models and gives APM / MEM.

## `atexog` — per-call counterfactual pins

`atexog=` is a per-call dict (or a list of dicts wrapped as
`scenarios=`) that pins specific variables to specific values. A
list-valued entry produces a Cartesian product (a grid). Variables
not mentioned in `atexog` follow the session's `at=` rule.

```{code-cell} python
# AAP at age=25, 45, 65, averaging the rest over the sample
print(Margins.log_scale(fit, at="overall").predict(
    atexog={"age": [25, 45, 65]}
).summary())

# APR at age=25, 45, 65, others held at typical profile
print(Margins.log_scale(fit, at="typical").predict(
    atexog={"age": [25, 45, 65]}
).summary())
```

## Why split it this way?

Because the aggregation choice is a methodological commitment (AME
vs MEM is an argument; if you flip mid-analysis the audit trail
should show it) but the counterfactual pins are not (you genuinely
do want to evaluate the same AME at several age points).

This is the same logic behind keeping `phi`, `vcov`, `level`, and
`method` session-level: the analytical *posture* belongs in the
constructor; the analytical *question* belongs in the method call.

See [](session_precommitment.md).
