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

# Exporting results


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
})
lp = -1.5 + 0.04 * df["age"] - 0.3 * df["female"]
df["y"] = rng.binomial(1, 1 / (1 + np.exp(-lp)))

fit = smf.glm("y ~ age + female", data=df,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, at="overall")
```


Every `MarginsResult` can be printed, framed, or serialized to LaTeX
/ HTML for inclusion in papers and reports.

```{code-cell} python
res = m.dydx("age", atexog={"female": [0, 1]})

print(res.summary(stars=True))      # text table with significance stars
res.to_frame()                      # pandas.DataFrame
print(res.to_latex())               # LaTeX tabular
print(res.to_html())                # HTML <table>
```

## Saving to CSV, Excel, or Parquet

`to_frame()` returns a tidy `pandas.DataFrame`, so any pandas export
works out of the box:

```{code-cell} python
df = res.to_frame()
df.to_csv("ame_results.csv", index=False)
df.to_excel("ame_results.xlsx", index=False)
df.to_parquet("ame_results.parquet")
```

The DataFrame includes scenario columns (e.g. `age`, `female`) when
available, making it ready for downstream plotting or reporting
without string parsing.

## Long-term storage with `materialize`

To save a result for later analysis without keeping the session and
gradient machinery alive, call `materialize()`:

```{code-cell} python
slim = res.materialize()           # estimates/SE/CI only; drops gradients
print(slim.summary())
```

Materialised results still support arithmetic (`+`, `-`, `*`, `/`,
`.scaled(by=...)`) for post-hoc combination.  The only thing you lose
is the ability to call `.conf_int(method="sup-t")` when the original
session used the delta method (sup-t requires draws, which are
dropped).
