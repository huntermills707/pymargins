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

# Exporting and persisting results
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.


```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pymargins import GComputation  # 0.4.0: Margins -> GComputation

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

## Long-term storage with `to_disk` / `from_disk`

For checkpointing or sharing results between scripts, use the built-in
pickle persistence methods. They automatically materialize the result
(so the saved object is self-contained) and include a version check on
load:

```{code-cell} python
# Save
res.to_disk("ame_results.pkl")

# Load in another session
from pymargins import MarginsResult
loaded = MarginsResult.from_disk("ame_results.pkl")
print(loaded.summary())
```

Materialised results support `.summary()`, `.to_frame()`, `.to_latex()`,
and `.to_html()`, but they **cannot** recompute CIs at new confidence
levels or run new hypothesis tests because the gradient and session
reference are dropped during materialisation.

## In-memory slimming with `materialize`

If you only need to reduce memory inside a single Python session (e.g.
when accumulating many results in a loop), call `materialize()` directly:

```{code-cell} python
slim = res.materialize()           # estimates/SE/CI only; drops gradients
print(slim.summary())
```

Materialised results still support arithmetic (`+`, `-`, `*`, `/`,
`.scaled(by=...)`) for post-hoc combination.  The only thing you lose
is the ability to call `.conf_int(level=...)` when the original
session used the delta method (recomputing CIs at new levels requires
the gradient, which is dropped).