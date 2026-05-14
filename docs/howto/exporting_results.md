# Exporting results

Every `MarginsResult` can be printed, framed, or serialized to LaTeX
/ HTML for inclusion in papers and reports.

```python
res = m.dydx("age", atexog={"female": [0, 1]})

print(res.summary(stars=True))      # text table with significance stars
res.to_frame()                      # pandas.DataFrame
print(res.to_latex())               # LaTeX tabular
print(res.to_html())                # HTML <table>
```

## Saving to CSV, Excel, or Parquet

`to_frame()` returns a tidy `pandas.DataFrame`, so any pandas export
works out of the box:

```python
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

```python
slim = res.materialize()           # estimates/SE/CI only; drops gradients
import joblib
joblib.dump(slim, "ame_age.joblib")
```

Materialised results still support arithmetic (`+`, `-`, `*`, `/`,
`.scaled(by=...)`) for post-hoc combination.  The only thing you lose
is the ability to call `.conf_int(method="sup-t")` when the original
session used the delta method (sup-t requires draws, which are
dropped).
