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

To save a result for later analysis without keeping the session and
gradient machinery alive, call `materialize()`:

```python
slim = res.materialize()           # estimates/SE/CI only; drops gradients
import joblib
joblib.dump(slim, "ame_age.joblib")
```

Materialised results still support arithmetic (`+`, `-`, `*`, `/`,
`.scaled(by=...)`) for post-hoc combination.
