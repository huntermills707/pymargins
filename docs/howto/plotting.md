# Plotting predictions and effects

`MarginsResult.to_frame()` returns a plot-ready table. Combine with
matplotlib for prediction curves and forest plots.

## Prediction curve over a continuous variable

```python
import matplotlib.pyplot as plt

ages = list(range(20, 81, 2))
res = m.predict(atexog={"age": ages})
df = res.to_frame()

fig, ax = plt.subplots()
ax.plot(df["age"], df["estimate"])
ax.fill_between(df["age"], df["ci_lower"], df["ci_upper"], alpha=0.25)
ax.set(xlabel="age", ylabel="P(y=1)")
```

## Forest plot of contrasts

```python
fig, ax = plt.subplots()
y = range(len(df))
ax.errorbar(df["estimate"], y,
            xerr=[df["estimate"] - df["ci_lower"],
                  df["ci_upper"] - df["estimate"]],
            fmt="o")
ax.axvline(0, color="grey", lw=0.5)
ax.set_yticks(list(y))
ax.set_yticklabels(df["label"])
```

## Subgroup curves (`atexog` with two variables)

`predict` with a multi-variable `atexog` returns a long-form table
with one row per grid point — group by the conditioning variable
when plotting.

```python
res = m.predict(atexog={"age": ages, "female": [0, 1]})
df = res.to_frame()
for level, sub in df.groupby("female"):
    ax.plot(sub["age"], sub["estimate"], label=f"female={level}")
```
