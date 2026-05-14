# Elasticities and semi-elasticities

`dydx` returns the level derivative. The other three elasticity flavors
correspond to the standard Stata `dyex` / `eyex` / `eydx` methods —
build them as compositions through `evaluate`, or scale the AME by
hand:

| Stata `margins`    | Quantity                            | Build                                        |
|--------------------|-------------------------------------|----------------------------------------------|
| `dydx(x)`          | level change                        | `m.dydx("x")`                                |
| `dyex(x)`          | `dy/d(ln x)`                        | `m.dydx("x").scaled(by=x_bar)`               |
| `eyex(x)`          | full elasticity `(dy/dx) (x/y)`     | `m.dydx("x").scaled(by=x_bar / y_bar)`       |
| `eydx(x)`          | `d(ln y)/dx`                        | `m.dydx("x").scaled(by=1 / y_bar)`           |

`scaled` is a deterministic transform — it propagates SE, CI, and
covariance correctly under the delta method.

## Computing `x_bar` and `y_bar`

For an **average** elasticity (`at="overall"`), `x_bar` is the sample
mean of `x` and `y_bar` is the average predicted response at the
observed covariate profiles.  Both are easy to recover from the
session:

```python
x_bar = df["x"].mean()                       # or median, depending on theory
y_bar = m.predict().estimate.item()          # AAP on the response scale

# eyex: full elasticity at the mean
m.dydx("x").scaled(by=x_bar / y_bar)
```

For an elasticity **at a representative profile** (`at="typical"` or
with `atexog`), use the values at that profile:

```python
profile_x = 5.0
y_at = m.predict(atexog={"x": profile_x}).estimate.item()
m.dydx("x", atexog={"x": profile_x}).scaled(by=profile_x / y_at)
```

## Subgroup elasticities

Because `.scaled()` propagates the joint covariance, you can compute
elasticities for several subgroups and test differences between them:

```python
# Elasticity of x for female=0 and female=1
res_0 = m.dydx("x", atexog={"female": 0}).scaled(by=x_bar_0 / y_bar_0)
res_1 = m.dydx("x", atexog={"female": 1}).scaled(by=x_bar_1 / y_bar_1)

# Difference in elasticities with a proper SE
diff = res_1 - res_0
print(diff.summary())
```

Elasticities are only defined for continuous variables — `pymargins`
raises on discrete inputs.
