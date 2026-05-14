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

For elasticities at representative values, evaluate the AME at the
profile and then scale by `x_bar = atexog["x"]` and `y_bar =
predicted_at_profile`:

```python
y_at = m.predict(atexog={"x": 5.0}).estimate.item()
m.dydx("x", atexog={"x": 5.0}).scaled(by=5.0 / y_at)
```

Elasticities are only defined for continuous variables — `pymargins`
raises on discrete inputs.
