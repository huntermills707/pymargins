# Simultaneous confidence intervals

When you ask for `m` margins in one call, the per-margin CIs are
*pointwise* by default. Family-wise coverage requires a wider
critical value.

```python
res = m.predict(atexog={"age": [25, 35, 45, 55, 65]})

res.conf_int(level=0.95)                       # pointwise
res.conf_int(level=0.95, method="bonferroni")  # works with any method
res.conf_int(level=0.95, method="sidak")
res.conf_int(level=0.95, method="sup-t")       # requires draws (sim/bootstrap)
```

`sup-t` reads the family-wise critical value off the empirical
distribution of `max_j |t_j|` from the simulation or bootstrap draw
matrix. It is typically narrower than Bonferroni and Šidák when the
margins in the family are correlated — exactly the case for
predictions evaluated at neighbouring covariate profiles.
