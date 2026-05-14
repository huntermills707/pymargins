# Cluster and block bootstrap

For panel / multilevel data, pass `cluster=` at session construction:

```python
m = Margins.log_scale(
    fit,
    method="bootstrap",
    n_boot=2000,
    cluster=df["firm"].values,
)
```

Whole clusters are resampled with replacement; within-cluster
dependence is preserved.

For time-series data, pass `block_size=` to use moving-block
resampling:

```python
m = Margins.linear_scale(
    fit,
    method="bootstrap",
    n_boot=2000,
    block_size=8,
)
```

`cluster=` and `block_size=` are mutually exclusive. The block length
should span the dependence horizon: too short under-covers, too long
collapses to fewer effective draws.
