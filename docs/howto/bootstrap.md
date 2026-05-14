# Bootstrap inference

Switch the session's inference method to `"bootstrap"` and pick the
number of replicates. The default scheme is *pairs* — rows resampled
IID with replacement.

```python
m = Margins.log_scale(fit, method="bootstrap", n_boot=2000, vcov="HC3")
m.dydx("age").summary()
```

Parallelism uses thread pools; BLAS threads are pinned to 1 per worker
to avoid oversubscription:

```python
m = Margins.log_scale(fit, method="bootstrap", n_boot=2000, n_jobs=-1)
```

## Point estimates under bootstrap

The point estimate stays the analytic `g(β̂)` — `pymargins` does not
report the bootstrap mean as the estimate, matching Stata's
convention.  The bootstrap is used *only* for the standard error and
the empirical quantiles of the CI.  This keeps the estimator
consistent even when the bootstrap distribution is biased (e.g. in
small samples).

## Failed refits

Failed refits are caught and counted; a `RuntimeWarning` fires when
the failure rate exceeds 5%.  If you see this warning, inspect the
model specification — non-convergence on 5% of bootstrap samples
usually indicates separation, perfect multicollinearity, or a
misspecified link function.

For correlated data use [](cluster_block_bootstrap.md).
