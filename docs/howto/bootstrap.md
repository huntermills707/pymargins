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

The point estimate stays the analytic `g(β̂)` — `pymargins` does not
report the bootstrap mean as the estimate, matching Stata's
convention. Failed refits are caught; a `RuntimeWarning` fires when
the failure rate exceeds 5%.

For correlated data use [](cluster_block_bootstrap.md).
