# Plan and pre-registration

New in **pymargins 0.4.0**.

A `Plan` is an immutable, hashable record of every analysis choice that
determines the numbers you will report. It is created when you construct an
estimator and is copied onto every result.

## What is in the Plan

Everything that defines the analysis:

- graph topology (`steps.*` node kinds and content hashes)
- estimand frame: `at`, `scale`, `method` (declared and resolved), `vcov`,
  `ci`, `level`, `B`, `n_sim`, `seed`
- data fingerprint and weights fingerprint
- constants overrides (e.g. κ borderline)
- population note from matching/transform stages

Execution knobs such as `n_jobs` and `progress_bar` are **not** in the Plan —
they change speed, not results.

## Reading the Plan

```python
from pymargins import GComputation

est = GComputation(model, method="delta", level=0.95)
print(est.plan.hash)        # short hash with recipe suffix, e.g. 'a7f3c21@1'
print(est.plan.describe())  # human-readable summary
```

The short hash appears in every `summary()` footer, making the analysis plan
visible in any rendered output.

## The level/CI doctrine

The confidence level and interval method are fixed at construction:

- `level` is the coverage probability (default 0.95).
- `ci` is the interval method: `"wald"` for delta and simulation;
  `"percentile"`, `"basic"`, `"bca"`, or `"studentized"` for bootstrap.

They cannot be changed on a result. Calling

```python
result.conf_int(level=0.90)  # level is locked; declare a new GComputation to change it
```

raises `TypeError`:

> `conf_int()` takes no `level=`. The confidence level is declared at the
> estimator constructor (`level=<x>` in this plan) and is part of the
> pre-registered analysis. To report at a different level, declare a new
> estimator (the recompute is cheap; the new plan hash is the point).

Family-wise corrections (`correction="bonferroni"`, `"sidak"`, `"sup-t"`)
are still available; they allocate the *declared* level across multiple
comparisons and only widen intervals.

## Why lock the level?

Pre-registration is not about preventing exploration — it is about making the
choice visible. Any change in `level`, `scale`, `method`, `vcov`, or `ci`
changes the plan hash. Two estimators with different hashes are different
analyses; a reviewer can see where the switch happened.

## Re-computing at a different level is cheap

Because `GComputation` stores nothing stateful and the graph is pure,
creating a second estimator with a different level costs one point execution
and one fit, not a full re-run of bootstrap replicates:

```python
est_90 = GComputation(model, method="delta", level=0.90)
```

## Plan hash stability

The hash recipe is versioned (`a7f3c21@1`). If the canonical serialization
changes in a future release, the recipe suffix is bumped so that plan hashes
never silently collide across package versions.
