# The session pre-commitment

`Margins` is a session. Once constructed, it commits to:

- the inference scale (`phi`, `phi_inv`)
- the variance estimator (`vcov`)
- the confidence level (`level`)
- the default evaluation point (`at`)
- the default inference method (`method`)

Every subsequent computation on the session inherits these
commitments. Switching any of them requires a new session.

## Operational commitment: the inference distribution

The declarative commitments above are the *user-facing* contract.
Operationally, the session also commits to the **random objects**
that implement those choices:

- **Σ̂** (`cov_params`) is resolved once at construction and frozen
  onto every result. Mutating the underlying model later does not
  change the session's analytical posture.

- **Bootstrap resample indices** are generated once (the resample-index
  bank) and reused for every bootstrap call in the session.

- **Refitted adapters** are harvested once (the bootstrap states cache)
  and replayed for every subsequent bootstrap-method call. Two calls
  on the same session evaluate their estimands on the *exact same*
  sequence of refitted models.

- **Simulation β* draws** are generated once (the simulation draws bank)
  and reused for every simulation call in the session.

This means a session commits to an **inference distribution**, not
just inference parameters. Once any of these caches exists, mutation
of the inputs that fed it raises `RuntimeError`.

## Joint inference across calls

Because the resample indices and simulation draws are session-level
banks, results from separate calls are **jointly composable**.

```python
m = Margins(fit, method="bootstrap", n_boot=1000, rng_seed=42)
p0 = m.predict(atexog={"treatment": 0})
p1 = m.predict(atexog={"treatment": 1})

# p0.draws_inf[b] and p1.draws_inf[b] come from the same refitted model
# for every replicate b.  The ratio is a valid bootstrap estimand.
ratio_draws = p1.draws_inf / p0.draws_inf
```

The same alignment holds for simulation-method sessions: every
result's `draws_inf` rows share the same underlying β* draws.

`MarginsResult` enforces this safety net via `_check_draws_match`.
If you somehow manage to produce two results whose draws are not
aligned (e.g. by bypassing the session), composition raises a clear
error.

## Mutability

Once an inference cache has been materialized, the following
attributes are frozen:

| Attribute | Why frozen |
|-----------|------------|
| `method` | Determines which cache type is built |
| `n_boot` | Determines the size of the resample bank |
| `rng_seed` | Determines the random stream |
| `n_sim` | Determines the size of the simulation bank |
| `cluster` | Determines the resampling units |
| `block_size` | Determines the block-bootstrap geometry |
| `bootstrap_config` | Determines resampling details (block type, etc.) |
| `matching` | Determines the rematching and cluster structure |

Attempting to mutate any of these after the cache exists raises:

```
RuntimeError: Cannot mutate 'n_boot' after the inference cache has been
materialized. Construct a new Margins session.
```

**Adapter mutation is out of scope.** Re-fitting the underlying model
outside the session is not caught by attribute freezing. The session
detects this via `_check_adapter_drift`: if `adapter.coefficients()`
changes after the cache was built, the next inference call raises:

```
RuntimeError: adapter.coefficients() has changed since the inference
cache was built. The cache is invalid. Construct a new Margins session.
```

To change any of the above, construct a fresh `Margins(...)` instance.

## Performance characteristics

- **Bootstrap sessions**: the first bootstrap-method call pays the full
  `n_boot` refit cost. Every subsequent call on the same session is
  evaluation-only (replaying `h_factory` over the cached states).

- **Simulation sessions**: the first simulation-method call materializes
  Σ̂ and the β* draw bank. Subsequent calls reuse both.

- **Delta-method sessions**: unaffected. There is no random object to
  cache, so each call is an independent analytical computation.

This makes multi-estimand workflows dramatically cheaper. A survival
multi-time curve with 10 time points costs ~1 bootstrap refit pass,
not 10.

## Why pre-commit?

The honest story: in observational analysis there are *several*
defensible choices for each of those five knobs. The data analyst's
job is to declare the choice up front, not to shop for the
combination that gives the desired conclusion.

A session forces that declaration. The constructor call is the
analytical posture — a reviewer reads one line and knows what scale,
what vcov, what level the entire downstream analysis is on. Changing
any of them shows up as a new `Margins(...)` in the audit trail; it
cannot quietly happen between calls.

The contrast tool here is per-call configuration (Stata's `margins`,
R's `marginaleffects`). Both are excellent, more flexible, and
strictly more permissive. They also make it very easy to write the
following script without noticing:

```python
m.contrast("treated", vce="robust")          # 0.13 (p = 0.08)
m.contrast("treated", vce="cluster", id=...) # 0.13 (p = 0.04)
```

`pymargins` is opinionated about exposing that switch. To get the
cluster SE you must construct a new session — and you, your reviewer,
and your future self can all see it happened.

## `strict=True`

For pre-registration and CI:

```python
m = Margins(fit, strict=True, vcov="HC3", level=0.95,
            at="overall", method="delta", phi=..., phi_inv=...)
```

Any unspecified session-level argument raises `ValueError` at
construction. The session refuses to fall back to defaults.

## Implications for implementers

- Do not add per-call overrides for `phi`, `vcov`, `level`, `at`,
  or `method` without a strong reason.
- `Margins.summary()` is the methods-section paragraph for the
  analysis. New session-level commitments must show up there.

See [](inference_scale.md), [](kappa_diagnostic.md),
[](delta_sim_bootstrap.md), [](howto/bootstrap.md),
[](howto/cluster_block_bootstrap.md), and [](howto/matching.md)
for why each particular commitment matters.
