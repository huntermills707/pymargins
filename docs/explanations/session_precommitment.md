# The session pre-commitment

`Margins` is a session. Once constructed, it commits to:

- the inference scale (`phi`, `phi_inv`)
- the variance estimator (`vcov`)
- the confidence level (`level`)
- the default evaluation point (`at`)
- the default inference method (`method`)

Every subsequent computation on the session inherits these
commitments. Switching any of them requires a new session.

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

See [](inference_scale.md), [](kappa_diagnostic.md), and
[](delta_sim_bootstrap.md) for why each particular commitment matters.
