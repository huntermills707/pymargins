# pymargins — Design Primer

This document is the entry point for implementers. It captures the
architectural philosophy and the design decisions that shaped the API,
so that downstream implementation work proceeds from the right premises
without relitigating questions that have already been settled.

Read this *before* reading any of the module files. The modules assume
the mental model below.

---

## 1. What this library is — and what it isn't

`pymargins` computes marginal-effects-style quantities — adjusted
predictions, slopes, contrasts, diff-in-diff, custom linear and nonlinear
combinations of predictions — from fitted statistical models, with
uncertainty quantification via the delta method, parametric simulation
(Krinsky–Robb), or nonparametric bootstrap.

It is **not** a general-purpose post-estimation library. It is a
deliberately narrow tool focused on one kind of question: "what does my
model predict, how does that prediction change across counterfactuals,
and how confident should I be?"

**What this library is not:**

- It is not `marginaleffects`. That library exists; we converged on a
  similar architectural shape because the math forces convergence, but we
  diverge in posture (see §2). If a user wants the broadest coverage and
  the gentlest learning curve, point them at `marginaleffects`.

- It is not a model-fitting library. Models are fit elsewhere
  (statsmodels, linearmodels, sklearn, etc.). We wrap fitted results.

- It is not a framework for arbitrary post-estimation. The estimands we
  compute are predictions, slopes, contrasts, and differentiable
  combinations of these. Anything that requires re-solving the model or
  altering the likelihood is out of scope.

---

## 2. The core philosophical commitment: session-level pre-commitment

This is the design decision that most distinguishes us from other tools
and shapes the rest of the architecture. Internalize it.

A `Margins` instance commits up front to:

- the inference scale (`phi`, `phi_inv`)
- the variance estimator (`vcov`)
- the confidence level (`level`)
- the default evaluation point (`at`)
- the default inference method (`method`)

Once the session is constructed, every subsequent computation inherits
these commitments. **Switching any of them requires a new session.**

### Why?

This forces the analyst to declare their analytical posture explicitly,
in code, at one location. A reviewer or auditor reading the code sees
the entire methodological commitment in the constructor call. Any change
in posture — different scale, different vcov, different level — requires
a new `Margins(...)` and shows up as such in the audit trail.

The alternative (per-call configuration, as in `marginaleffects` and
`margins`) is more flexible but invites a particular form of p-hacking:
trying multiple scales, multiple vcov flavors, multiple levels until
something looks significant. We make this *possible but visible*: you
can do it, but you'll have multiple `Margins` instances in your code,
each documenting its own posture.

### What this means for implementers

- Resist the temptation to add per-call overrides for session-level
  commitments unless there is a strong reason. The current API does not
  permit per-call overrides for method, level, scale, vcov, or `at`;
  switching any of them requires constructing a new `Margins` instance.

- The session's `summary()` method is the methods-section paragraph for
  the analysis. Take it seriously — make sure new session-level
  commitments show up there.

- A `strict=True` mode disables all defaults — any unspecified config
  argument raises `ValueError` at construction. Implemented via a
  `_NOT_GIVEN` sentinel.

---

## 3. The three orthogonal axes

Almost every design choice in the library reflects this three-axis
decomposition. Hold it in your head:

1. **Estimand**: what quantity is being computed?
   Prediction (level), slope (∂μ/∂x), contrast (linear combination of
   predictions), or arbitrary nonlinear composition. Per-call.

2. **Aggregation**: where in the design space is it evaluated?
   Empirical (per-row, then averaged → AME/AAP), at-typical (single
   representative point → MEM/APM), or at-representative (user-supplied
   grid via `atexog=` → MER/APR). Session-level default controlled by
   the session's `at` argument; per-scenario values via `atexog=`.

3. **Inference**: how is uncertainty quantified?
   Delta method, parametric simulation, or bootstrap. Session-level
   default; per-call override.

These are *orthogonal*. Any combination is meaningful: AME of a slope
under bootstrap, APM of a prediction under simulation, etc. The code
should keep them factored — never combine "AME-via-delta" into a single
code path that can't be decomposed.

---

## 4. The estimand triple

Internal to the library, every estimand is a triple `(h, phi, phi_inv)`:

- `h(β) → scalar or vector` — the estimand function on the **inference
  scale**. This is what gets differentiated for delta and what gets
  evaluated for simulation.
- `phi` — back-transform from inference scale to reporting scale,
  applied to CI endpoints (and the point estimate if it's not already on
  the reporting scale).
- `phi_inv` — forward transform, used to lift response-scale quantities
  to the inference scale internally and to convert user-supplied null
  values onto the inference scale for hypothesis tests.

The phi/phi_inv pair is **session-level** (not per-estimand). All
estimands within one session are on the same inference scale, which is
what makes the session-level κ diagnostic and the inter-call
composability work.

The `h` function itself is built per-call from user inputs (variables,
scenarios, contrasts) by the factories in `_estimands.py`.

---

## 5. The five distinctive design choices we made

These are the choices most likely to be questioned during implementation
or by reviewers. Each has a reason; don't undo them lightly.

### 5.1 The κ curvature diagnostic with auto-fallback

We compute Skovgaard's relative curvature κ for every estimand (when
diagnostics are enabled) and auto-fall-back to simulation when κ
exceeds the session threshold (default 0.3). This is a meaningful
divergence from `marginaleffects` and Stata's `margins`, both of which
just do delta and never tell you when delta is suspect.

The κ math lives in `_kappa.py`. The whitening transform via Cholesky is
critical — without it, κ is parameterization-dependent and meaningless.
The thresholds (0.1 / 0.3) are calibrated from the nonlinear-regression
literature; expose them as configurable but keep these as defaults.

### 5.2 `atexog` as the scenario specification key

Scenario dicts passed to `predict()`, `dydx()`, `contrasts()`, and
`evaluate()` use a single key `atexog` to specify counterfactual values:

```python
m.predict(atexog={"treatment": [0, 1]})
m.dydx("age", atexog={"treatment": 1})
```

`atexog` accepts per-variable values or lists of values (which produce a
grid via Cartesian product). Variables not mentioned in `atexog` are filled
per the session's `at` setting ("overall" uses observed values, "typical"
uses type-aware representative values, etc.).

`at` is a **session-level** setting, not a per-scenario key. It controls
the default evaluation rule; `atexog` overrides specific variables within
that rule.

### 5.3 The `phi` is session-level, not per-call

Fully discussed above. This is the architectural commitment that
underwrites the κ diagnostic (κ has a stable referent only when the
inference scale is fixed) and the inter-call composability (results
from the same session share an inference scale).

### 5.4 Three-tier gradient backend

- **autodiff**: pure JAX path. The cleanest when the prediction can be
  reimplemented in JAX (most GLMs, all linear models).
- **wrapped_fd**: autodiff over the estimand structure, with FD only at
  the model's predict boundary via custom JVP. Hides the FD inside a
  primitive; downstream autodiff is exact. Recommended for black-box
  models where JAX reimplementation would be error-prone.
- **fd**: full FD over the entire estimand. Fallback when no JAX path
  exists at all. Rarely needed; gradient quality is fine but Hessians
  (for κ) compound poorly.

The custom JVP pattern (`make_predict_with_fd_jvp` and
`make_glm_jvp_wrapper` in `_gradients.py`) is the bridge between
non-JAX models and the JAX-based inference engine. Implementers
adding adapters for new model classes should choose between native JAX
predict, analytical-derivative JVP, or FD-JVP based on what the
underlying framework exposes.

### 5.5 Linear combinations as the contrast primitive

`Margins.contrasts(scenarios=..., contrasts=[w1, w2, ...])` is the
canonical contrast method. It accepts a list of scenarios and either a
single weight vector (one contrast → scalar result) or a dict of named
weight vectors (multiple contrasts → vector result with joint
inference). This subsumes pairwise contrasts, diff-in-diff, triple
difference, and arbitrary linear hypotheses.

There is intentionally no `contrasts="pairwise"` shorthand at the API
level. Users specify scenarios and weights explicitly. Convenience
helpers (in a separate `pymargins.contrasts` helper module, when
written) generate scenarios+weights for common patterns; users still
pass them through the canonical method.

The reason is the same pre-commitment philosophy: explicit at the call
site, math visible to reviewers.

---

## 6. The two scales of "obvious" that need to remain distinct

A subtle point that's easy to confuse:

- **Response scale** vs **link scale** vs **inference scale** are three
  different things. See discussion in `margins.py` docstrings and the
  conversation that produced this design.

- **Response scale** is how the model speaks about y (probabilities for
  logit, counts for Poisson). The model produces this directly.
- **Link scale** is how the model is linear in β (log-odds for logit).
  Property of the model.
- **Inference scale** is how delta-method math happens (configurable by
  the session via phi). Property of the analysis.

These can all differ in one analysis. A logit model fit on log-odds,
predicted on probability, contrasted as a relative risk on log scale —
three scales, all in play.

User-facing `at=` values are always specified on the **data scale** of
the variable as it was fit (so `at={"age": 65}` means age 65 in the
units the model was fit on). No scale conversion happens for user
inputs.

---

## 7. Composability rules

Two `MarginsResult` objects from the same `Margins` session can be
combined via `+`, `-`, `*` (by scalar), `/` (by scalar) to produce a new
`MarginsResult` with proper joint inference using the shared Σ̂.

Two `MarginsResult` objects from **different** sessions cannot be
combined. The arithmetic operators raise `ValueError`. Combining across
sessions would require explicit scale conversion logic that we
deliberately don't provide.

Multiplication and division of two `MarginsResult` objects (as opposed
to result × scalar) is **not** supported via operators because these
are nonlinear and require autodiff through the composition. Users who
want product or ratio of two results should use `m.evaluate(...)` with
a custom `compose=` function.

---

## 8. The architectural layers

```
┌─────────────────────────────────────────────────────────────┐
│ User-facing                                                  │
│   margins.py   — Margins class (session)                     │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│ User-facing + orchestration                                   │
│   _result.py   — MarginsResult straddles both layers         │
│                  (test() calls _inference, composition calls │
│                  _delta; imports are top-level, not lazy)    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│ Orchestration                                                 │
│   _inference.py   — engine: dispatch delta/sim/bootstrap     │
│   _estimands.py   — estimand factory functions               │
│   _scenarios.py   — counterfactual design construction       │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│ Numerical kernels                                             │
│   _gradients.py  — gradient/Hessian/JVP wrappers             │
│   _delta.py      — delta-method math                         │
│   _kappa.py      — curvature diagnostic                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│ Framework adapters                                            │
│   _adapter.py            — abstract interface                │
│   _adapters/*.py         — concrete framework implementations │
└──────────────────────────────────────────────────────────────┘
```

Dependencies flow downward only, with the exception of `_result.py`,
which imports `_inference` and `_delta` at the top of the module so
that `MarginsResult.test()` and the composition operators can call
into the orchestration and kernel layers without late-import hacks.
This makes `_result.py` a straddling layer: user-facing for reporting,
but dependent on orchestration/kernels for computable operations.
The numerical kernels know nothing about adapters or sessions;
orchestration knows about kernels but not adapters; adapters bridge
orchestration and frameworks.

---

## 9. What's explicitly out of scope (deliberately)

These have come up and been deferred:

- **Bayesian inference paths via posterior draws.** The current
  `simulation` method does Krinsky–Robb (Gaussian draws around β̂);
  posterior draws from a Bayesian fit would require a different
  adapter shape and, frankly, a different library. If users want
  Bayesian inference, point them at Bambi/PyMC.

- **Conformal prediction intervals.** Different inferential paradigm.
  Could fit alongside but not on this roadmap.

- **Per-call scale switching.** See §2.

- **Sklearn coverage beyond linear models with hand-computed Σ̂.**
  Tree ensembles get bootstrap-only adapters. Anything more
  sophisticated is deferred until a real use case appears.

---

## 10. A quick mental model for each module

- `_gradients.py`: "Given h(β), return ∇h(β)." Three backends; all
  produce identical numerics for autodiff/wrapped_fd. Pure functional.

- `_delta.py`: "Given ∇h and Σ̂, compute SE/CI/Wald." Pure numerical
  kernel. Knows nothing about estimands or sessions.

- `_kappa.py`: "Given h, β̂, and Σ̂, is the delta linearization
  reliable?" Returns a dimensionless curvature score and a verdict.

- `_estimands.py`: Factory functions that build h(β) from session
  configuration and per-call arguments. The plumbing layer between
  user inputs and the gradient/inference machinery.

- `_scenarios.py`: Turns user-facing `atexog=`/`over=` into concrete
  DataFrames of evaluation rows. Handles type-aware
  aggregation rules (typical, mean, etc.).

- `_adapter.py`: Abstract interface that adapters implement. Defines
  the four shapes: GLM, LinearPrediction, WrappedFD, BootstrapOnly.

- `_inference.py`: Orchestration. Receives an estimand h and a
  configuration; runs delta/sim/bootstrap and returns a result dict.
  Auto-fallback policy lives here.

- `_result.py`: Result dataclasses with reporting, hypothesis tests,
  and arithmetic for inter-call composition.

- `margins.py`: User-facing `Margins` class. Wires user inputs to the
  internal pipeline; exposes the convenience constructors
  (`linear_scale`, `log_scale`, etc.).

- `_adapters/*.py`: Concrete implementations per framework.

---

## 11. Conventions and style

- **No private classes called `Estimand`, `Prediction`, `Slope`, etc.**
  The estimand layer is functional (factory functions returning
  callables), not class-based. This was a deliberate cleanup partway
  through the design.

- **No `compose` parameter that does multiple jobs.** Composition slots
  are kept distinct: per-prediction transforms via the estimand
  algebra (within `predict()`'s `compose=`), inter-call composition
  via `MarginsResult` arithmetic, post-inference cosmetic via
  `MarginsResult.scaled()`.

- **Type hints on all public APIs.** Internal helpers can be looser.

- **Docstrings with NumPy/SciPy-style Parameters/Returns sections.**
  Every public method should have a usable docstring.

- **No emoji, no excess formatting.** Plain prose docstrings.

---

## 12. Where to start reading

Suggested reading order:

1. This primer.
2. `_gradients.py` — small, self-contained, foundational. Read end-to-end.
3. `_delta.py` and `_kappa.py` — the numerical kernels above gradients.
4. `_estimands.py` — see how the kernels get composed.
5. `_adapter.py` — the abstract interface.
6. `_adapters/statsmodels_glm.py` — a concrete implementation (skeleton).
7. `_scenarios.py` — counterfactual construction.
8. `_result.py` — output types.
9. `_inference.py` — orchestration above kernels.
10. `margins.py` — user-facing wrapper that ties everything together.

After reading these, look at `IMPLEMENTATION_GUIDE.md` for the
prioritized list of what's a stub and what needs filling in.
