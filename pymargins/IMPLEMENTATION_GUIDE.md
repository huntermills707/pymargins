# pymargins — Implementation Guide

This document enumerates what's already specified versus what implementers
need to fill in, in priority order. Read `PRIMER.md` first for the
architectural philosophy.

---

## State of the scaffold

The library is currently a complete API specification with implementation
stubs. Every public interface is defined with type hints and docstrings.
Numerical kernels (`_gradients.py`, `_delta.py`, `_kappa.py`) are mostly
implemented and should work as-is. The orchestration and adapter layers
are partially implemented; concrete framework adapters are skeletons.

What works (or should work) end-to-end as written:
- `_gradients.gradient`, `_gradients.hessian`, `_gradients.directional_derivative`
- `_gradients.make_predict_with_fd_jvp`, `_gradients.make_glm_jvp_wrapper`
- `_delta.delta_se`, `_delta.delta_confint`, `_delta.delta_wald_test`,
  `_delta.joint_wald_test`
- `_kappa.kappa`, `_kappa.kappa_vector`, `_kappa.classify_kappa`
- `_kappa.session_kappa`, `_kappa.delta_simulation_disagreement`
- `_estimands.make_prediction_estimand`, `make_slope_estimand`,
  `make_linear_combination_estimand`, `make_evaluate_estimand`,
  `is_jax_differentiable`
- `_inference._run_delta` and `_run_simulation` (modulo the issues below)
- `_result.MarginsResult.summary`, `to_frame`, `conf_int`, `test`,
  `joint_test`, `scaled`, `materialize`

What's a known stub or has known issues:
- The dispatch in `_adapter.auto_detect_adapter` is intentionally
  delegated to `_adapters/__init__.py:_detect_adapter_class` (one
  framework currently registered).
- `_inference._run_bootstrap` is implemented for i.i.d. nonparametric
  bootstrap. It uses `h_factory` callbacks so that estimands are rebuilt
  against each resampled adapter. Cluster and block bootstrap are not yet
  implemented.
- `Margins._base_data` requires adapter cooperation — adapters must
  expose `training_data` for diagnose() and scenario expansion to work.
- `MarginsResult.__sub__` and `__add__` only work for delta-method
  results (with gradients). Simulation/bootstrap composition isn't yet
  implemented; would need matched draws.
- `StatsmodelsGLMAdapter` has a working skeleton but several methods
  need real implementation (see below).

---

## Priority 0 — make one end-to-end path work

Goal: be able to fit a logit model with statsmodels, wrap it in
`Margins.log_scale(...)`, compute a relative-risk contrast, and get a
sensible CI. This is the minimum viable proof-of-architecture.

Tasks in order:

### 0.1 Verify the gradient module against analytical truth

Write tests for `_gradients.gradient` and `_gradients.hessian`:
- Compare autodiff output against hand-derived gradients for OLS
  predictions (∂(Xβ)/∂β = X) and logit predictions (∂σ(Xβ)/∂β =
  σ(1-σ)X).
- Verify `make_glm_jvp_wrapper` produces the same gradients as a JAX
  reimplementation of the same prediction.
- Verify central-difference FD agrees with autodiff to ~10 digits.
- Verify the custom-JVP wrapper compositions correctly through
  `jax.grad` and `jax.hessian`.

Tests live in `tests/test_gradients.py`.

### 0.2 Verify the delta module against statsmodels

Write tests for `_delta`:
- Compare `delta_se` to `statsmodels.regression.linear_model.OLS.t_test`
  for known linear contrasts. Should agree to many decimals.
- Compare `delta_confint` to statsmodels' `get_prediction().summary_frame()`
  for a logit model's predicted probabilities.
- Verify `joint_wald_test` against statsmodels' `wald_test`.

Tests live in `tests/test_delta.py`.

### 0.3 Complete `StatsmodelsGLMAdapter`

The skeleton in `_adapters/statsmodels_glm.py` needs:

- **`design_matrix_from_df`**: handle non-formula fits. Currently only
  formula-fit models work (uses patsy.dmatrix with the cached
  design_info). For direct-array fits, a fallback that just returns
  the columns by name in exog_names order. Decide: refuse non-formula
  fits, or implement the fallback?

- **`column_index_of_variable`**: handle factor expansions. Currently
  has a heuristic (prefix match); needs proper handling of patsy's
  expansion of categorical variables. For categorical variables this
  should arguably raise rather than return — `dydx()` doesn't apply
  to categoricals anyway, and the index is used only by `dydx()`.

- **`covariance` for cluster-robust**: currently raises for cluster
  specifications. Decide whether to support (requires refit) or to
  document it as out-of-scope and require user to pass an explicit
  ndarray.

- **`refit`**: the current implementation only handles formula-fit
  models. Decide whether to support array-fit refit or document it as
  unsupported.

### 0.4 Wire up Margins to the adapter

In `margins.py`, `Margins._base_data` currently requires the adapter
to expose `training_data`. The `StatsmodelsGLMAdapter` already does
this, so this should work, but verify.

Also in `margins.py`: `_build_prediction_estimand` and
`_build_slope_estimand` reference `adapter.design_matrix_from_df` and
`adapter.column_index_of_variable`. Both are declared as abstract
methods in `ModelAdapter` (along with `training_data`).

### 0.5 Smoke test: relative risk via log_scale

Write a test that:
- Fits a logit model on synthetic data via statsmodels.formula.api.
- Wraps with `Margins.log_scale(fit, vcov=None)`.
- Computes a relative risk contrast.
- Verifies the result has reasonable structure (estimate close to
  expected, CI exists, κ is computed, etc.).

Don't worry about exact numerical agreement with another tool yet —
just that the pipeline runs end-to-end without errors.

---

## Priority 1 — fill out the adapter family

### 1.1 `StatsmodelsOLSAdapter` (LinearPredictionAdapter)

Should be much simpler than the GLM version. The `predict` method is
inherited (just `Xβ`); only `coefficients`, `covariance`,
`design_matrix_from_df`, and `variable_metadata` need implementing.

### 1.2 Auto-detection for the new adapter

Register `StatsmodelsOLS` in `_adapters/__init__.py:_detect_adapter_class`.

### 1.3 Adapter interface is settled

The abstract `ModelAdapter` interface declares `design_matrix_from_df(df)`,
`column_index_of_variable(v)`, and `training_data`. Scenario expansion
happens in `_scenarios.expand_scenario` which produces a DataFrame; the
adapter turns that into a JAX array. This separation is clean and should
not be changed.

---

## Priority 2 — strict mode and robustness

### 2.1 Strict mode (implemented)

`Margins.__init__` uses a `_NOT_GIVEN` sentinel. When `strict=True`, any
config argument still set to `_NOT_GIVEN` raises `ValueError`. Defaults
are applied only after the strict check passes.

### 2.2 Better error messages

When user passes a model that doesn't have a registered adapter, the
`auto_detect_adapter` error should suggest the closest registered
adapter and link to docs on writing custom adapters.

When a user passes an unknown variable in `atexog=`, the error should
print exactly which variables are unrecognized.

When κ is high and auto-fallback fires, the fallback should be visible
in the result (already wired via `fallback_triggered` and
`fallback_reason`); make sure `MarginsResult.summary()` surfaces this
clearly.

### 2.3 Validate session-adapter compatibility

`adapter.attach(session)` is called from `Margins.__init__`. The base
`ModelAdapter.attach` is a no-op, but `GLMAdapter.attach` now validates
that `phi` and `phi_inv` are approximate inverses at a test point.
Concrete adapters should extend this pattern:
- A survival adapter that doesn't support log scale should error if
  `phi=jnp.exp` is supplied.
- An adapter that doesn't support a requested vcov flavor should error
  here, not on first computation.
- Any adapter with link/scale constraints should validate them at
  attach time so misconfigurations surface immediately.

---

## Priority 3 — bootstrap improvements

i.i.d. nonparametric bootstrap is implemented. Remaining work:

### 3.1 Additional resampling strategies

- Cluster bootstrap (sample clusters, take all rows from chosen clusters)
- Block bootstrap (for time series)

Each strategy is parameterized — cluster needs cluster IDs, block needs
block size, etc. Add a `bootstrap_config=` argument to `Margins.__init__`
or `InferenceConfig` to specify.

### 3.2 Parallelization

Refit is the bottleneck. Use `joblib.Parallel` or `concurrent.futures`
to parallelize across replicates. Add a `n_jobs` parameter.

### 3.3 Alternative bootstrap CI methods

Currently the engine takes percentile CIs. Other options worth
supporting:
- BCa (bias-corrected and accelerated)
- Basic bootstrap
- Studentized bootstrap (when SE estimates are available per replicate)

---

## Priority 4 — additional adapters

In rough priority order:

### 4.1 `LinearmodelsPanelAdapter` and `LinearmodelsIVAdapter`

Use `LinearPredictionAdapter` as base. The work is in:
- Coefficient extraction (`results.params`)
- Σ̂ extraction across the various flavors linearmodels supports
  (`cov_type='kernel'`, etc.)
- Design matrix construction handling fixed-effects absorption
  correctly (decide what "prediction" means under absorbed FE)
- Refit

Panel models add a semantic question: when computing a counterfactual,
what value of the fixed effects is used? Needs design discussion.

### 4.2 `StatsmodelsMixedLMAdapter`

Use `WrappedFDAdapter`. Subject-specific predictions are tricky
(involve BLUPs); population-average predictions are easier. Decide
which to support first.

### 4.3 `SklearnLinearAdapter`

Use `LinearPredictionAdapter` plus hand-computed Σ̂. For
`LinearRegression`, Σ̂ is `σ̂² (X^T X)^-1` from residual variance and
the design's Gram matrix. For `LogisticRegression`, Σ̂ from Fisher
information. Both straightforward but require user to supply training
data (sklearn doesn't store it).

### 4.4 `SklearnTreeAdapter`

Use `BootstrapOnlyAdapter`. Just needs `refit` and `predict_for_bootstrap`.
The latter is just `model.predict(X)` for the trained model.

---

## Priority 5 — reporting and ergonomics

### 5.1 Better `summary()` formatting

The current `MarginsResult.summary()` is a serviceable string. For
real applied use, people will want:
- Tables aligned in columns
- Significance stars (configurable)
- Optional per-row p-values (currently you have to call `.test()`
  separately)

### 5.2 Plotting

A `MarginsResult.plot()` method using matplotlib for forest plots and
slope-as-function-of-covariate visualizations. Cosmetic; ship without
initially.

### 5.3 LaTeX / HTML output

`to_latex()` and `to_html()` on `MarginsResult` for inclusion in
papers and reports. Wraps `to_frame()` with appropriate formatting.

### 5.4 Helper module for common contrast patterns

A `pymargins.contrasts` (note: helper module, not the method) with
functions like `pairwise(model, var)`, `reference(model, var,
ref_level)`, `did(treatment, time, ...)` that return
(scenarios, weights) tuples for users to pass to `m.contrasts(...)`.
Keeps the named-pattern conveniences without baking them into the
core API.

---

## Priority 6 — testing and documentation

### 6.1 Test strategy

Three layers of tests:
1. **Numerical kernels** (`_gradients`, `_delta`, `_kappa`): unit tests
   against analytical truth and known statsmodels outputs.
2. **Orchestration** (`_inference`, `_estimands`, `_scenarios`):
   integration tests covering combinations of estimand × inference
   method × `at` rule.
3. **End-to-end** (Margins): smoke tests covering each convenience
   constructor and method.

Cross-validation against R's `marginaleffects` is valuable for
end-to-end correctness. Pick 5-10 canonical scenarios and verify
agreement to several decimal places.

### 6.2 Documentation site

Beyond docstrings:
- A "getting started" guide showing the Margins-session workflow.
- A "scales" guide explaining when to use linear/log/logit/lift.
- A "diagnostics" guide explaining κ and how to interpret it.
- A "writing your own adapter" guide for users with unusual models.

These are MUCH more valuable than API reference docs (which are
adequately covered by the docstrings).

---

## Open design questions (deferred from earlier work)

These were raised during design but not fully resolved. Implementers
will hit them eventually:

### Q1. `at` default — `overall` or `typical`?

The current default is `overall` (AME). `typical` is arguably
more "correct" for many applied contexts (since it doesn't average over
nonsensical individuals like 0.4-treated). But `overall` matches
default expectations from R's `marginaleffects` and from much of the
applied literature.

Stick with `overall` for now. Document `typical` prominently as
the recommended alternative.

### Q2. Multi-output models

Multinomial logit, ordered logit, multivariate regression: predictions
are vector-valued. The current architecture handles this in principle
(JAX `jacobian` instead of `grad`, vector-valued `MarginsResult`), but
several details aren't worked out:

- Output dimension labeling (class names? indices?)
- How does `over=` interact with multi-output? Cartesian product?
- Default behavior for `predict()` — return all outputs or require
  selecting one?

Defer until first implementer hits a multinomial use case.

### Q3. Posterior-draw inference for Bayesian models

If a future user wants to wrap a Bambi or PyMC fit, the natural
inference path is "use the posterior draws directly" rather than
delta or KR. This would be a fourth inference method
(`method="posterior"`) requiring a different adapter shape.

Out of scope currently. If implemented, it's a clean addition because
the engine's three-method dispatch generalizes naturally to four.

### Q4. Conformal-style CIs for prediction intervals

Different inferential paradigm. The current `predict()` returns CIs on
the *expected* prediction (Var(μ̂)), not prediction intervals
(Var(y - μ̂)). Some users want the latter. Could fit alongside but
needs design.

### Q5. Caching of expensive intermediates

In an interactive session, a user calling `m.predict(...)` then
`m.dydx(...)` with the same `at=` is paying for design-matrix
construction twice. An LRU cache on `expand_scenario` keyed on the
scenario tuple would help. Not critical but easy win.

---

## A reasonable first sprint

Given all the above, a one-week implementer sprint to get to a
demonstrable working version:

- **Day 1**: Read `PRIMER.md`, run through the existing modules to
  understand the structure. Set up tests/ directory and CI.
- **Day 2**: Implement priorities 0.1 and 0.2 — gradient and delta
  module tests against statsmodels. Fix any kernel bugs surfaced.
- **Day 3**: Complete `StatsmodelsGLMAdapter` (priority 0.3) including
  `design_matrix_from_df`, `column_index_of_variable`, vcov flavors.
- **Day 4**: Wire end-to-end (priority 0.4 + 0.5). Smoke test with a
  real fit. Cross-check outputs against `marginaleffects` in R for
  one or two cases.
- **Day 5**: Implement `StatsmodelsOLSAdapter` (priority 1.1). Document
  what works and ship a 0.0.1 alpha.

After that, prioritize based on whose feedback is most valuable —
adding more model coverage (priority 4) versus polishing reporting
(priority 5) versus filling in bootstrap (priority 3).

Don't worry about getting the API perfect on the first iteration. The
architecture is the asset; the API is replaceable. As long as the
internal layers stay clean (the estimand triple, the three inference
methods, the adapter shapes, the κ diagnostic), the user-facing API
can evolve based on feedback without destabilizing the rest.
