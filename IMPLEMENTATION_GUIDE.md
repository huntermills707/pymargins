# pymargins — Implementation Guide

This document enumerates what's already specified versus what implementers
need to fill in, in priority order. Read `PRIMER.md` first for the
architectural philosophy.

---

## State of the scaffold

The library is end-to-end usable with statsmodels GLM and OLS/WLS/GLS.
Numerical kernels (`_gradients.py`, `_delta.py`, `_kappa.py`),
orchestration (`_inference.py`, `margins.py`), and the two registered
adapters all work; the architecture is no longer scaffolding.

What works:
- `_gradients.gradient`, `_gradients.hessian`, `_gradients.directional_derivative`,
  `_gradients.hessian_vector_product`
- `_gradients.make_predict_with_fd_jvp`, `_gradients.make_glm_jvp_wrapper`
  (covers Logit, Probit, CLogLog, LogLog, LogC, Log, Identity, Power
  including Power(0)→log, InversePower, InverseSquared, Sqrt, Cauchy,
  NegativeBinomial)
- `_delta.delta_se`, `_delta.delta_confint`, `_delta.delta_wald_test`,
  `_delta.joint_wald_test`, `combined_gradient`, `joint_covariance_of_results`
- `_kappa.kappa`, `_kappa.kappa_vector`, `_kappa.classify_kappa`,
  `_kappa.session_kappa`, `_kappa.delta_simulation_disagreement`
  (vector-estimand-aware)
- `_estimands.make_prediction_estimand`, `make_slope_estimand`
  (data-side total derivative; matches Stata/R semantics — see Resolved
  list below), `make_linear_combination_estimand`,
  `make_evaluate_estimand`, `is_jax_differentiable` (probes vmap +
  hessian to mirror engine trace patterns)
- `_inference._run_delta`, `_run_simulation`, `_run_bootstrap`
  (i.i.d. nonparametric; computes κ at β̂ when h is differentiable)
- `_result.MarginsResult.summary`, `to_frame`, `conf_int`, `test`,
  `joint_test`, `scaled`, `materialize`, composition via `__add__` /
  `__sub__` / `__mul__` with `phi`/`phi_inv` propagation
- `StatsmodelsGLMAdapter` and `StatsmodelsOLSAdapter` (formula and
  array fits, HC0–HC3 vcov, cluster vcov via refit, formula and
  array-fit refit for bootstrap)
- `_adapters._detect_adapter_class` registers GLM and OLS/WLS/GLS

What's a known stub or has known issues:
- `_inference._run_bootstrap` implements i.i.d. nonparametric
  bootstrap only. Cluster and block bootstrap (Priority 3.1) are not
  yet implemented.
- `Margins._base_data` requires adapter cooperation — adapters must
  expose `training_data` for diagnose() and scenario expansion to work.
- `MarginsResult.__sub__` and `__add__` only work for delta-method
  results (with gradients). Simulation/bootstrap composition isn't yet
  implemented; would need matched draws.
- `over=` is pandas-coupled (uses `DataFrame.groupby`); not in the
  adapter contract (B17).

Resolved since initial review:
- `_inference._run_simulation` now actually uses `jax.vmap` for the
  JAX path (the original B6 fix was incomplete due to a missing
  `import jax` in `_inference.py`; the vmap call silently fell back
  to a Python loop until that was added).
- `make_slope_estimand` rewritten as a **data-side** central
  difference: perturbs the source DataFrame's column ±ε and rebuilds
  the design through `adapter.design_matrix_from_df`, so patsy
  regenerates every interaction, polynomial, spline, and `I(...)`
  transform. `dydx(v)` now returns the *total* derivative ∂μ/∂v,
  matching Stata's `margins, dydx()` and R's
  `marginaleffects::slopes()`. The previous column-wise partial was
  silently wrong for any model with interactions or transforms.
  `column_index_of_variable` is now a type guard for `dydx()` only;
  the index it returns is unused.
- `_jax_link_inverse` corrected for `LogC` (was using the CLogLog
  formula); `Power(0)` mapped to `jnp.exp` (statsmodels' log-link
  equivalence); `NegativeBinomial.alpha` read directly without a
  defensive `getattr` default.
- `is_jax_differentiable` strengthened to probe `jax.vmap` and
  `jax.hessian` — matches the trace patterns the engine actually
  uses, including κ. Previous probe via `jax.grad(h)(beta)` traced
  with concrete inputs and missed `TracerBoolConversionError` cases.
- `_run_simulation` and `_run_bootstrap` compute κ at β̂ when h is
  JAX-differentiable; PRIMER §5.1 frames κ as the universal delta
  validity diagnostic, so it's now reported uniformly across the
  three inference paths.
- `delta_simulation_disagreement` works for vector estimands (returns
  the maximum per-component relative disagreement); previously
  hardcoded to scalar.
- `_infer_variable_type` no longer emits `"discrete"`; integer
  columns with few unique values are classified `"continuous"` (or
  `"binary"` when there are exactly 2 unique values), so `dydx()`
  works on integer-coded continuous covariates.
- Auto-fallback from delta to simulation when κ exceeds threshold;
  fallback is visible on the result via `fallback_triggered` /
  `fallback_reason`, propagates through composition.
- `MarginsResult.__mul__` and `scaled()` correctly apply
  `phi(scalar * phi_inv(estimate))` on non-identity scales (B10).
- `MarginsResult` captures `phi` and `phi_inv` at construction so
  reporting no longer depends on the session weakref (A1).
- `rng_seed` is accepted as a session-level argument and plumbed through
  `_inference_config()` (B14).
- `Σ̂` is frozen eagerly at session construction (B15).
- DataFrame `atexog` is routed through `scenario["data"]` (A6).
- The delta path skips the `is_jax_differentiable` probe for adapters
  that declare `supports_jax_autodiff=True` (B27).

---

## Priority 0 — make one end-to-end path work *(complete)*

Goal: be able to fit a logit model with statsmodels, wrap it in
`Margins.log_scale(...)`, compute a relative-risk contrast, and get a
sensible CI. This is the minimum viable proof-of-architecture.

All sub-tasks are done; tests live in `tests/test_gradients.py`,
`tests/test_delta.py`, `tests/test_adapter_statsmodels_glm.py`, and
`tests/test_end_to_end.py`. Sub-task descriptions retained below for
reference on resolved decisions.

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

### 0.3 Complete `StatsmodelsGLMAdapter` *(complete)*

Resolutions:

- **`design_matrix_from_df`**: formula-fit uses cached `design_info`
  via patsy; array-fit falls back to `df[exog_names].values`. Shared
  helper in `_adapters/_common.py`.
- **`column_index_of_variable`**: raises `ValueError` for binary or
  categorical variables. The function is now a type guard for
  `dydx()` only — its return value is unused since `dydx()` is
  computed via data-side FD through `design_matrix_from_df`, which
  handles patsy's interactions, polynomials, splines, and `I(...)`
  transforms automatically.
- **`covariance`**: HC0–HC3 supported (refit if not the original
  `cov_type`); cluster-robust supported via refit with
  `cov_kwds={"groups": ...}`; ndarray pass-through for advanced use.
- **`refit`**: handles both formula-fit and array-fit models.

### 0.4 Wire up Margins to the adapter *(complete)*

`_build_prediction_estimand` uses `adapter.design_matrix_from_df`;
`_build_slope_estimand` uses both that and `column_index_of_variable`
(as a type guard only — slopes are now data-side FD through the
formula).

### 0.5 Smoke test: relative risk via log_scale *(complete)*

See `tests/test_end_to_end.py::test_relative_risk_contrast` and
adjacent end-to-end tests.

---

## Priority 1 — fill out the adapter family *(complete for statsmodels)*

### 1.1 `StatsmodelsOLSAdapter` (LinearPredictionAdapter) *(complete)*

`pymargins/_adapters/statsmodels_ols.py`. Inherits the linear-predict
path; implements `coefficients`, `covariance` (HC0–HC3 directly,
cluster via refit), `design_matrix_from_df`, `variable_metadata`, and
`refit` (formula and array fits).

### 1.2 Auto-detection for the new adapter *(complete)*

`_detect_adapter_class` matches `RegressionResultsWrapper` for
OLS/WLS/GLS.

### 1.3 Adapter interface is settled *(no work — design statement)*

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

## What's next

Priorities 0 and 1 are done. The natural next steps, in rough order of
user-visible payoff:

1. **Priority 2** — strict mode, better error messages, attach-time
   adapter validation. Tightens the user-facing surface against the
   adapters that are now shipping.
2. **Priority 3** — cluster and block bootstrap (3.1), parallelization
   (3.2), BCa / basic / studentized CIs (3.3). The current i.i.d. path
   covers the common case but cluster bootstrap is the natural next
   need for any panel-data user.
3. **Priority 4** — additional adapters (sklearn, linearmodels, mixed
   models). Each new adapter exercises the existing interface; if the
   four-shape factoring is right, none should require core changes.
4. **Priority 5** — reporting polish, plotting, LaTeX/HTML output.

Don't worry about getting the API perfect on the first iteration. The
architecture is the asset; the API is replaceable. As long as the
internal layers stay clean (the estimand triple, the three inference
methods, the adapter shapes, the κ diagnostic), the user-facing API
can evolve based on feedback without destabilizing the rest.
