# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] — 2026-06-18 — BREAKING: the `Margins` session is removed

### Removed

- **`Margins` and `MarginsResult`** — the session/result API from 0.3.x and
  earlier is deleted. The facade/shim architecture that kept `Margins` alive
  while `GComputation` grew underneath it was the source of the silent-wrong
  bugs listed under *Corrections*; 0.4.0 removes it rather than patching it.
- **`from_posterior`** — constructing estimators from MCMC posterior draws was
  tied to the legacy session. It may re-enter as its own small front-end later
  (design §3.9), but there is no equivalent in 0.4.0.
- **`strict=True`** — the constructor now *is* strict mode: unknown kwargs are
  a `TypeError`, and every analysis-defining parameter is bound once at
  construction.
- **`diagnostics=False`** — diagnostics are always on and routed by severity
  (refuse / warn / note) into the `CompileReport` and result metadata.
- **`bootstrap_config={...}`** — the dict is dissolved into explicit constructor
  and `steps.input(...)` parameters (`ci=`, `B=`, `seed=`, `block_type=`).
- **Result post-hoc algebra** — `MarginsResult` operators (`+`, `-`, `*`, `/`),
  `compose_results()`, and `materialize()` are gone. `scaled()` and within-result
  `contrast()` / `pairwise_contrasts()` remain as readouts of a single result.
- **`MarginsResult.conf_int(level=...)`** — the confidence level is locked at
  estimator construction. Calling `result.conf_int(level=...)` now raises
  `TypeError` with a steer to re-declare the estimator.

### Migration

Every shipped 0.3.x spelling has a new home:

| 0.3.x | 0.4.0 |
|---|---|
| `Margins(model)` | `GComputation(model)` (implicit input) |
| `phi=f, phi_inv=g` | `scale=(f, g)` or a named scale |
| `Margins.log_scale(m)` | `GComputation(m, scale="log")` |
| `vcov=` | unchanged; `ndarray` forces tier-2 inference |
| `weights=` | unchanged (known weights only) |
| `at=`, `level=`, `method=` | same names, constructor-bound |
| `kappa_threshold=x` | choose `method="delta"` or `method="simulation"` explicitly; user-tunable constants overrides are compile-level only in 0.4.0 |
| `rng_seed=` / `n_sim=` / `n_boot=` | `seed=` / `n_sim=` / `B=` |
| `n_jobs=` / `progress_bar=` | execution knobs, not plan fields |
| `gradient_backend=` / `fd_step=` | engine options, now in the Plan |
| `cluster=` / `block_size=` | `steps.input(df, cluster=...)` / `steps.input(df, block=...)` |
| `survey_design=` | `steps.input(df, design=...)` |
| `matching=` | `steps.match(node, matcher)` |
| `transforms=` | `steps.trim` / `steps.drop_outliers` / `steps.reimpute` chain |
| `formula=`/`data=` (`from_formula`) | spec form of `outcome=` on a wiring node |
| `Margins(model).predict(...)` | `GComputation(model).predict(...)` |
| `Margins(model).dydx(...)` | `GComputation(model).dydx(...)` |
| `Margins(model).contrasts(...)` | `GComputation(model).contrasts(...)` |
| `Margins(model).evaluate(...)` | `GComputation(model).evaluate(...)` |
| `Margins(model).rmst(...)` | `GComputation(model).rmst(...)` |
| `session.diagnose()` | `est.plan.describe()` + `CompileReport` |
| `pool_imputations(results, ...)` | unchanged (now returns a `GraphResult`) |
| `adjust(result, ...)` | unchanged (accepts `GraphResult`) |

See the updated tutorials and how-to guides for worked examples.

### Corrections

These are numbers that changed on purpose — 0.4.0 ships the oracle-correct
value rather than the 0.3.x value. Each entry cites the oracle evidence in
`tests/oracle/` or `tests/test_analytic.py`.

- **D4/D8 — cluster covariance no longer silently dropped.** Passing
  `cluster=g` to the legacy session did not reach the delta-path covariance
  estimator; 0.3.x returned the non-robust SE. In 0.4.0 cluster/block/design
  declarations live on `steps.input(...)` and route to both the analytic
  covariance and the resampler. Oracle: R `sandwich::vcovCL(cluster = ~g,
  type = "HC1")`; explicit `vcov={"type": "cluster", "groups": g}` now matches.
- **D9 — GLM HC1 finite-sample correction restored.** `StatsmodelsGLMAdapter`
  now scales the HC0 sandwich by `nobs / df_resid` so that `vcov="HC1"` matches
  R `sandwich::vcovHC(type = "HC1")`. Oracle:
  `tests/oracle/golden/logit_ame_x1_hc1.json` and
  `tests/oracle/golden/poisson_ame_x1_hc1.json`.
- **D16 — `weights=` + `over=` computes per-group weighted means.** In 0.3.x
  this combination raised a shape error. In 0.4.0 each `over=` group uses only
  its positional subset of `weights=`, producing a weighted group mean. Oracle:
  `marginaleffects::avg_predictions(by = ..., wts = ...)`; test:
  `tests/test_engine_queries.py::test_predict_weights_plus_over_weighted_group_means`.
- **D17 — `contrasts()` and `evaluate()` honor declared weights.** In 0.3.x
  per-scenario aggregation ignored `weights=`, so weighted and unweighted
  sessions returned identical contrast values. In 0.4.0 scenario-level
  predictions are weighted before the linear combination / compose. Oracle:
  `marginaleffects::avg_comparisons(wts = ...)`; tests:
  `tests/test_engine_queries.py::test_contrasts_weights_honored_in_aggregation`
  and `test_evaluate_weights_honored_in_aggregation`.
- **D19 — `GraphResult.influence()` includes the bread matrix.** The per-query
  influence function is now `ψ^h = ∇h · Σ̂ · score_obs`, consistent with the
  influence-variance identity. Test:
  `tests/test_psi_h_includes_bread_scale_equivariance`.

Other ledgered items (D1/D3/D6/D7/D10/D11/D15/D18/D20/D21) are fixes,
label-only changes, R-script resyncs, golden regenerations, environment-fidelity
notes, or implementation deviations with no user-visible numeric change.
D2/D5/D12/D13/D14 are precursor/superseded entries whose resolutions are cited
above.

### Reproducibility

- Same-seed simulation and bootstrap draw streams may differ from 0.3.0.
  Seed-tree ownership moved from the legacy session into the engine (`BankSet`
  keyed by `(plan_hash, branch_id, seed)`), and the M=1 derivation was reviewed
  at R1. Determinism is enforced by the regression golden suite in
  `tests/golden/`.

### Added

- **`GComputation`** — the new estimator noun. It compiles a wiring graph (or
  an implicit input from a hand-fit model) into an immutable `Plan` and runs
  queries through the real engine.
- **`steps.*` wiring verbs** — `input`, `match`, `trim`, `drop_outliers`,
  `reimpute`. Each stage is a pure, content-addressed node; dependence
  declarations (`design=`, `cluster=`, `block=`) live on `steps.input`.
- **`Plan`** — immutable, hashable analysis descriptor. `plan.hash` is printed
  on every result summary for pre-registration semantics; `plan.describe()`
  reports the resolved method and any auto-resolution reason.
- **`GraphResult`** — self-contained result object. Stores estimates, standard
  errors, confidence intervals, per-method payload (gradient + Σ̂, or draws),
  and `ψ^h` when available. `conf_int()` accepts corrections
  (`bonferroni`, `sidak`, `sup-t`) but no `level=`.
- **Oracle validation suite** — analytic closed-form tests plus R
  `marginaleffects` / `survey` / `sandwich` goldens under `tests/oracle/`.
- **Regression golden suite** — layer-4 checks in `tests/golden/` recorded from
  the validated new engine, compared within the documented oracle tolerances so
  they hold across the Python/numpy/BLAS matrix.
- **Session-free banks** — `BankSet` owns resample indices, refit states, and
  simulation draws per `(plan, branch, seed)`; replay makes results from one
  estimator jointly composable.
- **Doctrine dispatch** — `execute_query` calls the delta/simulation/bootstrap
  kernels directly; there are no runtime fallbacks, κ-flips, or silent
  reroutes.
- **Soundness layer** — `CompileReport` with severity-routed predicates for
  method/CI compatibility, tail counts, cluster counts, lonely PSU, and ESS.
- **Tier-1 influence exposure** — `ModelAdapter.influence()` returns the
  per-observation influence function `ψ^β` where `score_obs()` is available;
  the engine computes `ψ^h = ψ^β @ ∇hᵀ`.

[Unreleased]: https://github.com/huntermills707/pymargins/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/huntermills707/pymargins/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/huntermills707/pymargins/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/huntermills707/pymargins/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/huntermills707/pymargins/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/huntermills707/pymargins/releases/tag/v0.1.1
[0.1.0]: https://github.com/huntermills707/pymargins/releases/tag/v0.1.0
[0.0.1]: https://github.com/huntermills707/pymargins/releases/tag/v0.0.1
