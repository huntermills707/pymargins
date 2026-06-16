# Test port ledger

Companion to `notes/061126_rewrite_implementation_guide.md`. Append-only.

## P1 — `tests/oracle/test_analytic.py` (R1)

Category: (a) reference-derived (closed-form analytic identities).

Disposition: replaced-by-refusal-test — not applicable; these are net-new
analytic oracle tests computed in-test from the fitted model with numpy/scipy.

## P2 — `tests/oracle/test_r_golden.py` (R1)

Category: (a) reference-derived (R goldens).

Disposition: re-anchored (R `marginaleffects` goldens in
`tests/oracle/golden/`). Per-case tolerances added where statsmodels and
`sandwich` covariance conventions differ (HC1, cluster, probit PDF).

## P3 — `tests/test_engine_seeds.py` / `tests/test_engine_banks.py` (R1)

Category: (c) semantic-change.

Disposition: replaced-by-refusal-test — not applicable; net-new determinism
and caching tests for the session-free bank model.

## P4 — Correction to P2: per-case tolerances are now ledgered defects (R1)

Category: (a) reference-derived (R goldens).

Disposition: re-anchored. The residual gaps P2 attributed to "statsmodels and
`sandwich` covariance conventions" are now individually ledgered:

- Probit nonrobust SE gap → D6 (observed vs expected information).
- Cluster SE gap → D8 (legacy `cluster=` silent drop, not a convention gap).
- GLM HC1 gap → D9 (statsmodels GLM omits n/(n-k); fixed in adapter).

The HC1 per-case tolerances were removed after the D9 fix; the remaining
tolerances (`probit std_error = 0.007`, `cluster std_error = 2e-5`) are
ledgered and explained in their golden notes.

## P5 — `tests/anchor/test_anchor_gcomputation.py::test_anchor_elasticity_ols` (R6)

Category: (b) implementation-difference tolerance.

Disposition: tolerated. The point estimates remain bit-exact-anchored, but
SE/CI use `rtol=1e-9` instead of the anchor matrix's usual bit-exact check.
Reason: the legacy `Margins._elasticity` computes the slope and prediction as
separate `MarginsResult`s and then applies the chain rule to their stored
gradients, while the new engine composes `slope_cq.h` and `pred_cq.h` into a
single estimand and autodiffs once. Under `jax_enable_x64` the two paths agree
to ~1e-11 (float-operation-order noise) but not bit-exact. The SE is
independently anchored by the closed-form analytic oracle
`tests/test_gcomputation_r6.py::test_eyex_se_against_analytic_delta`, so the
anchor tolerance is not self-referential.

## P6 — `tests/test_adapter_statsmodels_ordered.py` (R7)

Category: (a) mechanical translation; (c) semantic-change.

Disposition: ported-mechanical. The two end-to-end tests were translated from
`Margins.linear_scale(model, adapter=adapter)` to
`GComputation(model, adapter=adapter, method="auto")` (the tier-2
WrappedFDAdapter requires auto-resolved simulation; the new default
`method="delta"` refuses the non-differentiable estimand).

The three outcome-subsetting tests (`predict(outcome=...)`,
`dydx(..., outcome=...)`, `result.outcome(...)`) were initially dropped because
`_outcome_shape` metadata was not emitted for adapter-based queries. P7 wired
that metadata for any adapter with `n_outcomes > 1`, so these tests were
re-enabled in the R7 audit and now pass (verified against the full-result slice
at `atol=1e-12`).

## P7 — `tests/test_adapter_statsmodels_mnlogit.py` (R7)

Category: (a) mechanical translation for the adapter tests; (c) semantic-change
for the outcome-subsetting tests.

Disposition: ported-mechanical. The `Margins` session references were replaced
with `GComputation(model, adapter=adapter)`. The five outcome-subsetting tests
(`predict(outcome=...)`, `dydx(..., outcome=...)`, `result.outcome(...)`) were
kept and now pass after wiring multi-outcome metadata through the new engine:

- `_engine/_queries.py` emits `_outcome_shape` for prediction/slope queries when
  the adapter exposes `n_outcomes > 1`.
- `_result/_graphresult.py` `outcome()` now slices 1D single-atom results and
  accepts a list/tuple of outcome indices, and slices the delta gradient for
  single-atom multi-outcome results.
- `estimators/_base.py` `_query()` applies `result.outcome(spec.outcome)` when
  the user passes `outcome=` to any query method.

All 28 tests in the file pass.

## P8 — `tests/test_adapter_sklearn.py` (R7)

Category: (a) mechanical translation for the estimator noun and inference
keywords; (c) semantic-change for formula verification.

Disposition: ported-mechanical. Replaced `Margins` with `GComputation` and
updated inference keywords (`n_boot=` → `B=`, `rng_seed=` → `seed=`). All
end-to-end assertions remain unchanged.

`test_formula_verification_catches_intercept_mismatch` is category (c): the new
engine no longer calls `adapter.attach(session)`, so formula verification is no
longer triggered by estimator construction. The test now calls
`adapter._verify_formula_spec()` directly to preserve the assertion that an
intercept-mismatching formula raises.

All 8 tests in the file pass.

## P9 — `tests/test_bootstrap_ci_methods.py` (R7)

Category: (a) mechanical translation for bootstrap CI method tests; (c)
semantic-change for CI-method validation and removed features.

Disposition: ported-mechanical. Replaced `Margins` with `GComputation`, updated
inference keywords (`n_boot=` → `B=`, `rng_seed=` → `seed=`), and replaced
`bootstrap_config={"ci_method": x}` with `ci=x`. `cluster=` was moved to
`steps.input(df, cluster=...)` and passed as the wiring node with
`outcome=fit`. `Margins.log_scale(...)` became `GComputation(..., scale="log")`.

Dropped tests (category c):
- `test_basic_conf_int_recomputation`
- `test_bca_conf_int_recomputation`
- `test_studentized_conf_int_recomputation`
  Reason: `GraphResult.conf_int()` no longer accepts `level=`; the confidence
  level is locked at estimator construction (design §4.3).
- `test_bca_with_provided_acceleration`
  Reason: the BCa `acceleration=` override has no new home in the
  `GComputation`/`steps.input` surface (implementation guide Appendix C #10).

Adjusted assertions (category c):
- `test_invalid_ci_method_raises`: invalid `ci=` is now rejected by
  `compile()` with `CompileError` (subclass of `ValueError`) instead of at
  query time; the test now asserts the constructor raises matching `"ci="`.

All 10 remaining tests in the file pass.

## P10 — `tests/test_bootstrap_jit_cache.py` (R7)

Category: (a) mechanical translation for the cache/kernel smoke tests; (c)
semantic-change for engine-comparison and removed-option tests.

Disposition: ported-mechanical. Replaced `Margins` with `GComputation` and
updated inference keywords (`n_boot=` → `B=`, `rng_seed=` → `seed=`), replaced
`bootstrap_config={"ci_method": x}` with `ci=x`, and removed `diagnostics=True`
(diagnostics are always on in the new engine).

Dropped tests (category c):
- All `test_bootstrap_batched_matches_loop_*` tests.
- `test_bootstrap_ame_all_replicates_fail_falls_back`.
- `test_bootstrap_batched_matches_loop_predict_generalized`.
- `test_bootstrap_data_coupled_slope_routes_to_loop`.
- `test_bootstrap_robust_probe_discrete_free_covariate`.
Reason: these tests exercised the legacy `bootstrap_config={"engine": "loop"}`
option, which has no equivalent in the `GComputation` surface; the batched fast
path is now the only user-facing bootstrap engine. The internal loop fallback
still exists but is not selectable.

All 10 remaining tests in the file pass.

## P11 — `tests/test_formula_interface.py` (R7)

Category: (a) mechanical translation for FormulaSpec internals and the
formula-outcome constructor; (c) semantic-change for Margins-specific surface
behavior.

Disposition: ported-mechanical. Replaced `from pymargins import Margins` with
`from pymargins import GComputation, steps`. FormulaSpec and `_has_derived_terms`
unit tests are unchanged (they exercise still-used internal helpers).

Translated tests:
- `test_from_formula_classmethod` → `test_formula_outcome_constructor`:
  `Margins.from_formula(model, formula=f, data=df)` became
  `GComputation(steps.input(df), outcome=f)`; the assertion now reads
  `est._compiled.adapter._formula_spec is not None`.
- `test_dydx_array_fit_with_formula_matches_formula_fit`:
  `Margins.linear_scale(..., formula=..., data=..., at="mean")` became
  `GComputation(steps.input(df), outcome=..., at="mean")`; numeric expectations
  unchanged.
- `test_dydx_array_fit_without_formula_warns_on_derived_terms`: the array-fit
  model is now wrapped in `StatsmodelsOLSAdapter(model, training_data=df)` before
  passing to `GComputation(..., at="mean")` because the new surface's auto-
  detection requires training data for array-fit statsmodels models. The warning
  assertion remains unchanged.

Dropped tests (category c):
- `test_predict_transform_arity_validation`
- `test_evaluate_compose_arity_validation`
  Reason: the new engine calls `transform`/`compose` directly at execution time;
  arity errors surface as `TypeError`, not as an upfront `ValueError` with a
  "must accept at least" message.
- `test_joint_test_neither_gradient_nor_draws`
- `test_run_test_neither_gradient_nor_draws`
  Reason: `GraphResult` has no `materialize()` method (results are already
  materialized), and the refusal messages/no-gradient paths are covered by
  `GraphResult` unit tests elsewhere.

All 11 remaining tests in the file pass.

## P12 — `tests/test_engine_queries.py` (R7)

Category: (a) mechanical translation for the query-layer unit tests; (b)
oracle-covered / legacy-deleted for the legacy-builder comparison tests; (c)
semantic-change where the legacy reference was removed.

Disposition: ported-mechanical. Replaced `from pymargins.margins import Margins`
with `from pymargins import GComputation` and replaced the legacy-session helper
`ctx_from_margins` with `ctx_from_gcomp`, reading adapter/base_data/weights/at/
phi/phi_inv from `est._compiled` and fd_step/gradient_backend from `est._plan`.
Inference keywords were updated (`n_boot=` → `B=`, `rng_seed=` → `seed=`). The
log-scale case became `GComputation(..., scale="log")`.

Dropped tests:
- All `test_*_matches_legacy*` tests that compared `compile_query()` output to
  the deleted `pymargins.margins._estimands._build_*_estimand` functions.
- The legacy-crash assertion in `test_predict_weights_plus_over_weighted_group_means`.
- The legacy-unweighted comparison assertions in
  `test_contrasts_weights_honored_in_aggregation` and
  `test_evaluate_weights_honored_in_aggregation`.
- `test_contrasts_2d_matrix_h_matches_legacy_numbers`.
- `test_contrasts_weights_at_mean_single_row_matches_legacy`.
Reason: the `pymargins.margins` package has been deleted at R7, so the legacy
builders are no longer importable. The underlying estimand semantics are covered
by the oracle suite (`tests/oracle`) and by the kept independent-expectation
probes for D16/D17/D18.

Adjusted assertions (category c):
- `test_dydx_h_matches_legacy_multivar` was converted from a legacy bit-exact
  comparison to a smoke test verifying the multi-variable dydx estimand returns
  two finite values.

All 31 remaining tests in the file pass.

## P13 — `tests/test_transforms_pipeline.py` (R7)

Category: (c) semantic-change; partly legacy-deleted.

Disposition: **file deleted**. The file exercised the legacy `Margins` session
plumbing for the removed `transforms=[...]` surface and the deleted session bank
cache (`_bootstrap_states_bank`, `_bootstrap_bank_key`).

- The `transforms=` list is replaced by `steps.*` verbs in the new surface;
  there is no public custom-stage injection point, so the `IdentityStage`,
  `_RecordingStage`, `_RowAlteringStage`, `_NonRowAlteringStage`, and
  `_UnpicklableStage` cases cannot be mechanically translated.
- The bank-key and `_bootstrap_states_bank` assertions are about the removed
  session cache; `BankSet` ownership and determinism are covered by
  `tests/test_engine_banks.py` / `tests/test_engine_seeds.py`.
- Equivalent new-surface step behavior is covered by existing files such as
  `tests/test_transforms_filters.py`, `tests/test_transforms_reimpute.py`,
  `tests/test_transforms_guards.py`, and `tests/test_transforms_protocol.py`.

Deleted: `tests/test_transforms_pipeline.py`.

## P14 — Deleted legacy-only test files (R7)

Category: (c) semantic-change / legacy-deleted.

Disposition: **files deleted**:

- `tests/test_strict_mode.py` — tested `strict=True` constructor validation,
  which is doctrine-removed in 0.4.0.
- `tests/test_margins_remaining_gaps.py` — exercised legacy `MarginsResult`
  internals (`conf_int`, `joint_test`, `compose_results`, arithmetic,
  `materialize`, `influence`, `_phi_to_name`, etc.).
- `tests/test_margins_result_coverage.py` — exercised legacy `MarginsResult`
  internals (`to_frame`, summary, test, pairwise contrasts, outcome slicing,
  etc.).
- `tests/test_session_bank_cache.py` — exercised the removed session-level
  bootstrap/simulation cache (`_simulation_draws_cache`,
  `_harvest_bootstrap_states`, session parameter mutation, adapter drift,
  resample bank IDs, matching rematch through the cache). `BankSet`
  determinism and retention are covered by `tests/test_engine_banks.py` and
  `tests/test_engine_seeds.py`.

## P15 — `tests/golden/test_regression_goldens.py` + `tools/record_goldens.py` (R7)

Category: (b) engine-derived regression goldens.

Disposition: net-new. Records anchor-matrix cells from the new engine as
byte-exact NPZ arrays and verifies they reproduce. The anchor matrix
(`tests/anchor/test_anchor_gcomputation.py`) is retired once these goldens land.

R7 audit expansion: added cluster-bootstrap predict/dydx cells for both OLS and
GLM (`steps.input(df, cluster=cluster_ids), outcome=fit`), bringing the matrix
to 22 cells. Determinism is enforced by a fixed seed and verified byte-exact.

## P16 — Bulk mechanical port summary (R7)

Category: (a) mechanical translation.

Disposition: ported-mechanical. In addition to the individually ledgered files
above, the R7 subagent swarm mechanically translated the following test files
from `Margins` to `GComputation` (noun rename, inference keyword updates
`n_boot`/`rng_seed` → `B`/`seed`, `bootstrap_config={"ci_method": x}` → `ci=x`,
`cluster=` → `steps.input(..., cluster=...)`, and `scale="log"`/`"identity"` as
needed):

- `tests/test_052426_improvements.py`
- `tests/test_adapter_lifelines_aalen_additive.py`
- `tests/test_adapter_lifelines_coxph.py`
- `tests/test_adapter_lifelines_coxph_survival.py`
- `tests/test_adapter_lifelines_cox_timevarying.py`
- `tests/test_adapter_lifelines_coxtimevarying.py`
- `tests/test_adapter_lifelines_crc_spline_hr.py`
- `tests/test_adapter_lifelines_crc_spline.py`
- `tests/test_adapter_lifelines_generalized_gamma.py`
- `tests/test_adapter_lifelines_loglogistic_aft.py`
- `tests/test_adapter_lifelines_lognormal_aft.py`
- `tests/test_adapter_lifelines_piecewise_exponential.py`
- `tests/test_adapter_lifelines_weibull_aft.py`
- `tests/test_adapter_linearmodels_absorbing.py`
- `tests/test_adapter_linearmodels_famamacbeth.py`
- `tests/test_adapter_linearmodels_iv.py`
- `tests/test_adapter_linearmodels_ols.py`
- `tests/test_adapter_linearmodels_panel.py`
- `tests/test_adapter_statsmodels_discrete_binary.py`
- `tests/test_adapter_statsmodels_discrete_count.py`
- `tests/test_adapter_statsmodels_gee.py`
- `tests/test_adapter_statsmodels_glm.py`
- `tests/test_adapter_statsmodels_mixedlm.py`
- `tests/test_adapter_statsmodels_ols.py`
- `tests/test_adapter_statsmodels_phreg.py`
- `tests/test_adapter_statsmodels_phreg_survival.py`
- `tests/test_adapter_statsmodels_quantreg.py`
- `tests/test_adapter_statsmodels_rlm.py`
- `tests/test_adapter_statsmodels_zi.py`
- `tests/test_audit_coverage_gaps.py`
- `tests/test_bootstrap_block.py`
- `tests/test_bootstrap_cluster.py`
- `tests/test_bootstrap_parallel.py`
- `tests/test_bug_fixes.py`
- `tests/test_correctness_linearmodels.py`
- `tests/test_correctness_mixed.py`
- `tests/test_correctness_survival.py`
- `tests/test_coverage_gaps.py`
- `tests/test_end_to_end_ols.py`
- `tests/test_end_to_end.py`
- `tests/test_engine_execute.py`
- `tests/test_engine_seeds.py`
- `tests/test_graphresult.py`
- `tests/test_inference.py`
- `tests/test_influence_contract.py`
- `tests/test_matching.py`
- `tests/test_methodological_completeness.py`
- `tests/test_pool_imputations.py`
- `tests/test_progress_bar.py`
- `tests/test_result_formatting.py`
- `tests/test_result_to_frame.py`
- `tests/test_survey_bootstrap.py`
- `tests/test_survey_correctness.py`
- `tests/test_survey_linearization.py`
- `tests/test_transforms_filters.py`
- `tests/test_transforms_guards.py`
- `tests/test_transforms_reimpute.py`
- `tests/test_weighted_estimands.py`
- `tests/test_williams_2012_correctness.py`

No semantic changes were required for these files; all tests pass under the new
surface.
