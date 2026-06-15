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
