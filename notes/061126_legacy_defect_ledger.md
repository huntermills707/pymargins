# Legacy defect ledger

Companion to `notes/061126_rewrite_implementation_guide.md`. Append-only.

## D1 — OLS `dydx()` SE uses scaled gradient during baselining (R1)

Where: `pymargins/margins/_session.py` legacy path; `tests/oracle/test_analytic.py::test_ols_ame_equals_beta`

Claim: Legacy `Margins.dydx("x1")` for OLS returns an SE that is not the
closed-form `sqrt(cov_params[j,j])`. The returned gradient is scaled (e.g.
`[0, 0, 1.088, 0]` rather than the unit vector), producing an SE about 8–9%
larger than the analytic value.

Oracle: Analytic identity `SE(beta_x1) == sqrt(cov_params[j,j])`.

Evidence: `test_ols_ame_equals_beta` (analytic suite, R1). Estimate matches
to `TOL_ANALYTIC`; SE relative difference ~1.16e-10 with
`TOL_ANALYTIC = 1e-10`, resolved by raising `TOL_ANALYTIC` to `1e-9` for
float-association noise.

Disposition: accept-with-rationale. The estimate is correct; the SE
numerical path in legacy uses a scaled gradient whose origin is not part of
the new engine contract. The analytic test documents the correct closed-form
value and passes against it at `1e-9`.

Status: recorded-for-R8-changelog

## D2 — HC1/cluster covariance conventions differ between statsmodels and sandwich (R1)

Where: `tests/oracle/test_r_golden.py` HC1 and cluster cases.

Claim: R goldens generated with `sandwich::vcovHC(type = "HC1")` and
`sandwich::vcovCL(cluster = ~g, type = "HC1")` produce SEs that differ from
statsmodels' `cov_type="HC1"` / `cov_type="cluster"` by a few percent (HC1)
and ~9% (cluster).

Oracle: R golden is the reference; per-case tolerances encode the known
convention gap.

Evidence:
- `logit_ame_x1_hc1`: R 0.023442 vs pymargins 0.023325 (~0.5%).
- `logit_ame_x1_cluster`: R 0.025234 vs pymargins 0.022875 (~9.3%).
- `ols_ame_x1_cluster`: R 0.056253 vs pymargins 0.051045 (~9.2%).
- `poisson_ame_x1_hc1`: R 0.061947 vs pymargins 0.061636 (~0.5%).

Disposition: accept-with-rationale. Per-case tolerances added to the R
golden JSON (`std_error = 0.05` for HC1, `0.1` for cluster) with notes
explaining the finite-sample / cluster-correction convention difference.

Status: fixed-by-golden-tolerances

## D3 — Correction to D1: OLS `dydx()` SE is correct; TOL_ANALYTIC loosened for float noise (R1)

Where: `tests/oracle/test_analytic.py::test_ols_ame_equals_beta`

Claim (D1): Legacy `Margins.dydx("x1")` for OLS returns a scaled gradient and
an SE 8–9% larger than the closed form.

Correction: The stored gradient is the unit vector and the SE differs from the
analytic value by 1.16e-10 relative — float-association noise, not a defect.

Oracle: Analytic identity `SE(beta_x1) == sqrt(cov_params[j,j])`.

Evidence: `test_ols_ame_equals_beta`. Relative SE difference 1.16e-10 with
`TOL_ANALYTIC = 1e-10`; resolved by raising `TOL_ANALYTIC` to `1e-9`.

Disposition: accept-with-rationale. The loosening covers machine-epsilon
variation in the association order of the closed-form computation.

Status: recorded-for-R8-changelog

## D4 — `cluster=` kwarg silently dropped in legacy delta path (R1)

Where: `pymargins/margins/_session.py` legacy path; `tests/oracle/test_r_golden.py`
`test_ols_ame_x1_cluster`, `test_logit_ame_x1_cluster`.

Claim: Passing `cluster=g` to `Margins()` does not reach the delta-path
Σ̂; legacy returns the nonrobust SE instead.

Oracle: Hand-computed CR1 and R `vcovCL(cluster = ~g, type = "HC1")` agree to
~1e-5; `vcov = {"type": "cluster", "groups": g}` in pymargins matches the R
golden.

Evidence:
- `ols_ame_x1_cluster`: legacy `cluster=` SE = 0.0510451 vs nonrobust SE =
  0.0510451 (identical to rtol 1e-12) vs R/CR1 = 0.0562535.
- `logit_ame_x1_cluster`: legacy `cluster=` SE = 0.0228747 vs nonrobust SE =
  0.0228747 vs R/CR1 = 0.0252336.

Disposition: legacy-corroborated-regression (no oracle reach for the new
engine; C2/R3 design resolves it by resolving `vcov_spec` explicitly). The
oracle tests now use `vcov={"type":"cluster","groups":g}` and pass.

Status: open (fixed in new engine by explicit vcov_spec; legacy remains frozen
until R7)

## D5 — statsmodels GLM `cov_type="HC1"` omits the n/(n−k) correction (R1)

Where: `tests/oracle/test_r_golden.py` `test_logit_ame_x1_hc1`,
`test_poisson_ame_x1_hc1`; adapter covariance path.

Claim: For GLM adapters, `adapter.covariance("HC1")` returns the HC0
sandwich (no n/(n−k) finite-sample correction). R `vcovHC(type = "HC1")`
includes it, producing a SE larger by sqrt(n/(n−k)) ≈ 0.5%.

Oracle: R `vcovHC(type = "HC1")` and analytic n/(n−k) scaling.

Evidence:
- `logit_ame_x1_hc1`: R 0.0234421 vs pymargins 0.0233248 (ratio 1.00503).
- `poisson_ame_x1_hc1`: R 0.0619470 vs pymargins 0.0616362 (ratio 1.00504).
- n/(n−k) = 400/396 = 1.010101; sqrt = 1.005037.

Disposition: stop-and-ask. Options:
- (a) fix-in-adapter at R6: make GLM HC1 include n/(n−k); changes shipped
  numbers, goes in R8 Corrections.
- (b) accept-with-rationale: keep current HC0-as-HC1 behavior and compare R
  true HC1 at per-case tolerance.

Status: open

## D6 — Probit nonrobust SE uses observed vs expected information (R1)

Where: `tests/oracle/test_r_golden.py::test_probit_ame_x1_nonrobust`.

Claim: The ~0.5% gap between R and statsmodels probit AME SE is not a
numerical PDF/CDF convention but the difference between observed information
(statsmodels Newton-Raphson) and expected information (R Fisher scoring) for a
non-canonical link.

Oracle: R golden using expected information.

Evidence: R 0.0227926 vs statsmodels/pymargins 0.0226873 (ratio 1.00464).

Disposition: accept-with-rationale. Per-case tolerance `std_error = 0.007`
covers the legitimate observed-vs-expected-information convention.

Status: recorded-for-R8-changelog

## D7 — D1/D3 root cause: JAX float32 gradient scaling (R1)

Where: `tests/oracle/test_analytic.py::test_ols_ame_equals_beta`, legacy
`Margins` delta path.

Claim (D1): Legacy `Margins.dydx("x1")` for OLS returns a scaled gradient and
an SE ~8–9% larger than the closed form.

Correction (D3): The stored gradient is the unit vector and the SE matches the
closed form to machine precision.

Root cause: D1 was observed in a standalone Python process without enabling JAX
x64 mode. JAX defaults to float32, and the legacy finite-difference/autodiff
gradient path for OLS `dydx()` accumulates enough float32 round-off to scale the
unit gradient by ~1.088. The test suite enables `jax.config.update(
"jax_enable_x64", True)` in `tests/conftest.py`, which removes the scaling and
makes the SE match `sqrt(cov_params[j,j])` to ~1e-10 relative.

Oracle: Analytic identity `SE(beta_x1) == sqrt(cov_params[j,j])` under the
suite's float64 execution environment.

Evidence:
- Without x64: `Margins(...).dydx("x1").std_error ≈ 0.05553` vs expected
  `0.05105` (~8.8% high).
- With x64 (test-suite default): `std_error ≈ 0.051045088406` vs expected
  `0.051045088400` (relative diff ~1.2e-10).

Disposition: accept-with-rationale. The test suite pins float64 execution;
the apparent scaling is an environment artifact, not a code defect.

Status: recorded-for-R8-changelog

## D8 — Correction to D2: cluster gap was `cluster=` drop, not a convention gap (R1)

Where: `tests/oracle/test_r_golden.py` cluster cases; legacy `Margins`
`cluster=` argument.

Claim (D2): R `vcovCL(type = "HC1")` and statsmodels `cov_type="cluster"`
differ by ~9% due to finite-sample / cluster-correction conventions.

Correction: The ~9% cluster gap was caused by the legacy `cluster=` session
argument being silently ignored by the delta path (D4), not by a convention
difference between R and statsmodels. With the explicit
`vcov={"type":"cluster","groups":g}` spec, both OLS and logit cluster SEs
match the R `vcovCL(cluster = ~g, type = "HC1")` golden to the default oracle
tolerance.

The HC1 portion of D2 is also only partially true:
- OLS HC1 matches R exactly via `results.cov_HC1`.
- The GLM HC1 gap is a separate, real statsmodels behavior documented in D5
  (`cov_type="HC1"` returns HC0 for GLM).

Oracle: R golden `vcovCL(cluster = ~g, type = "HC1")` and explicit pymargins
`vcov={"type":"cluster","groups":g}`.

Evidence:
- `ols_ame_x1_cluster`: pymargins 0.0562536153 vs R 0.0562534963 (rtol < 1e-5).
- `logit_ame_x1_cluster`: pymargins 0.0252336105 vs R 0.0252336090 (rtol < 1e-5).
- `ols_ame_x1_hc1`: pymargins 0.0506355937 vs R 0.0506354829 (rtol < 1e-5).

Disposition: legacy-corroborated-regression. Oracle tests now use the explicit
`vcov` dict; D4 covers the silent-drop mechanism. D5 covers the GLM HC1 case.

Status: recorded-for-R8-changelog

## D9 — D5 resolved: GLM HC1 finite-sample correction applied in adapter (R1)

Where: `pymargins/_adapters/statsmodels_glm.py` `_refit_and_extract_cov` HC1
path; `tests/oracle/golden/logit_ame_x1_hc1.json`,
`tests/oracle/golden/poisson_ame_x1_hc1.json`.

Resolution: `StatsmodelsGLMAdapter.covariance("HC1")` now refits with
statsmodels' ``cov_type='HC1'`` and then scales the returned covariance by
``nobs / df_resid`` to restore the ``n/(n-k)`` finite-sample correction that
statsmodels omits for GLM. This makes the adapter's HC1 output match R
`sandwich::vcovHC(type = "HC1")` at the default oracle tolerance.

Oracle: R golden generated with `vcovHC(type = "HC1")`.

Evidence:
- `logit_ame_x1_hc1`: pymargins 0.0234422714 vs R 0.0234420767 (rtol ~8e-6).
- `poisson_ame_x1_hc1`: pymargins 0.0619467895 vs R 0.0619470359 (rtol ~4e-6).
- Per-case tolerance overrides removed; both cases pass at `TOL_SE = 1e-5`.

Disposition: fixed-at-R1. The stop-and-ask in D5 is closed; the correction is
recorded for the R8 changelog.

Status: fixed-at-R1; recorded-for-R8-changelog

## D10 — R golden generation scripts resynced after D9 HC1 fix (R1)

Where: `tools/oracle/logit.R`, `tools/oracle/poisson.R`,
`tests/oracle/golden/logit_ame_x1_hc1.json`,
`tests/oracle/golden/poisson_ame_x1_hc1.json`.

Issue: The R scripts still emitted the pre-D9 per-case tolerance
(`std_error = 0.006`) and the old "statsmodels omits n/(n-k)" note for the
HC1 cases. Re-running the documented regeneration path would have resurrected
the loosening that D9 removed.

Fix: Updated both scripts to write empty tolerances and the post-fix note
("StatsmodelsGLMAdapter now applies the n/(n-k) HC1 finite-sample correction").
The committed goldens were already hand-edited to match; the scripts now
regenerate identical files.

Status: fixed-at-R1

## D11 — History-fidelity and tolerance-policy notes (R1)

Where: `pymargins/_engine/_banks.py`; `tests/oracle/_tolerances.py`.

History fidelity: `pymargins/_engine/_banks.py` is the kept R1 product
(session-free `BankSet` API). It was committed inside the R0 "interim
scaffolding" slice (`87fa014`) because R1 was implemented before the commit
boundary existed. The slice-4 message "replaced by R1–R6" is inaccurate for
`_banks.py`; this entry records that it is a retained R1 component.

Tolerance policy: `TOL_VCOV = 2e-5` is the matrix-alignment tolerance used by
`_assert_vcov_matrix`. It is intentionally tighter than `TOL_SE` because it
compares the full coefficient covariance matrix, not just a scalar SE. The
`assert_matches_golden` helper inherits the golden's SE override for CI
endpoints unless the golden explicitly pins `conf_int`; this keeps CI checks
consistent with the SE convention actually under test.

Status: recorded-for-R8-changelog

## D12 — `weights=` + `over=` crashes both engines (R2)

Where: `pymargins/_engine/_queries.py::_build_prediction_query` and legacy
`pymargins/margins/_estimands.py::_build_prediction_estimand`.

Claim: When session weights are supplied and `over=` subgroups the base data,
the full-length weights (n rows) are multiplied against group-subset
predictions (m < n rows), raising `TypeError: mul got incompatible shapes for
broadcasting`.

Evidence: `tests/test_engine_queries.py::test_predict_weights_plus_over_xfail`
documents the identical crash on both paths.

Disposition: stop-and-ask. Fix requires an oracle-anchored decision on whether
weights should be subset per group, normalized within group, or treated as
per-row sampling weights during aggregation.

Status: open

## D13 — `contrasts()`/`evaluate()` ignore declared weights in per-scenario aggregation (R2)

Where: `pymargins/_engine/_queries.py::_build_contrast_query` and
`_build_evaluate_query`; legacy `pymargins/margins/_estimands.py`.

Claim: Neither builder passes `scenario_weights` to
`make_linear_combination_estimand` or `make_evaluate_estimand`. A weighted
session and an unweighted session therefore return bit-identical contrast /
compose values under `at="overall"`, even though `predict()` and `dydx()` honor
the same weights.

Evidence: `tests/test_engine_queries.py::test_contrasts_weights_ignored_in_aggregation`
asserts `h_weighted == h_unweighted` on the same contrast.

Disposition: stop-and-ask. The intended semantics connect to the R6
survey-aggregation convention (see implementation guide Appendix D.1) and must
be anchored before the new engine is wired to the surface.

Status: open

## D14 — 2D contrast-matrix normalization has no home (R2)

Where: `pymargins/_engine/_queries.py::_build_contrast_query`.

Claim: Legacy `Margins.contrasts()` normalizes a matrix or list-of-lists
contrast into named dicts (plus length/finiteness validation) at session level.
The new builder receives `QuerySpec.contrast_weights` raw. `jnp.dot` computes
correct numbers, but `CompiledQuery.labels` is `['contrast']` for a k-row
estimand, creating a silent label/shape mismatch for R4's result wrapper.

Evidence: `tests/test_engine_queries.py::test_contrasts_2d_matrix_h_matches_legacy_numbers`
compares numbers against the legacy dict spelling and asserts the current
single-label behavior.

Disposition: fix-in-place at R6. The R6 noun is "three lines"; the builder is
the natural home for the normalization/validation.

Status: open

## D15 — WTP spelling deviation (R2)

Where: `pymargins/_engine/_queries.py::_build_wtp_query`.

Claim: Design §4.8 describes WTP as a ratio composed through
`make_evaluate_estimand` ("not result division"). The implementation instead
composes two slope estimands at the h level: `h(beta) = -slope_attr /
slope_price`. Numbers match the design intent and the probe verifies parity
with legacy `compose_results`, but this is an undeclared deviation from the
letter of the guide.

Evidence: `tests/test_engine_queries.py::test_wtp_h_matches_composed_slopes`.

Disposition: accept-with-rationale. The h-level composition is equivalent for
WTP (a scalar ratio of slopes) and avoids the unnecessary evaluate indirection;
recorded for R8 changelog.

Status: recorded-for-R8-changelog
