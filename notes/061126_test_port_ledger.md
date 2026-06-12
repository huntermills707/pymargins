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
