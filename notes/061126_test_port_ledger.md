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
