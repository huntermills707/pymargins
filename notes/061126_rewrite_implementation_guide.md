# 0.4.0 clean-break rewrite — implementation guide

2026-06-11 · companion to
[`061126_computation_graph_rewrite_plan.md`](061126_computation_graph_rewrite_plan.md)
(**rev. 2**). Written for the implementing model (Sonnet-class).

## How to use this guide

**Authority chain:** plan (rev. 2) → 061026 plan (Phases 3–5 only) →
requirements → design. This guide sits **below the plan**: it tells you
*how*, with exact signatures, file paths, test names, and pitfalls. If
this guide conflicts with the plan, **the plan wins — stop and report the
conflict**. If something is specified nowhere, **stop and ask; do not
improvise statistics** (I4). Decisions this guide pins that the plan left
open are listed in Appendix C so they are visible and overridable.

**Work loop per workstream:** read the listed inputs → implement →
run the workstream's tests → run the full gate (`pytest -m "not slow"`
+ `ruff check .`) → commit with the prescribed message → report status
honestly (a red test is reported red; "done" means the gate passed).

**Prime directives** (operational form of the plan's invariants):

1. **I1′ — oracle anchor.** A number is correct when the analytic suite or
   an R golden says so. Agreement with legacy `Margins` is corroboration,
   not proof; disagreement with legacy is *arbitrated* (§R1.8), never
   auto-blamed on your code — but never auto-excused either.
2. **I2 — tier-2 no-regress.** FD-wrapped adapters (lifelines,
   linearmodels, MixedLM) keep working through the new engine unchanged.
3. **I3″ — keep is earned.** Before a legacy module becomes a permanent
   dependency of the new engine, read it, trace its formulas to citations,
   and confirm the oracle suite covers the paths you use. Findings go in
   the defect ledger. Do **not** rewrite code that passes review.
4. **I4 — never invent methodology.** Every formula you write traces to a
   design-note citation or to reviewed shipped code being moved.
5. **I5 — severity texts are spec.** Refusal/warn/note texts come verbatim
   from design §6 (the tables in §6.1–§6.7). Steers at unshipped machinery
   carry *(future)*.
6. **I6 — no delegation to legacy.** Files under `pymargins/_engine/`,
   `pymargins/estimators/`, `pymargins/_result/_graphresult.py`,
   `pymargins/_result/_intervals.py` never import
   `pymargins.margins`, `pymargins._result._margins`, or call
   `pymargins._inference._dispatch.run_inference`. Legacy is imported by
   *test files only*, until R7 deletes it.

**Forbidden moves** (each one happened or nearly happened before):

- Loosening a tolerance, adding `atol`, or wrapping a comparison in
  `try/except` to make a test pass. Tolerances change only with a ledger
  entry.
- Regenerating any golden or expected value **from the new engine** to fix
  a failing test (self-anchoring). Expectations come from oracles.
- Silently dropping a kwarg, a node param, or a dict key you don't
  recognize. Unknown ⇒ raise. (The facade's silent survey-design drop was
  the worst audit finding.)
- `**kwargs` on any new public constructor or function.
- Editing anything under `pymargins/margins/`, `pymargins/_result/_margins.py`,
  or `pymargins/_inference/_dispatch.py` before R7 (they are the frozen
  corroboration oracle; R7 deletes them).
- Calling `run_inference` (`_inference/_dispatch.py`) from new-engine code
  — it contains the fallback policy the doctrine forbids.
- Claiming a workstream done in a commit message or CHANGELOG when its
  acceptance gate has not run green (this happened; it was audit finding
  #13).
- Marking a test `skip`/`xfail` to get a gate green. A genuine blocker is
  a stop-and-ask.

---

## G1. The codebase as it stands (orientation)

### G1.1 Module inventory

| Path | Lines | Role | Fate |
|---|---|---|---|
| `pymargins/steps/__init__.py` | 235 | wiring verbs (`input`, `match`, `trim`, `drop_outliers`, `reimpute`; `impute`/`imputed`/`propensity` raise NotImplementedError) | keep, evolves |
| `pymargins/_graph/_node.py` | 172 | frozen content-addressed `Node`, `_fingerprint` | keep |
| `pymargins/_graph/_plan.py` | 133 | `Plan` dataclass, hash recipe 1 | keep, extended R5 |
| `pymargins/_graph/_compile.py` | 241 | C1/C2 — interim, has known flaws (§R5 pitfalls) | rebuilt R5 |
| `pymargins/_soundness/_constants.py` | 170 | §6.7 constants with citation docstrings | keep |
| `pymargins/_soundness/_predicates.py` | 191 | `Severity`, `CompileError`, `SoundnessWarning`, `CompileReport`, 5 predicates | keep, grows R5 |
| `pymargins/_engine/_seeds.py` | 73 | `legacy_resample_indices`, `legacy_sim_draws`, `seed_sequence_for_branch` | reviewed R1 |
| `pymargins/_engine/_banks.py` | 41 | `BankSet` (passive dataclass), `BankRetentionError`, `RetentionPolicy` | rebuilt R1 (`RetentionPolicy` deleted) |
| `pymargins/estimators/_base.py` | 249 | **facade** — `GComputation` delegating to `Margins` via `_extract_legacy_kwargs` | replaced R6 |
| `pymargins/_result/_graphresult.py` | 160 | **wrapper** around `MarginsResult` | replaced R4 |
| `pymargins/margins/_session.py` | 1564 | legacy session (`Margins`), `from_posterior:720` | frozen → deleted R7 |
| `pymargins/margins/_inference_glue.py` | 378 | banks (`_bootstrap_resample_bank`, `_simulation_draws_bank`), `_inference_config`, `_frozen_cov`, `_wrap_result`, `_check_adapter_drift` | frozen → deleted R7; semantics absorbed R2/R3 |
| `pymargins/margins/_estimands.py` | 369 | session-level query builders `_build_{prediction,slope,contrast,evaluate}_estimand`, `_scenario_adapter`, `_get_base_data`, `_bootstrap_weights_for_adapter` | frozen → deleted R7; absorbed R2 |
| `pymargins/margins/_atoms.py` | 182 | `_enumerate_groups`, `_format_atom_label`, `_finalize_atoms`, `_slice_by_outcome` | frozen → deleted R7; absorbed R2 |
| `pymargins/_result/_margins.py` | 2435 | `MarginsResult`: `conf_int:671` (incl. sup-t), `test:903`, `joint_test:1025`, `influence:1648`, formatting | frozen → deleted R7; interval math moved R4 |
| `pymargins/_inference/_delta.py` | 107 | `_run_delta` kernel (**contains κ-flip at :36–53** — neutralized by `kappa_threshold=inf`, deleted R7) | keep (reviewed R3) |
| `pymargins/_inference/_simulation.py` | 134 | `_run_simulation`, `_generate_simulation_draws:13` | keep (reviewed R1/R3) |
| `pymargins/_inference/_bootstrap.py` | 1303 | `_generate_resample_indices:44`, `_harvest_bootstrap_states:685`, `_run_bootstrap:916` | keep (reviewed R1/R3) |
| `pymargins/_inference/_linearization.py` | 107 | `linearization_meat`, `linearization_cov` (survey Taylor) | keep (reviewed R3) |
| `pymargins/_inference/_dispatch.py` | 193 | `run_inference` (fallback dispatch, **never call**), `run_test` | fallback deleted R7; `run_test` math reviewed R4 |
| `pymargins/_inference/_config.py` | 129 | `InferenceConfig` dataclass | keep |
| `pymargins/_kappa.py` | 583 | `kappa`, `kappa_vector`, `session_kappa:283`, `delta_simulation_disagreement` | keep |
| `pymargins/_estimands.py` (root) | — | **atoms**: `make_prediction_estimand:91`, `make_slope_estimand:193`, `make_linear_combination_estimand:324`, `make_evaluate_estimand:473`, `is_jax_differentiable:581` | keep (citation-reviewed R2) |
| `pymargins/_scenarios.py`, `pymargins/scenarios` | — | `expand_scenario`, `make_aggregation_resolver`, scenario helpers | keep |
| `pymargins/_adapter.py` | 817 | `ModelAdapter` ABC + `GLMAdapter`/`LinearPrediction`/`WrappedFD`/`BootstrapOnly` shapes; `influence():404`, `data_fingerprint():415`, `refit():450` | keep |
| `pymargins/_delta.py`, `_gradients.py` | — | `delta_se`, `delta_confint`, `delta_wald_test`; `gradient(h, beta, backend, fd_step)` | keep |
| `pymargins/_transforms/`, `matching/`, `survey.py` | — | stages (Stage protocol), matcher client, `SurveyDesign` (`hash_key()`) | keep |
| `pymargins/_result/_pooling.py` | 354 | `pool_imputations` Rubin arithmetic | re-pointed R4 |

### G1.2 How a number is produced today (legacy path — the semantics you replicate)

```
Margins(model, **posture)                      # session binds posture
  └ query method (predict/dydx/…)              # _session.py:772/886/1065/1229
      ├ _build_*_estimand(session, scenario…)  # margins/_estimands.py → (h, labels, scenarios)
      │    └ uses root atoms make_*_estimand + expand_scenario + _enumerate_groups
      ├ h_factory (inline def)                 # rebuilds h on a refit adapter (bootstrap)
      ├ _inference_config(session)             # glue: banks + frozen Σ̂ + InferenceConfig
      ├ run_inference(h, adapter, config…)     # dispatch (fallbacks live here — doctrine deletes)
      │    └ _run_delta | _run_simulation | _run_bootstrap   # kernels (the numbers)
      └ _wrap_result(session, result_data)     # glue → MarginsResult
```

The new engine reimplements the *orchestration* boxes (builders, config,
dispatch, wrap) session-free and doctrine-shaped; the kernels and atoms are
consumed as-is after review.

### G1.3 Contracts you consume (verified 2026-06-11)

**Kernel result dict** (`_run_delta`/`_run_simulation`/`_run_bootstrap` all
return this; `_wrap_result` consumes it): keys `estimate`, `std_error`,
`conf_int_lower`, `conf_int_upper`, `method`, `level`, `kappa`,
`delta_sim_disagreement`, `fallback_triggered`, `fallback_reason`,
`gradient`, `draws`, `draws_inf`, `estimand_metadata`, `ci_method`,
`bootstrap_extras` (+ bootstrap adds `n_boot_effective`, `n_boot_failed`).
All reporting-scale except `draws_inf`/`gradient` (inference scale).

**`InferenceConfig`** (`_inference/_config.py`): the new engine sets
`kappa_threshold=float("inf")` **always** (kills the `_run_delta:36` flip
without touching the kernel), `diagnostics=True` always, and injects banks
via `all_idx` / `all_states` / `all_states_failures` / `sim_draws`.

**Seed derivations** (the determinism facts):
- simulation draws: `np.random.default_rng([seed, 0])` →
  `_generate_simulation_draws(beta, Sigma, rng, n_sim)`
- bootstrap indices: `np.random.default_rng([seed, 1])` **inside**
  `_generate_resample_indices(rng_seed, n_boot, n_obs, cluster_ids,
  block_size, block_type, strata)` — iid / cluster / block
  (moving|circular|nonoverlapping) / stratified Rao–Wu paths.
- `[seed, 0]` vs `[seed, 1]` is load-bearing; swapping them changes every
  stochastic result.

**vcov_spec resolution** (legacy `_session.py:372–575`, replicate at C2):
explicit `vcov=` wins; else `cluster` declared ⇒
`{"type": "cluster", "groups": arr}`; else survey design declared ⇒
`{"type": "survey", "design": design}`; else `None` (model default Σ̂).
Adapters consume the spec via `adapter.covariance(vcov_spec)` — string
("HC1", …) refits with that `cov_type`; survey dict runs
`_survey_covariance` (linearization). **The survey design must also reach
the resampler** (PSU/strata into `_generate_resample_indices`) — two
consumers, one declaration (design §6.1 "forgotten dependence" row).

**Frozen surface signatures** (validated by the facade; do not rename):

```python
GComputation(wiring_or_model=None, *, outcome=None, at="overall",
             scale="response", method="delta", vcov=None, ci="wald",
             level=0.95, B=0, n_sim=0, seed=None,
             n_jobs=1, progress_bar=False)        # R6 adds: weights, adapter,
                                                  # gradient_backend, fd_step (req §7)
est.predict(*, atexog=None, over=None, transform=None, label=None, outcome=None)
est.dydx(variables=None, *, atexog=None, over=None, transform=None, label=None, outcome=None)
est.eyex(variable, **k) / est.eydx(variable, **k) / est.dyex(variable, **k)
est.contrasts(*, scenarios=None, contrasts=None, outcome=None)
est.evaluate(*, scenarios=None, compose=None, outcome=None)
est.rmst(*, horizon=None, atexog=None, over=None, n_grid=80)
est.joint(*results)            # NotImplementedError naming 0.5.0
result.conf_int(correction=None)   # None | "bonferroni" | "sidak" | "sup-t"
```

Behavioral notes: `dydx` **refuses binary/categorical/discrete variables**
(`column_index_of_variable` raises — treatment effects go through
`contrasts`); slopes are data-side central differences (R/Stata-style).

### G1.4 Environment facts

- `pyproject.toml` version is **0.3.0** (plan assumed 0.4.0 — bump at R0).
- pytest runs with `--strict-markers` and **no markers are registered**:
  the first `@pytest.mark.slow` will error until you add
  `markers = ["slow: weekly-lane statistical tests"]` under
  `[tool.pytest.ini_options]`. Do this in R0.
- `ruff` is configured (`[tool.ruff]`) but **not installed/declared** —
  add `ruff>=0.4` to the `test` extra in R0.
- `tests/conftest.py` enables JAX x64 — any new script entry point
  (`tools/`) that touches pymargins must do the same before importing jax
  consumers.
- Local R stack (verified): R 4.6.0, `marginaleffects` 0.32.0, `survey`
  4.5, `sandwich` 3.1.1, `margins` 0.3.28. Missing (install when their
  cases land): `emmeans`, `multcomp`, `survRM2`.
- `hypothesis` is in the test extra (use for graph-law property tests).

---

## G2. Global conventions

- **Errors:** `CompileError` (in `_soundness/_predicates.py`, subclasses
  `ValueError`) for every compile/query refusal; `SoundnessWarning` for
  warns; `BankRetentionError` (in `_engine/_banks.py`) reserved for the
  0.5.0 fan path. Do not create new exception types without a plan basis.
- **Module headers:** every new module's docstring carries the design
  citation and `Added in 0.4.0 (R<n>).`, e.g.
  `"""Query construction. Design §4.8, req §2. Added in 0.4.0 (R2)."""`
- **Tests:** one test file per module (`tests/test_engine_queries.py` for
  `_engine/_queries.py`); oracle suites under `tests/oracle/`; legacy
  corroboration under `tests/anchor/`. Numeric tolerances come from
  `tests/oracle/_tolerances.py` constants — never inline literals.
- **The two ledgers** (create on first entry, append-only):
  - `notes/061126_legacy_defect_ledger.md` — entry template:

    ```
    ## D<n> — <one-line title>            (date, workstream)
    Where: <file:line / case id>
    Claim: <what legacy computes> vs Oracle: <what the oracle says>
    Evidence: <oracle case id(s), numbers, tolerance>
    Disposition: fix-in-place | rewrite | facade-only (dies at R5/R6) |
                 accept-with-rationale
    Status: open | fixed (<commit>) | recorded-for-R8-changelog
    ```
  - `notes/061126_test_port_ledger.md` — entry template:

    ```
    ## P<n> — tests/<file>::<test>         (date)
    Category: (a) reference-derived | (b) engine-derived | (c) semantic-change
    Disposition: ported-mechanical | re-anchored (<oracle case>) |
                 replaced-by-refusal-test | dropped (<reason>) |
                 legacy-corroborated-regression (no oracle reach)
    ```
- **Gate commands:** `pytest -m "not slow" -q` and `ruff check .` — both
  green before a workstream is declared done. Weekly lane:
  `pytest -m slow -q` (gates release tags only).
- **Commits:** one commit (or small series) per workstream, message
  prefixed `R<n>:`. Never commit a red gate.

---

## R0 — Checkpoint

**Goal:** the uncommitted tree lands in history as structured commits; the
dev environment can actually run the gates.

Steps (on branch `graph_api`):

1. Env: add `ruff>=0.4` to `[project.optional-dependencies].test`; add
   `markers = ["slow: weekly-lane statistical tests"]` to
   `[tool.pytest.ini_options]`; bump `version = "0.4.0"` (the *tag* is
   gated, not the string). Install: `pip install -e ".[test,statsmodels]"
   && pip install ruff`.
2. Commit slices (plan §1) — suggested messages:
   1. `R0: Phase 1 — adapter influence() contract + tests` →
      `pymargins/_adapter.py`, `pymargins/_adapters/*`,
      `tests/test_influence_contract.py`.
   2. `R0: soundness layer (constants + predicates) + tests` →
      `pymargins/_soundness/`, `tests/test_soundness_*.py`.
   3. `R0: graph surface (Node/Plan/steps) + tests` →
      `pymargins/_graph/`, `pymargins/steps/`, `tests/test_graph_node.py`,
      `tests/test_steps.py`.
   4. `R0: interim scaffolding — replaced by R1–R6 (see 061126 plan)` →
      `pymargins/estimators/`, `pymargins/_engine/`,
      `pymargins/_result/_graphresult.py`, `pymargins/_graph/_compile.py`
      if not in slice 3, `tests/anchor/`, `tests/test_graphresult.py`,
      `tests/test_compile.py`.
   5. `R0: metadata — exports, changelog, pyproject, plan rev. 2 + guide`
      → `pymargins/__init__.py`, `pymargins/_result/__init__.py`,
      `CHANGELOG.md`, `pyproject.toml`, `docs/` stubs, `notes/061126_*`.
3. Gate: `pytest -m "not slow" -q` and `ruff check .`. **Expect ruff
   findings in pre-existing code** — fix only what blocks (syntax-level);
   record a count; do not reformat the legacy tree (it's frozen and dies
   at R7; churn there pollutes the diff).

**Acceptance:** both gates green (or ruff failures isolated to frozen
legacy files and recorded); five commits pushed.

**Pitfalls:** don't squash the slices (the interim/permanent distinction
is the point); don't "fix" legacy lint.

---

## R1 — Validation harness, seeds, banks

This is the foundation workstream. Build the oracle stack **first**, run
it against the current tree (facade → legacy numbers) to baseline, then do
seeds and banks.

### R1.1 File layout

```
tools/oracle/
  lib.R              # write_golden() helper, version capture
  ols.R  logit.R  probit.R  poisson.R  survey_glm.R   # one per family
  README.md          # how to regenerate; regeneration = ledger entry
tests/oracle/
  __init__.py
  _tolerances.py     # the only place tolerances live
  _datasets.py       # deterministic generators + write_data() entry point
  data/oracle_main.csv          # committed
  golden/<case_id>.json         # committed, R-generated
  conftest.py        # load_golden(), fit fixtures, assert helpers
  test_analytic.py   # layer 1 (closed forms)
  test_r_golden.py   # layer 2 (golden comparisons)
tests/golden/        # created at R7, empty until then
```

### R1.2 Tolerances (`tests/oracle/_tolerances.py`)

```python
"""Oracle tolerances. Plan §4 layer 2. Changing any value = ledger entry."""
TOL_COEF = 1e-8      # β̂ fit-alignment gate (statsmodels vs R, tightened fits)
TOL_EST  = 1e-6      # effect estimates
TOL_SE   = 1e-5      # standard errors
TOL_CI   = 1e-5      # CI endpoints (only when conventions declared equal)
TOL_ANALYTIC = 1e-10 # closed-form identities (float association order only)
```

Comparisons are `np.testing.assert_allclose(actual, expected, rtol=TOL_*,
atol=0.0)`. If a case has true zeros, that case's golden records an
explicit `atol` with a one-line reason — never a global atol.

### R1.3 Dataset (`tests/oracle/_datasets.py`)

One frame serves every family. Deterministic, committed as CSV (pandas
`to_csv` writes shortest-roundtrip float repr — lossless both ways; R's
`read.csv` parses full doubles, so **both languages see bit-identical
data**).

```python
def make_oracle_main(seed: int = 20260611, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    treat = rng.binomial(1, 0.5, size=n).astype(float)
    eta_b = -0.3 + 0.8 * treat + 0.5 * x1 - 0.4 * x2
    y_bin = rng.binomial(1, 1 / (1 + np.exp(-eta_b))).astype(float)
    y_count = rng.poisson(np.exp(0.1 + 0.4 * treat + 0.3 * x1)).astype(float)
    y_cont = 1.0 + 2.0 * treat + 1.5 * x1 - 1.0 * x2 + rng.normal(size=n)
    g = np.repeat(np.arange(25), n // 25)            # 25 clusters of 16
    strata = np.repeat(np.arange(4), n // 4)         # 4 strata of 100
    obs_per_psu = n // 4 // 5                        # 5 PSUs per stratum
    psu = strata * 100 + np.tile(np.repeat(np.arange(5), obs_per_psu), 4)
    # ≥2 PSUs per stratum guaranteed (5 each) — avoids the lonely-PSU refusal
    w = np.exp(rng.normal(0.0, 0.3, size=n))         # positive survey weights
    return pd.DataFrame({...all columns...})
```

`python -m tests.oracle._datasets` writes `data/oracle_main.csv`.
Regenerating the CSV invalidates every golden — same discipline as golden
regeneration (ledger entry); the seed/n are frozen after R1.

### R1.4 Golden JSON schema (one file per case)

```json
{
  "case_id": "logit_ame_x1_hc1",
  "created": "2026-06-11",
  "r_version": "R version 4.6.0",
  "packages": {"marginaleffects": "0.32.0", "sandwich": "3.1.1"},
  "data": "oracle_main.csv",
  "model": {"formula": "y_bin ~ treat + x1 + x2", "family": "binomial(logit)",
            "fit_control": "glm.control(epsilon = 1e-12, maxit = 200)"},
  "r_call": "avg_slopes(fit, variables = 'x1', vcov = vcovHC(fit, type = 'HC1'))",
  "vcov": "HC1",
  "ci_convention": {"dist": "z", "level": 0.95},
  "labels": ["x1"],
  "quantities": {
    "coefficients": [/* β̂ from R, full precision */],
    "estimate": [...], "std_error": [...],
    "conf_low": [...], "conf_high": [...]
  },
  "tolerances": {},          // only per-case overrides; empty = use _tolerances.py
  "notes": ""
}
```

Rules: `coefficients` is **mandatory** in every golden — it is the
fit-alignment gate (§R1.7). `quantities` may add `vcov_matrix`
(row-major flattened) for the alignment cases. jsonlite must write with
`digits = NA` (no rounding).

### R1.5 R side — `tools/oracle/lib.R` and a complete worked example

`lib.R`:

```r
library(jsonlite)
write_golden <- function(case_id, data, model, r_call, quantities,
                         vcov = "nonrobust",
                         ci_convention = list(dist = "z", level = 0.95),
                         labels = NULL, tolerances = list(), notes = "") {
  pkgs <- c("marginaleffects", "sandwich", "survey")
  pkgs <- pkgs[vapply(pkgs, requireNamespace, TRUE, quietly = TRUE)]
  payload <- list(
    case_id = case_id, created = format(Sys.Date()),
    r_version = R.version.string,
    packages = setNames(lapply(pkgs, function(p) as.character(packageVersion(p))), pkgs),
    data = data, model = model, r_call = r_call, vcov = vcov,
    ci_convention = ci_convention, labels = labels,
    quantities = quantities, tolerances = tolerances, notes = notes)
  path <- file.path("tests", "oracle", "golden", paste0(case_id, ".json"))
  write(toJSON(payload, digits = NA, auto_unbox = TRUE, pretty = TRUE), path)
  cat("wrote", path, "\n")
}
```

`tools/oracle/logit.R` (run from repo root: `Rscript tools/oracle/logit.R`):

```r
source("tools/oracle/lib.R")
library(marginaleffects); library(sandwich)
df <- read.csv("tests/oracle/data/oracle_main.csv")
fit <- glm(y_bin ~ treat + x1 + x2, family = binomial(), data = df,
           control = glm.control(epsilon = 1e-12, maxit = 200))
mod <- list(formula = "y_bin ~ treat + x1 + x2", family = "binomial(logit)",
            fit_control = "glm.control(epsilon = 1e-12, maxit = 200)")

p <- avg_predictions(fit)                       # average adjusted prediction
write_golden("logit_predict_overall_nonrobust", "oracle_main.csv", mod,
  "avg_predictions(fit)",
  list(coefficients = unname(coef(fit)), estimate = p$estimate,
       std_error = p$std.error, conf_low = p$conf.low, conf_high = p$conf.high))

s <- avg_slopes(fit, variables = "x1")          # AME, model-default vcov
write_golden("logit_ame_x1_nonrobust", "oracle_main.csv", mod,
  "avg_slopes(fit, variables = 'x1')",
  list(coefficients = unname(coef(fit)), estimate = s$estimate,
       std_error = s$std.error, conf_low = s$conf.low, conf_high = s$conf.high),
  labels = list("x1"))

sh <- avg_slopes(fit, variables = "x1", vcov = vcovHC(fit, type = "HC1"))
write_golden("logit_ame_x1_hc1", "oracle_main.csv", mod,
  "avg_slopes(fit, variables = 'x1', vcov = vcovHC(fit, type = 'HC1'))",
  list(coefficients = unname(coef(fit)), estimate = sh$estimate,
       std_error = sh$std.error, conf_low = sh$conf.low, conf_high = sh$conf.high),
  vcov = "HC1", labels = list("x1"))

cmp <- avg_comparisons(fit, variables = "treat")   # counterfactual risk diff
write_golden("logit_contrast_treat_nonrobust", "oracle_main.csv", mod,
  "avg_comparisons(fit, variables = 'treat')",
  list(coefficients = unname(coef(fit)), estimate = cmp$estimate,
       std_error = cmp$std.error, conf_low = cmp$conf.low, conf_high = cmp$conf.high))

# counterfactual all-treated prediction (pymargins predict(atexog={'treat': 1}))
g1 <- avg_predictions(fit, newdata = datagrid(treat = 1, grid_type = "counterfactual"))
write_golden("logit_predict_at_treat1_nonrobust", "oracle_main.csv", mod,
  "avg_predictions(fit, newdata = datagrid(treat = 1, grid_type = 'counterfactual'))",
  list(coefficients = unname(coef(fit)), estimate = g1$estimate,
       std_error = g1$std.error, conf_low = g1$conf.low, conf_high = g1$conf.high))
```

`survey_glm.R` uses `svydesign(ids = ~psu, strata = ~strata, weights = ~w,
data = df, nest = TRUE)` + `svyglm`, then `avg_predictions`/`avg_slopes` on
the svyglm fit (marginaleffects supports survey objects).

### R1.6 Python side — `tests/oracle/conftest.py` + test shape

```python
GOLDEN_DIR = Path(__file__).parent / "golden"

def load_golden(case_id: str) -> dict: ...

@pytest.fixture(scope="session")
def oracle_df(): return pd.read_csv(Path(__file__).parent / "data" / "oracle_main.csv")

@pytest.fixture(scope="session")
def fit_logit(oracle_df):
    return smf.glm("y_bin ~ treat + x1 + x2", data=oracle_df,
                   family=sm.families.Binomial()).fit(tol=1e-12, maxiter=200)

def assert_coef_aligned(fit, golden):
    """Fit-alignment gate. A failure here is misalignment, NOT a defect."""
    np.testing.assert_allclose(np.asarray(fit.params),
                               np.asarray(golden["quantities"]["coefficients"]),
                               rtol=TOL_COEF, atol=0.0)

def assert_matches_golden(golden, *, estimate, std_error=None,
                          conf_low=None, conf_high=None): ...
```

Test shape (one per case, parametrization optional later):

```python
def test_logit_ame_x1_hc1(fit_logit):
    g = load_golden("logit_ame_x1_hc1")
    assert_coef_aligned(fit_logit, g)
    est = GComputation(fit_logit, at="overall", method="delta", vcov="HC1")
    r = est.dydx("x1")
    assert_matches_golden(g, estimate=r.estimate, std_error=r.std_error)
```

CI endpoints are compared **only when** pymargins' convention equals the
golden's `ci_convention` (pymargins delta CIs are z-based — verify once in
the analytic suite, then compare CIs on z-convention cases).

### R1.7 Starter case matrix (v1 — ~24 goldens)

| family | queries | vcov axes | population axes |
|---|---|---|---|
| OLS (`y_cont`) | predict overall; AME x1; contrast treat | nonrobust, HC1, cluster(g) | unweighted |
| logit (`y_bin`) | predict overall; predict at treat=1; AME x1; contrast treat | nonrobust, HC1, cluster(g) | unweighted, weights(w) |
| probit (`y_bin`) | AME x1 | nonrobust | unweighted |
| Poisson (`y_count`) | predict overall; AME x1 | nonrobust, HC1 | unweighted |
| survey logit | predict overall; AME x1 | survey linearized | design(w, psu, strata) |

Mapping pymargins → marginaleffects (record the final call per golden):
`predict()` ↔ `avg_predictions(fit)`;
`predict(atexog={"v": c})` ↔ `avg_predictions(fit, newdata=datagrid(v=c, grid_type="counterfactual"))`;
`dydx("x")` ↔ `avg_slopes(fit, variables="x")` (continuous only);
`contrasts(scenarios=[{atexog:{treat:1}},{atexog:{treat:0}}], contrasts=[1,-1])`
↔ `avg_comparisons(fit, variables="treat")`;
cluster vcov ↔ `vcov = vcovCL(fit, cluster = ~g, type = "HC1")` — **verify
this pairing empirically** against statsmodels `cov_type="cluster"` in an
alignment case (golden carrying `vcov_matrix`) before recording effect
goldens; if no `type=` matches within `TOL_SE`, record the closest pairing
+ a note, and compare SEs at a per-case tolerance with the difference
explained (df-correction conventions), or stop and ask.
Weighted cases: pymargins `weights=` (aggregation weights) ↔
marginaleffects `wts = "w"` in `avg_*` — same caveat: verify semantics
match (weighted average of unit-level quantities) before recording.

### R1.8 Baselining workflow (run the suites against the *current* tree)

1. Generate data → run R scripts → commit goldens.
2. Run `pytest tests/oracle -q` with the **current facade**
   (`GComputation` → legacy engine).
3. Triage every failure:
   - fit-alignment gate failed → fix alignment (tolerance/convergence/
     data parsing); not a defect.
   - estimand-semantics mismatch (systematic, e.g. wrong population) →
     fix the **R call** to match the declared pymargins estimand; record.
   - genuine numeric disagreement → defect-ledger entry. Decide
     `facade-only` (bug in kwarg translation — dies at R5/R6; mark the
     test `xfail(strict=True, reason="D<n>: facade defect, fixed by R6")`)
     vs `legacy/kernel` (the rewrite must *not* reproduce it; the oracle
     test stays red until the new engine passes it — and the matching
     anchor cell gets an expected-divergence marker).
4. Arbitration order when sources disagree: analytic > R-oracle consensus >
   single R oracle > legacy. Beyond tolerance with no explanation →
   **stop and ask.**

### R1.9 Analytic suite (`tests/oracle/test_analytic.py`)

Each test computes the expected value **in-test with numpy from the fitted
model** (no hardcoded floats, no pymargins call on the expected side).
Standard delta-method results (chain rule); cite "delta method — standard,
e.g. Wooldridge (2010) §3" in the module docstring.

1. `test_ols_ame_equals_beta` — linear model: `dydx("x1")` estimate ==
   `params["x1"]`, SE == `bse_nonrobust["x1"]` (= `sqrt(Σ̂[j,j])`).
   rtol `TOL_ANALYTIC`.
2. `test_ols_mean_prediction_equals_ybar` — `predict()` == `mean(y)` (OLS
   with intercept); SE == `sqrt(x̄ᵀ Σ̂ x̄)`.
3. `test_logit_ame_closed_form` — AME_j = `mean(p*(1-p)) * β_j` with
   `p = expit(Xβ̂)`; gradient
   `∇_k = mean(p(1-p) * (δ_jk + β_j (1-2p) x_k))`; SE = `sqrt(∇ᵀ Σ̂ ∇)`.
4. `test_poisson_ame_closed_form` — AME_j = `mean(exp(Xβ̂)) * β_j`;
   `∇_k = mean(exp(Xβ)(δ_jk + β_j x_k))`.
5. `test_logit_risk_difference_closed_form` — Δ = `mean(p₁) − mean(p₀)`
   under counterfactual design matrices X₁/X₀;
   `∇ = mean(p₁(1-p₁)X₁) − mean(p₀(1-p₀)X₀)`.
6. `test_weighted_logit_ame` — same as 3 with normalized weights
   (`Σwᵢ·/Σw`).
7. `test_hand_ols_micro` — n=4, X=[1, (0,1,2,3)], y=(1,3,2,5): compute
   β̂=(XᵀX)⁻¹Xᵀy, σ̂²=RSS/(n−2), Σ̂=σ̂²(XᵀX)⁻¹ with explicit numpy ops;
   pymargins predict/dydx estimates and SEs match. Full pipeline, zero
   library trust.
8. `test_ci_convention_is_z` — delta CI == `estimate ± z_{level}·SE`
   computed with `scipy.stats.norm.ppf` — pins the convention the R
   comparisons rely on.

These run against the current tree at R1 (baselining legacy) and against
the new engine from R6 on — same tests, no edits.

### R1.10 Seeds (`_engine/_seeds.py`)

Review gate (I3″) on `_generate_resample_indices` and
`_generate_simulation_draws`: read both; they are plain numpy RNG
engineering (no statistics to cite beyond Rao–Wu for the stratified path —
confirm the within-stratum PSU resample matches Rao–Wu 1988 as design
§6.5 asserts). Expected outcome: **keep**. If you find a defect, ledger +
stop and ask before changing a derivation (plan trap 2: redesign happens
once, here, or never).

Keep the wrappers; rename is allowed at R7 (drop the `legacy_` prefix),
not now. Fix `seed_sequence_for_branch` inefficiency (spawns
`branch_id+1` children per call) **only if** the determinism property
tests still pass byte-identically — otherwise leave it.

`tests/test_engine_seeds.py`:

- `test_resample_indices_deterministic` — two calls, same args ⇒
  `np.array_equal` on every replicate array; parametrize over
  {iid, cluster, block-moving, block-circular, stratified} × seeds
  {0, 7, 20260611} with tiny sizes (n=12, B=4).
- `test_resample_indices_regression_golden` — the same matrix asserted
  against literal recorded arrays **pasted into the test after one
  verified run** (these are layer-4 regression goldens; the paste-in run
  is recorded in the commit message; regeneration thereafter = ledger).
- `test_sim_draws_regression_golden` — seeds × (p=2, n_sim=5) literals;
  also assert `legacy_sim_draws(seed,…)` ≡
  `_generate_simulation_draws(beta, cov, default_rng([seed, 0]), n)` (the
  wrapper-vs-derivation identity) and the indices wrapper ≡ direct call
  (which pins `[seed, 1]` internally).
- `test_spawn_tree_order_invariant` — `seed_sequence_for_branch(s, b, n)`
  equal regardless of evaluation order of `b`; and distinct across
  branches.

### R1.11 Banks (`_engine/_banks.py` rebuilt)

```python
@dataclass
class BankSet:
    """Per-(estimator, branch) inference banks. Design §9.4, req §5. Added in 0.4.0 (R1)."""
    plan_hash: str
    branch_id: int
    seed: int | None
    # internal: _index_bank, _states_bank (adapters; no-fan retention),
    #           _states_failures, _draws_bank

    def resample_indices(self, *, n_obs, B, cluster=None, block=None,
                         block_type="moving", strata=None) -> list[np.ndarray]:
        # get-or-build via legacy_resample_indices(self.seed, ...)
    def bootstrap_states(self, *, adapter, data, indices, matching=None,
                         transforms=None, n_jobs=1, progress=False):
        # get-or-build via _harvest_bootstrap_states; returns (states, failures)
    def sim_draws(self, *, beta, cov, n_sim) -> np.ndarray:
        # get-or-build via legacy_sim_draws(self.seed, n_sim, beta, cov)
```

- Build once, replay across queries (this is what makes results from one
  estimator composable — the legacy glue's bank semantics, owned by an
  object instead of session attribute stuffing).
- **Delete `RetentionPolicy` entirely** (byte-budget rescinded).
- Keep `BankRetentionError` with the message text:
  `"This query needs replicate products that were not retained from the
  fan run. Issue queries together so products are captured in one pass;
  re-running is deterministic (same seed tree), but costs another fan
  execution."` — unraised in 0.4.0 (no fans); 0.5.0 wires it.
- `tests/test_engine_banks.py`: `test_indices_built_once` (two calls, same
  list object / arrays equal), `test_draws_built_once`,
  `test_states_replayed` (harvest called once — monkeypatch-count the
  kernel), `test_distinct_seeds_distinct_banks`.

### R1.12 Acceptance gate (R1)

`pytest tests/oracle tests/test_engine_seeds.py tests/test_engine_banks.py -q`
green (oracle failures only as ledgered xfails per §R1.8) + full gate
green. Commit `R1: validation harness (analytic + R goldens) + seeds/banks`.

### R1 pitfalls

- **jsonlite rounding:** forgetting `digits = NA` writes 4 significant
  digits and every comparison fails mysteriously at ~1e-4.
- **CSV dtype drift:** `treat` as int in Python but float in R (or
  vice-versa) changes nothing numerically but `read.csv` + factors can —
  never let R treat a numeric column as factor (no `stringsAsFactors`
  surprises; all columns numeric).
- **statsmodels default IRLS tol is 1e-8** — fit with `tol=1e-12` or the
  β̂ gate flickers.
- **marginaleffects `avg_slopes` on `treat`** silently computes a 1-0
  contrast; pymargins `dydx("treat")` refuses. Slope cases use continuous
  variables only.
- **z vs t:** statsmodels OLS `conf_int()` is t-based; pymargins delta CIs
  are z-based; marginaleffects defaults to z. Compare CI arrays only on
  declared-z cases (analytic test 8 pins pymargins' convention).
- **Don't compare draws/bootstrap to R.** Resampling validation is
  layer 4 (determinism) + layer 5 (calibration, R8 slow lane) only.
- Survey: `nest = TRUE` in `svydesign` (PSU ids restart within strata in
  the dataset above). Omitting it silently merges PSUs across strata.

**Stop-and-ask triggers:** any oracle-vs-oracle disagreement beyond
tolerance; any seed-derivation defect; cluster-vcov pairing that matches
no `vcovCL` type.

---

## R2 — Query layer (`_engine/_queries.py`)

**Goal:** session-free query construction — the heaviest-validated module
in the package, because this is where the facade's bug class lived.

**Read first:** `margins/_estimands.py` (all of it — you are porting it),
`margins/_atoms.py`, root `_estimands.py` atom signatures, design §4.8,
the inline `h_factory` defs at `_session.py:870, 967, 1208, 1340`.

### R2.1 API (pinned)

```python
"""Query construction: spec → estimand. Design §4.2/§4.8, req §2. Added in 0.4.0 (R2)."""

@dataclass(frozen=True)
class QueryContext:
    """Everything query construction may read. No session anywhere."""
    adapter: ModelAdapter
    base_data: Any                  # wiring point-execution output (post-match/trim)
    at: str
    weights: np.ndarray | None
    phi: Callable | None
    phi_inv: Callable | None
    fd_step: float
    gradient_backend: str

@dataclass(frozen=True)
class QuerySpec:
    kind: str                       # "predict"|"dydx"|"eyex"|"eydx"|"dyex"|
                                    # "contrasts"|"evaluate"|"rmst"|"wtp"
    scenario: Mapping | None = None # {"atexog":…, "over":…} for predict/dydx
    variables: tuple[str, ...] | None = None
    scenarios: tuple[Mapping, ...] | None = None   # contrasts/evaluate
    contrast_weights: Any | None = None
    compose: Callable | None = None
    transform: Callable | None = None
    label: str | None = None
    outcome: int | tuple[int, ...] | None = None
    horizon: float | None = None    # rmst
    n_grid: int = 80                # rmst

@dataclass(frozen=True)
class CompiledQuery:
    h: Callable                     # β → estimand (inference scale)
    h_factory: Callable | None      # refit-adapter → h (bootstrap re-execution)
    labels: list[str] | None
    scenarios: list[dict]
    estimand_metadata: dict

def compile_query(spec: QuerySpec, ctx: QueryContext) -> CompiledQuery: ...
def resolve_scale(scale) -> tuple[Callable | None, Callable | None]:
    # named: response/identity→(None,None), log, logit, probit (lift the
    # facade's _scale_to_phi verbatim); tuple(callable, callable) passes
    # through; anything else → CompileError listing supported names.
```

`compile_query` dispatches to private per-kind builders. Each builder is
the corresponding `_build_*_estimand` from `margins/_estimands.py` with
`session.X` replaced by `ctx.X` — a **mechanical port**:

| ctx field | replaces |
|---|---|
| `ctx.at` | `session.at` |
| `ctx.weights` | `session.weights` |
| `ctx.phi_inv` | `session.phi_inv` |
| `ctx.fd_step` | `session.fd_step` |
| `ctx.adapter` | `session.adapter` |
| `ctx.base_data` | `_get_base_data(session, adapter)` |

`_get_base_data`'s matching branch dissolves: in the new engine the wiring
already delivers matched/trimmed data as `base_data` (the noun collects it
at C2). Keep `_bootstrap_weights_for_adapter`'s subsetting logic
(`_pymargins_bootstrap_idx`) — it is load-bearing for weighted bootstrap.
Port `_enumerate_groups` / `_format_atom_label` / `_finalize_atoms` as
module-level functions (their `session` arg is already unused or
trivially replaced). `h_factory` builders: port the inline defs from the
four session methods — each closes over the spec and rebuilds the estimand
on a refit adapter (its `base_data` = the refit adapter's `training_data`,
matching legacy `_get_base_data` semantics for replicates).

- `eyex/eydx/dyex`: port `_elasticity` (`_session.py:1031`) semantics as
  spec kinds (slope and/or prediction scaling; read the method body and
  move it).
- `wtp`: declared ratio via the evaluate path (design §4.8) — port
  `_session.py:995`'s construction onto `make_evaluate_estimand`, not
  result division.
- `rmst`: port `_session.py:1361` (trapezoid over survival-curve
  predictions on an `n_grid` time grid; requires a time-aware adapter).
- This workstream is also the **citation review (I3″) of root
  `_estimands.py` and `margins/_atoms.py`**: read the atom kernels,
  confirm each matches its docstring formula, note findings in the ledger.

### R2.2 The config builder

```python
def build_inference_config(plan, adapter, wiring_facts, banks,
                           *, n_jobs=1, progress_bar=False) -> InferenceConfig:
```

Doctrine-shaped, mirroring `_inference_config` (glue) minus the session:
`method=plan.method_resolved`, `level=plan.level`,
`kappa_threshold=float("inf")` **unconditionally**, `diagnostics=True`,
`phi/phi_inv` from `resolve_scale(plan.scale)`, `n_sim/n_boot/rng_seed`
from plan, `cov_params=` the estimator's **frozen Σ̂** (computed once via
`adapter.covariance(resolved_vcov_spec)` and cached on the noun — port
`_frozen_cov` semantics), `cluster/block_size/strata/survey_design/
matching/transforms` from `wiring_facts` (R5.1 step 2), banks injected
(`all_idx`, `all_states`, `all_states_failures`, `sim_draws`) by the
executor (R3), `bootstrap_config={"ci_method": plan.ci, "block_type":
wiring_facts.block_type}` rebuilt from plan + wiring facts — **never a
raw user dict** (`block_type` declares at `steps.input`, Appendix C #10).

### R2.3 Tests

`tests/test_engine_queries.py` — for every query kind, dual-construct and
compare against the legacy builder *directly* (this is allowed in tests):

```python
def test_predict_h_matches_legacy(fit_logit):
    m = Margins(fit_logit, at="overall", method="delta")
    h_old, labels_old, scen_old = _build_prediction_estimand(m, {}, None)
    cq = compile_query(QuerySpec(kind="predict", scenario={}), ctx_from(m))
    beta = m.adapter.coefficients()
    np.testing.assert_array_equal(np.asarray(h_old(beta)), np.asarray(cq.h(beta)))
    assert labels_old == cq.labels
```

Parametrize across: kinds × {no options, atexog scalar, atexog grid,
over=, weights=, scale="log", transform=} (each axis that changes input
construction — plan R2). Plus `test_resolve_scale_*` (named, callable
pair, unknown → CompileError) and `test_config_doctrine_shape`
(`kappa_threshold == inf`, banks fields None before executor injection).

### R2 pitfalls

- The estimand is built **on the inference scale** (`phi_inv` baked into
  the atom); the kernel applies `phi` for reporting. Dropping `phi_inv`
  here double-transforms or under-transforms — compare against legacy on
  a `scale="log"` case specifically.
- `at="overall"` vs single-row aggregation: the legacy builders pick
  `agg_kind = "none" if X_i.shape[0] == 1 else "overall"` for non-overall
  `at` — port that branch exactly; it's easy to flatten and wrong to.
- Multi-atom stacking order is label order is estimate order — the anchor
  matrix catches transpositions only if you include an `over=` case.
- Weighted aggregation uses `make_aggregation_resolver(at, weights)` —
  pass `ctx.weights`, not normalized weights (the resolver normalizes).
- Do not import `pymargins.margins` here (I6) — the test file imports
  legacy, the module under test never.

**Acceptance:** query tests green; oracle suite (R1) unchanged-green
(still running through the facade — R2 is not wired into the surface yet);
full gate green. Commit `R2: session-free query layer`.

---

## R3 — Doctrine dispatch and executor (`_engine/_execute.py`)

**Goal:** the engine entry point. Kernels called directly; no fallback
branches exist anywhere in this module.

**Read first:** `_inference/_dispatch.py` (what you are *not*
reproducing), `_inference/_delta.py` / `_simulation.py` / `_bootstrap.py`
kernel signatures, glue `_bootstrap_resample_bank`/`_bootstrap_states_bank`
(bank wiring you replicate via `BankSet`), design §5.2, §6.1.

### R3.1 API (pinned)

```python
"""Doctrine executor. Design §5, req §5. Added in 0.4.0 (R3)."""

def execute_query(compiled: CompiledQuery, *, adapter, plan, wiring_facts,
                  banks: BankSet, frozen_cov, n_jobs=1,
                  progress_bar=False) -> dict:
    """Run one compiled query through the resolved method. Returns the
    kernel result dict (G1.3) — the result layer (R4) wraps it."""
```

Behavior spec:

1. `method = plan.method_resolved` — read, never recomputed, never changed.
2. **delta:** check `is_jax_differentiable(compiled.h, beta)` first. Not
   differentiable ⇒ `CompileError` with the design §6.1 steer (verbatim
   row text; the steer is `method="simulation"`). Then call
   `_run_delta(h, adapter, config, metadata)` — with `kappa_threshold=inf`
   in config the kernel's flip branch is dead; κ is still computed and
   lands in the result dict (decide-once doctrine, design §5.2.3).
3. **simulation:** `banks.sim_draws(beta=…, cov=frozen_cov,
   n_sim=plan.n_sim)` → `config.sim_draws` → `_run_simulation(...)`.
4. **bootstrap:** resolve the resampling declaration from `wiring_facts`:
   `cluster_ids = matching.cluster_ids if matching else wiring_facts.cluster`;
   survey design present ⇒ `cluster_ids = design.psu`,
   `strata = design.strata` (port the glue's `_bootstrap_resample_bank`
   resolution **exactly** — this is the both-consumers rule: the design
   drives the resampler even when analytic Σ̂ is dead code). `n_obs` from
   the resample source (first stage `source_data` override if a transform
   pipeline declares one — port that branch). Then
   `banks.resample_indices(...)` → `banks.bootstrap_states(...)` →
   `_run_bootstrap(h, adapter, config, metadata, h_factory=compiled.h_factory)`.
5. Replicate failures: `_run_bootstrap` already records
   `n_boot_effective/n_boot_failed`; R3 adds the §6.7 thresholding —
   failure rate > 1% appends a note, > 5% a `SoundnessWarning`
   (constants `REPLICATE_FAILURE_NOTE/WARN`), recorded into
   `estimand_metadata["diagnostics"]`.
6. No other branches. `method` not in {delta, simulation, bootstrap} is
   unreachable (C1 validated) — `raise AssertionError` if hit.

### R3.2 Tests (`tests/test_engine_execute.py`)

- `test_no_fallback_attributes` — result dict from every method has
  `fallback_triggered is False` and `fallback_reason is None` on the new
  path, even for a high-κ estimand that legacy would have flipped
  (construct one: a near-boundary logit prediction; assert
  `method == "delta"` and `kappa` is recorded and large).
- `test_nondifferentiable_delta_refuses` — `compose=lambda p: jnp.where(...)`
  (non-differentiable) under method="delta" raises `CompileError` whose
  message contains the §6.1 steer text; **no warning, no silent sim
  result**.
- `test_kappa_recorded_not_steering` — same query, `method="delta"` vs
  legacy with `kappa_threshold=0.3`: legacy flips (warns), new engine
  doesn't; new κ equals legacy's computed κ.
- `test_survey_design_drives_resampler` — wiring with
  `steps.input(df, design=...)`, method="bootstrap": monkeypatch
  `_generate_resample_indices` to capture kwargs; assert
  `cluster_ids is design.psu` and `strata is design.strata`. **This is the
  regression test for the worst facade bug.** Same for `cluster=`/`block=`.
- `test_banks_replayed_across_queries` — two queries, one estimator:
  indices generated once (count via monkeypatch), draws once.
- Dual-run: extend `tests/anchor/` (see R6) — the executor's numbers are
  checked end-to-end there; unit tests here pin behavior, not numbers.

### R3 pitfalls

- The non-differentiable check must run **before** `_run_delta`, which
  has no such check (only the dispatch you're not calling does).
- `is_jax_differentiable` needs `beta = adapter.coefficients()` — call it
  once; don't re-trigger adapter work per query.
- Simulation draws bank keys off (seed, n_sim, Σ̂): if you compute Σ̂
  twice (e.g., once nonrobust, once HC1) you've broken the frozen-Σ̂
  doctrine — Σ̂ resolves once per estimator (R2.2), banks consume the
  frozen one.
- `_run_simulation` falls back to a Python loop on JAX tracer errors —
  that's an internal kernel mechanism, not a method fallback; leave it.
- Don't pass `bootstrap_config` user dicts through; the plan's `ci` and
  block_type fields are the only sources.

**Acceptance:** execute tests green + full gate. Commit
`R3: doctrine executor (no fallbacks)`.

---

## R4 — The real result (`_result/_graphresult.py` rebuilt + `_result/_intervals.py`)

**Goal:** a self-contained result object; interval math reviewed-then-moved
out of `MarginsResult`.

**Read first:** `_result/_margins.py` `conf_int:671–900`, `test:903`,
`joint_test:1025`, `influence:1648`, `_summary_rows:185`, `to_frame:379`;
`_result/_pooling.py`; req §6; design §4.3 (level doctrine), §4.7 (sup-t).

### R4.1 `_result/_intervals.py` (review-then-move)

Lift these bodies from `MarginsResult` (with docstrings; cite original
line in the module header), as pure functions:

```python
def wald_interval(estimate_inf, se_inf, level, phi=None) -> (lower, upper)
def delta_interval(estimate_inf, gradient, cov_params, level, phi=None)
def draws_interval(draws_inf, level, phi=None, ci_method="percentile",
                   bootstrap_extras=None)            # percentile/basic/bca/studentized
def supt_interval_draws(draws_inf, estimate_inf, se_inf, level, phi=None)
def supt_interval_delta(estimate_inf, gradient, cov_params, level, phi=None)
def bonferroni_level(level, k) -> float              # 1 - (1-level)/k
def sidak_level(level, k) -> float                   # level ** (1/k)
def wald_test(estimate_inf, gradient, cov_params, *, null_value, alternative)
def draws_test(estimate, draws, *, null_value, alternative)   # from _dispatch.run_test
def joint_wald(estimates_inf, gradient, cov_params, *, null_value)
```

Review gate while moving: each formula traced (Wald: standard; sup-t via
draws: max-|t| quantile — Montiel Olea–Plagborg-Møller 2019 convention,
matches the legacy docstring; sup-t delta: MVN equicoordinate quantile;
BCa: Efron 1987 — confirm the legacy acceleration uses the jackknife unit
consistent with §6.5's "BCa × cluster" row, and ledger if it doesn't).
Oracle coverage: `wald_interval`/`wald_test` against scipy closed forms in
the analytic suite; sup-t against `multcomp::glht` max-t **when multcomp
is installed** — otherwise mark the sup-t oracle case *(deferred — R install)*
in the ledger and rely on the moved-code dual-run equality.

Legacy `MarginsResult` is **not edited** — it keeps its own copies until
R7 deletes them.

### R4.2 `GraphResult` (true object, no wrapper)

```python
@dataclass
class GraphResult:
    """Self-contained result. Req §6, design §7.1. Added in 0.4.0 (R4)."""
    estimate: np.ndarray            # reporting scale
    std_error: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    labels: list[str] | None
    method: str                     # resolved
    level: float                    # declared (locked)
    ci: str
    scale: str
    at: str
    plan: Plan                      # immutable copy
    population_note: str | None
    n_obs: int
    estimand_metadata: dict
    # diagnostics
    kappa: np.ndarray | float | None
    delta_sim_disagreement: float | None
    n_boot_effective: int | None
    n_boot_failed: int | None
    # per-method payload (inference scale where noted)
    gradient: np.ndarray | None     # delta
    cov_params: np.ndarray | None   # delta (frozen Σ̂)
    draws: np.ndarray | None        # sim/boot, reporting scale
    draws_inf: np.ndarray | None    # inference scale
    psi_h: np.ndarray | None        # tier-1 ψ^h = adapter.influence() @ ∇h
    ci_method: str | None
    bootstrap_extras: dict | None
    phi: Callable | None            # named-scale callables reconstructible
    phi_inv: Callable | None        #   from plan.scale; customs ride along
```

Constructor: `GraphResult.from_engine(result_data, *, plan, labels,
population_note, n_obs, psi_h, phi, phi_inv)` mapping the G1.3 dict.
`psi_h` is computed by the **noun** at query time when
`adapter.influence()` is not None and `gradient` is present:
`psi_h = np.asarray(adapter.influence()) @ np.asarray(gradient).T`
(the W1.3-pinned identity — `tests/test_influence_contract.py` already
asserts the underlying identity; add `test_psi_h_variance_identity`:
`psi_h.T @ psi_h ≈ ∇hᵀ Σ̂ ∇h` on a nonrobust OLS case, rtol 1e-8).

Methods (doctrine surface — signatures frozen):

- `conf_int(correction=None)` — None → stored arrays; `"bonferroni"` /
  `"sidak"` → recompute via `_intervals` at the allocated level (delta:
  `delta_interval`; draws: `draws_interval`); `"sup-t"` → the sup-t
  functions. Define `conf_int(self, correction=None, **dead)` **without**
  a `level` parameter; add an explicit guard:
  `if "level" in dead: raise TypeError(LEVEL_LOCKED_MSG)`; any other key
  in `dead` → standard TypeError. Pinned text:

  > `conf_int() takes no level=. The confidence level is declared at the
  > estimator constructor (level=<x> in this plan) and is part of the
  > pre-registered analysis. To report at a different level, declare a new
  > estimator (the recompute is cheap; the new plan hash is the point).
  > Family corrections — conf_int(correction="bonferroni"|"sidak"|"sup-t")
  > — allocate the declared budget and only widen.`

- `test(value=0, *, null_scale="reporting")`, `joint_test(value=0, *,
  kind="wald")` — port semantics from `MarginsResult` via `_intervals`
  (null on reporting scale mapped with `phi_inv` exactly as legacy does —
  read `test:903` for the mapping before porting).
- `summary()` — port `_summary_rows` + table rendering, simplified (no
  fallback fields); footer: `plan <hash> | population: <note> | κ = <x>`.
- `to_frame()/to_latex()/to_html()` — port the legacy formatting bodies.
- `outcome(index)` — port `_slice_by_outcome` over the stored arrays.
- `scaled(by, units="")` and `contrast(C, labels=None)` /
  `pairwise_contrasts()` — port from `MarginsResult` (linear ops over the
  stored payload; delta: `C @ gradient`-pushforward; draws: `draws @ C.T`;
  read the legacy bodies and move them).
- `influence()` — return stored `psi_h`; if None and BCa extras carry
  jackknife values, return those (legacy `influence():1648` branch);
  else raise `ValueError` naming the tier and steering to bootstrap.
- `to_disk(path)` / `from_disk(path)` — pickle the dataclass dict +
  `{"format_version": 1}`; **no `format=` param**. Custom (non-named)
  `phi` that fails to pickle → raise with the unhashable-callable
  explanation, don't silently strip.

**Sequencing (R4 lands before R6):** the facade still calls
`GraphResult._from_margins_result(result, plan)`. Keep that classmethod on
the new dataclass as an interim adapter (maps `MarginsResult` fields —
estimate/std_error/CIs/gradient/cov_params/draws/draws_inf/kappa/… — onto
the new fields; docstring marked `interim — removed at R6`), so the facade
and `tests/anchor/` stay green through the window.

### R4.3 `pool_imputations` re-point

Duck-type the input check to accept any object with
`estimate/std_error/phi/phi_inv/level/method` (both result types satisfy
it during the window); output type switches to `GraphResult` only at R6
when the new result is the only one the new surface emits — keep returning
its input type's class via a small factory hook so legacy tests stay green
until R7. `_rubin_pool` arithmetic untouched (citation-reviewed: Rubin
1987; Barnard–Rubin 1999 df — already in docstrings).

### R4.4 Tests

`tests/test_graphresult.py` (rewrite): construction from a delta dict and
a sim dict; `test_conf_int_level_typeerror` (message contains
"declared at the estimator constructor"); `test_corrections_only_widen`
(bonferroni/sidak/sup-t intervals ⊇ uncorrected, parametrized
delta/sim); `test_supt_delta_vs_draws_consistency` (same toy estimand,
sup-t from draws ≈ sup-t from MVN at rtol 0.05 — MC tolerance, slow-mark
if B large); `test_roundtrip_disk` (every field equal after
to_disk/from_disk; arrays via array_equal); `test_no_session_reference`
(`pickle.dumps` succeeds; no weakrefs in the object graph);
`test_psi_h_variance_identity`. `tests/test_intervals.py`: each function
against scipy-computed expectations (z quantiles, chi2 for joint Wald).

### R4 pitfalls

- Legacy `conf_int(level=…)` recomputes from gradient/draws — the
  *machinery* is kept (corrections need it); only the public `level=`
  door is removed. Don't delete the recompute paths.
- Sup-t over draws uses `draws_inf` (inference scale) with the se on the
  same scale, then back-transforms — read the legacy body; mixing scales
  here is silent-wrong.
- `phi` on the result must be the *same object* the engine used —
  reconstruct from `plan.scale` for named scales; never re-derive a
  callable pair from user kwargs at result time.
- `dataclass` with ndarray fields: implement `__eq__`-free comparisons in
  tests (compare field-by-field); don't add `eq=True` semantics that
  numpy breaks.

**Acceptance:** result + intervals tests green; full gate. Commit
`R4: self-contained GraphResult + interval math moved`.

---

## R5 — Compile C2 for real + soundness spine

**Goal:** the constructor pipeline earns the "pre-registration" name.

**Read first:** current `_graph/_compile.py` (you are replacing most of
it), req §3–§4, design §4.3/§4.5/§5.2, `_kappa.py` `session_kappa:283` and
`delta_simulation_disagreement`, design §6 tables (all rows).

### R5.1 `compile()` rebuilt

Signature (explicit; **delete `**extra` and `mode=`** — there is no
legacy mode):

```python
def compile(wiring, outcome, *, at="overall", scale="response",
            method="delta", vcov=None, ci=None, level=0.95, B=0,
            n_sim=0, seed=None, weights=None, gradient_backend="autodiff",
            fd_step=1e-6, constants_overrides=()) -> tuple[Plan, CompileReport, Compiled]
```

`Compiled` (new, internal) carries what the noun needs beyond the Plan:
`adapter`, `wiring_facts`, `base_data`, `frozen_cov`, `phi/phi_inv`.

Pipeline order (each numbered step is testable):

1. **C1 structural:** unknown-kwarg strictness is now the signature
   itself; validate `at`/`scale`/`ci`/`method` values; walk the wiring —
   unknown node kind ⇒ `CompileError` (never skip); `match` node together
   with any `alters_rows` filter stage ⇒ the 0.4.0 refusal (steer text:
   `"match + row-filter stages in one wiring lands with the fan engine in
   0.5.0; today, apply filters before matching outside the wiring or use
   matching alone"`); fan nodes ⇒ refusal naming 0.5.0.
2. **WiringFacts extraction** (replaces `_extract_legacy_kwargs`, but
   validating): walk once, collect `design/cluster/block` from the input
   node, matcher payload, ordered transform stages. **Stage order = wiring
   topological order from the input outward** — write
   `test_transform_order_matches_wiring` (the facade got this wrong and
   needed `reversed()`; don't inherit the iteration-order accident: derive
   order by following `inputs` edges from the root, not by `_flatten_graph`
   stack order).
3. **vcov_spec resolution** (G1.3 rule) — explicit `vcov=` + a survey
   design is a conflict ⇒ `CompileError` telling the user the design
   already determines Σ̂ (unless `vcov` *is* the survey dict). Cluster
   declared at input + string vcov "cluster" → normalize as legacy does.
4. **C2 data:** `base_data = wiring.collect()` (point execution);
   template-vs-wiring fingerprint check — **no try/except skip**: if the
   wiring can't collect, that's a `CompileError`, and a fingerprint
   mismatch refuses naming both fingerprints (current code's
   `except NotImplementedError: wiring_fp = None` is the
   exception-swallowing path the plan kills — delete it).
5. **Σ̂ freeze:** `frozen_cov = adapter.covariance(vcov_spec)`.
6. **`method="auto"` resolution** (decide once; record reason):
   - differentiability probe: build the posture estimand (a prediction
     query via R2 on the template) and run `is_jax_differentiable`;
   - tier-1/autodiff: κ pre-pass via `session_kappa(h_factory, beta,
     frozen_cov, representative_design)`; resolve **delta** iff
     differentiable and worst-case κ ≤ `KAPPA_BORDERLINE` (0.3), else
     **simulation** with reason `"auto: posture κ=<x> > 0.3"` (pinned —
     Appendix C #3);
   - tier-2 (FD): `delta_simulation_disagreement` on the template ≤ 5%
     (`DISAGREEMENT_WARN`) → delta, else simulation (design §11.8);
   - bootstrap is **never** auto-resolved (design §5.2.4).
7. **Adequacy predicates** (existing `_soundness` functions, now actually
   fed): tail counts (B, level, ci), cluster count G, lonely PSU (the
   design object now *flows* — the current compile already calls it, keep),
   ESS when weights present (add `check_ess` to `_soundness/_predicates.py`
   per §6.7: ESS = (Σw)²/Σw², ESS/n < 0.5 → note).
8. **ci defaults per method** (design §4.3): `ci=None` resolves to
   `"wald"` (delta/sim) / `"percentile"` (bootstrap); explicit values
   validated by `check_ci_method_compatibility`.
9. Population notes joined; Plan built; `report.raise_for_refusals()`;
   warnings emitted once.

Plan gains fields at R5: `gradient_backend`, `fd_step`, `weights_fingerprint`
(hash of the weights array or None) — they are analysis-defining
(Appendix C #4). Hash recipe stays `1` (it has never shipped).

`resolve_scale` callables: fingerprint by `inspect.getsource` hash when
retrievable, else qualname, else set `unhashable_callable=True` (the Plan
field exists).

**Sequencing (R5 lands before R6):** the facade calls
`compile(wiring, outcome, …, mode="doctrine", **kwargs)` and unpacks a
2-tuple. Update the facade's call site in the same commit (drop `mode=`,
unpack the 3-tuple, ignore `Compiled`) — `estimators/_base.py` is interim
scaffolding, not frozen legacy, so editing its call site does not violate
I6. The facade dies at R6 regardless.

### R5.2 `SOUNDNESS_ROWS` registry

```python
@dataclass(frozen=True)
class SoundnessRow:
    id: str                  # e.g. "6.1-nondiff-compose-delta"
    design_section: str      # "§6.1"
    severity: str            # "unrepresentable"|"refuse"|"warn"|"note"|"sound"
    text: str                # verbatim from the design table (steer included)
    predicate: str | None    # qualname of the implementing check, or None

SOUNDNESS_ROWS: tuple[SoundnessRow, ...] = (...)
```

Enumerate **every** row of design §6.1–§6.5 tables (read them; ~30 rows)
plus the §6.6 quantitative roster. Implemented in 0.4.0:
`6.1-method-unsupported`, `6.1-nondiff-compose-delta`,
`6.1-ci-method-incompatible` (studentized row), `6.5-lonely-psu`,
`6.5-few-clusters`, `6.7-tail-counts`, `6.7-bca-b`, `6.7-se-b`,
`6.7-replicate-failures`, `6.7-ess`, `6.1-match-filter-04-refusal`
(this one cites the plan, not the design). Everything else:
`predicate=None` (lands 0.5.0/0.6.0). Texts with steers at unshipped
machinery carry *(future)* verbatim.

`tests/test_soundness_predicates.py` grows: iterate the registry —
implemented rows resolve their qualname and have ≥1 test asserting
severity + text substring; `None` rows assert the *(future)* discipline
(no predicate, text present). Add the req-§1 lattice-consistency test:
parametrize over the adapter registry; tier (has `influence()`/score vs
FD vs bootstrap-only) consistent with `supported_inference_methods`.

### R5.3 Tests

`tests/test_compile.py` (rewrite): one test per pipeline step above —
notably `test_unknown_kwarg_typeerror` (`compile(w, m, kapa_threshold=1)`
→ TypeError naming valid kwargs), `test_unknown_node_kind_refuses`,
`test_template_mismatch_refuses_names_both_fingerprints`,
`test_no_fingerprint_skip_path` (a wiring whose collect raises ⇒
CompileError, not silent pass), `test_vcov_survey_conflict`,
`test_auto_resolves_delta_low_kappa` / `_simulation_high_kappa` (toy
near-boundary posture), `test_auto_reason_recorded`,
`test_match_plus_filter_refused`, `test_ci_defaults_per_method`,
`test_plan_hash_golden` — build the fixed toy plan (hand-written field
values, no data), assert `plan.plan_hash == "<recorded constant>"` with
the recipe-bump rule in the docstring (changing the recipe requires
bumping the `@1` suffix and this constant in one commit with a recorded
justification).

### R5 pitfalls

- The current `_resolve_outcome` swallows `NotImplementedError` around the
  wiring fingerprint — the single worst pattern in the interim code;
  your rewrite must make absence-of-check impossible, not just unlikely.
- `session_kappa` needs a `representative_design` (list of covariate
  rows) — build it from `base_data` like `diagnose()` does
  (`_session.py:1415`; read it).
- `constants_overrides` must land in the Plan (they're in the hash —
  §6.7 "overrides are recorded in the plan hash").
- `_DataFingerprintAdapter` (current compile) instantiates an abstract
  hack — replace with a standalone `fingerprint_frame(df)` function
  refactored out of `ModelAdapter.data_fingerprint`.
- Plan must stay JSON-serializable: no DataFrames, no callables, no
  ndarray in Plan fields (fingerprint them).

**Acceptance:** compile + soundness tests green; full gate. Commit
`R5: real C2 + soundness registry`.

---

## R6 — Nouns on the new engine

**Goal:** `GComputation` becomes `compile → Plan → BankSet → R2/R3 → R4`.
No `Margins` import anywhere under `pymargins/` (I6 grep is a test).

### R6.1 Constructor

Frozen signature + the req-§7 additions:

```python
class GComputation:
    def __init__(self, wiring_or_model=None, *, outcome=None,
                 at="overall", scale="response", method="delta",
                 vcov=None, ci=None, level=0.95, B=0, n_sim=0,
                 seed=None, weights=None, adapter=None,
                 gradient_backend="autodiff", fd_step=1e-6,
                 n_jobs=1, progress_bar=False):
```

(no `**kwargs`; `n_jobs`/`progress_bar` are execution knobs, not Plan
fields). Wiring/outcome disambiguation as the facade does (positional
`Node` without `outcome=` refuses; positional model = implicit input from
template training data, **no properties** — design §4.5 fence). Spec-form
outcome:

```python
if isinstance(outcome, str):
    data = wiring.collect()
    family = ...   # new kwarg? NO — pinned: spec form is
                   # outcome=("y ~ x1 + x2", family) a 2-tuple, or a plain
                   # string for OLS. (Appendix C #5)
    fitted = smf.glm(formula, data=data, family=family).fit(tol=1e-12) \
             if family is not None else smf.ols(formula, data=data).fit()
    adapter = auto_detect_adapter(fitted, formula=formula, data=data)
```

Then: `plan, report, compiled = compile(...)`; `self._banks =
BankSet(plan.plan_hash, 0, seed)`; build `QueryContext` from `compiled`;
store nothing session-like.

### R6.2 Query methods

Each is three lines: build `QuerySpec` from kwargs → `compile_query` →
`execute_query` → `GraphResult.from_engine(...)` (computing `psi_h` when
tier-1 delta). `joint(*results)` raises
`NotImplementedError("joint() lands in 0.5.0")` (unchanged).

### R6.3 req-§7 row-by-row audit (acceptance checklist)

Work through the table; every row gets a line in the R6 commit message:

| row | disposition to verify |
|---|---|
| `Margins(model)` → `GComputation(model)` | anchor path works |
| `phi/phi_inv` → `scale=` named or callable pair | callable pair test |
| `vcov` → unchanged (str/dict/ndarray; ndarray ⇒ tier-2 demotion) | test all three |
| `weights` → `weights=` (known weights only) | added this WS |
| `at/level/method` → constructor-bound | already |
| `kappa_threshold` → `constants_overrides` | override changes plan hash |
| `rng_seed/n_sim/n_boot` → `seed/n_sim/B` | already |
| `n_jobs/progress_bar` → execution options | not in Plan |
| `gradient_backend/fd_step` → engine options | added this WS, in Plan |
| `diagnostics` → gone (always on) | passing it ⇒ TypeError |
| `cluster/block_size` → `steps.input(cluster=, block=)` | routed both consumers |
| `bootstrap_config` → `ci=` + `steps.input(block_type=)` | dict ⇒ TypeError |
| `matching=` → `steps.match` | works |
| `transforms=` → `steps.*` | order test |
| `survey_design=` → `steps.input(design=)` | routed both consumers |
| `formula=/data=` → spec-form `outcome=` | **lands here, mandatory** |
| `strict=` → gone | TypeError |
| `adapter=` → unchanged | added this WS |
| `from_posterior` → dropped | release-notes row only |
| queries | all eight + wtp |
| `diagnose()` → `est.plan.describe()` + CompileReport | describe shows κ pre-pass when auto |
| scale classmethods → `scale="log"` etc. | covered by resolve_scale |
| `pool_imputations` | works on GraphResult (R4.3) |
| `adjust` | accepts GraphResult (duck-typed p-values — verify) |

### R6.4 Anchor matrix expansion (`tests/anchor/`)

Now the dual-run earns its keep. Rewrite `test_anchor_gcomputation.py`
into a parametrized matrix:

- fixtures: OLS, logit GLM, Poisson, probit (statsmodels); WrappedFD
  fixture (lifelines or linearmodels if installed — I2 coverage);
- methods × queries × postures: delta/sim/boot × predict/dydx/contrasts/
  evaluate × {plain, scale="log" (Poisson), weights, cluster-boot,
  block-boot, survey (Σ̂ + boot), matching, transforms(trim)};
- assert `np.testing.assert_array_equal` on estimate/SE/CI lower+upper,
  and draws when stochastic;
- on failure emit the localization diagnostic — implement once in
  `tests/anchor/conftest.py`:

```python
def assert_anchored(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        diff = np.max(np.abs(a - b)) if a.shape == b.shape else "shape-mismatch"
        raise AssertionError(
            f"[anchor:{name}] max|a-b|={diff} dtypes=({a.dtype},{b.dtype}) "
            f"shapes=({a.shape},{b.shape}) strides=({a.strides},{b.strides})")
```

- cells where R1 ledgered a legacy defect: decorate
  `@pytest.mark.xfail(strict=True, reason="D<n>: legacy defect — oracle
  case <id> is authoritative")`.

### R6 pitfalls

- The four audit silent-wrongs all lived between constructor kwargs and
  the engine: survey/cluster dropped, transform order inverted, κ-flip
  alive, `conf_int` crash. Each now has a named test (R3/R5/R4) — run
  them first when an anchor cell goes red.
- The implicit-input fence: `GComputation(model)` then `vcov={"type":
  "survey", ...}` is *allowed* (vcov is Σ̂ spec); `design=` only exists on
  `steps.input`. Don't add convenience kwargs.
- Spec-form must fit on the **wiring output** (`collect()`), not on the
  template's training data.
- `est.plan.describe()` replaces `diagnose()` — make sure the κ pre-pass
  result (when `method="auto"`) is in the Plan's
  `method_resolution_reason`, because describe() prints it.

**Acceptance:** anchor matrix green (minus ledgered xfails) + **oracle
suites now pass through the new engine** (remove the R1-era facade xfails
that the rewrite fixed; each removal cites its D-entry) + req-§7 checklist
in the commit message + full gate. Commit
`R6: GComputation on the real engine (req §7 audit complete)`.

---

## R7 — Test porting, parity gate, deletion

Measured scope (2026-06-11): 68 of 91 test files reference `Margins`.

### R7.1 Mechanical translation table

| legacy spelling | new spelling |
|---|---|
| `Margins(model, …)` | `GComputation(model, …)` |
| `rng_seed=` | `seed=` |
| `n_boot=` | `B=` |
| `bootstrap_config={"ci_method": x}` | `ci=x` |
| `bootstrap_config={"block_type": x}` | `steps.input(df, block=k, block_type=x)` (Appendix C #10) |
| `phi=f, phi_inv=g` | `scale=(f, g)` or named |
| `Margins.log_scale(m)` | `GComputation(m, scale="log")` |
| `cluster=ids` / `block_size=k` | `steps.input(df, cluster=ids)` / `(block=k)` |
| `survey_design=d` | `steps.input(df, design=d)` |
| `matching=client` | `steps.match(inp, client)` |
| `transforms=[s1, s2]` | `steps.*` chain in the same order |
| `from_formula(model, f, data)` | `GComputation(steps.input(data), outcome=f)` |
| `strict=True` | (delete — doctrine) |
| `diagnostics=False` | (delete — always on) |
| `kappa_threshold=x` | `constants_overrides=(("kappa_borderline", x),)` only where the test is *about* the override; κ-flip tests are category (c) |
| `result.conf_int(level=x)` | category (c): re-declaration test or drop |
| `m.diagnose()` | `est.plan.describe()` / CompileReport assertions |
| `from_posterior` tests | category (c): dropped, ledger cites plan §0 |

### R7.2 Triage protocol

Work file-by-file (`grep -rl "Margins" tests --include="*.py"`). Per file:
classify each test (a)/(b)/(c) per the plan; (a) translate mechanically,
assertions untouched; (b) check the oracle matrix — covered ⇒ drop with
ledger entry; unique ⇒ re-anchor (add an oracle case or derive in-test
from statsmodels/scipy) or keep as `legacy-corroborated-regression` with
the ledger note; (c) per the table above. **Never regenerate an
expectation from the new engine.** A ported test that fails is: a
new-engine bug (fix it), a semantic change (recategorize to (c)), or a
ledgered legacy defect (update the assertion to the oracle-correct value,
citing the D-entry inline).

Adapter test files (`test_adapter_*`) mostly never touch `Margins` —
they're untouched. Files testing `MarginsResult` mechanics map to
`GraphResult`/`_intervals` equivalents (many already exist from R4).

### R7.3 Parity gate

All of: `tests/oracle` green through the nouns; ported suite green;
anchor matrix green except ledgered xfails; `pytest -m slow` green once
(run it manually — it gates the tag); ruff green.

### R7.4 Regression goldens + deletion (one commit sequence)

1. `tools/record_goldens.py`: runs the anchor matrix cells through the
   **new engine** and writes `tests/golden/<cell_id>.npz`
   (estimate/SE/CI/draws arrays, exact binary) + `manifest.json`
   (cell id → constructor spelling, package version, date). Guard rails
   in the script: refuses to overwrite without `--force`, and `--force`
   prints the ledger reminder.
2. `tests/golden/test_regression_goldens.py`: parametrized
   `np.array_equal` comparisons (the layer-4 suite). Sanity-check 2–3
   cells by hand against oracle values before committing.
3. Delete `tests/anchor/` (retired, not converted).
4. Delete legacy: `pymargins/margins/` (entire package),
   `pymargins/_result/_margins.py`, the `run_inference` fallback body in
   `_inference/_dispatch.py` (keep/move `run_test` math if R4 imports it —
   otherwise delete the module), `_run_delta`'s κ-flip branch
   (`_delta.py:36–53` — now provably dead; deleting it is the one
   permitted kernel edit, with the anchor… now golden suite proving
   no-op), legacy exports from `pymargins/__init__.py` (`Margins`,
   `MarginsResult`, `DiagnosticResult`, scale classmethod docs) and from
   `pymargins/_result/__init__.py`.
5. Relocations: nothing moves into `margins/`'s place — by now
   `_engine/_queries.py` owns the builders; root `_estimands.py`/
   `_scenarios.py` stay where they are. Rename `_engine/_seeds.py`
   wrappers to drop `legacy_` (`resample_indices`, `sim_draws`) and fix
   imports.
6. Full suite + ruff + `pytest -m slow` green. Commit sequence:
   `R7: record regression goldens`, `R7: port suite (ledger: <n> entries)`,
   `R7: delete legacy orchestration`.

### R7 pitfalls

- Deleting `margins/` breaks `pymargins/__init__.py` first — update
  exports in the same commit, run `python -c "import pymargins"` before
  pytest.
- `_result/__init__.py` re-exports `MarginsResult` and `pool_imputations`
  — pooling stays, result goes.
- Some docs/demo files import `Margins` — `grep -rn "Margins" docs demo`
  and stub/update them (R8 rewrites docs; don't leave broken imports).
- npz goldens: save with `np.savez` (not `savez_compressed` — exactness is
  unaffected by compression, but keep it simple and diffable in size);
  load with `allow_pickle=False`.
- The dead-code κ-flip deletion must be its own tiny commit so the golden
  suite isolates it.

---

## R8 — Docs and the 0.4.0 release

- `CHANGELOG.md` rewritten as a breaking release. Skeleton:

  ```
  ## 0.4.0 — <date>  — BREAKING: the Margins session is removed
  ### Removed
  - Margins, MarginsResult, from_posterior (rationale: design §3.9) …
  ### Migration
  <the req §7 table, rendered>
  ### Corrections (numbers that changed on purpose)
  - D<n>: <what was wrong in ≤0.3.0> — oracle evidence: <case id>  …
  ### Reproducibility
  - Same-seed simulation/bootstrap draw streams may differ from 0.3.0
    (seed-tree ownership moved to the engine; rev. 2 amendment to verdict 5).
  ### Added
  - GComputation, steps.*, Plan (pre-registration + plan hash), oracle
    validation suite (analytic + R marginaleffects/survey goldens), …
  ```

- Docs: expand `docs/explanations/computation_graph.md` (design §2
  condensed + "how pymargins is validated" — the §4 oracle stack);
  `docs/tutorials/graph_quickstart.md` (worked examples 8.1–8.3 from the
  design); plan/pre-registration guide (level/ci doctrine, the
  `conf_int(level=)` TypeError explained positively); rewrite
  `kappa_fallback.md` (decide-once); mark `session_precommitment.md`
  superseded. Check every doc snippet executes (myst-nb runs them).
- Weekly slow lane content (now real): coverage sims per req §9 +
  sim-vs-delta calibration cases; optional `tools/oracle/check_drift.R`
  rerun comparing fresh R output to committed goldens.
- Release gates: full suite + golden suite + oracle suite + ruff +
  `pytest -m slow` green at the tag; CHANGELOG corrections section
  matches the defect ledger 1:1.

---

## G3. The facade post-mortem — failure modes this guide is engineered against

1. **Translation-layer silent drops.** The facade walked the graph and
   `if`-matched known params; unknown ones vanished (survey design +
   cluster — silently wrong SEs). Rule: validation by construction —
   unknown kind/key/kwarg raises (R5 step 1–3).
2. **Order accidents.** `_flatten_graph` yields stack order; the facade
   compensated with `reversed()` — a coincidence, not a contract. Rule:
   derive order from edges (R5 step 2 + its test).
3. **Doctrine as patches.** κ-flip "disabled" by passing `inf` *through
   the legacy session* while `run_inference`'s non-differentiable reroute
   stayed live. Rule: the new path simply has no branch to disable
   (R3.1.6); the kernels' dead branches are deleted at R7 with golden
   proof.
4. **Wrapper results.** `GraphResult` held a live `MarginsResult` (with a
   session weakref) — `to_disk` was lossy-by-luck. Rule: dataclass of
   arrays; `test_no_session_reference`.
5. **Exception-swallowing checks.** The template fingerprint check
   silently skipped when collect raised. Rule: a check that can't run is
   a refusal (R5 step 4).
6. **Done-claims without gates.** CHANGELOG claimed W2.8 complete; it
   wasn't. Rule: gates in commit messages; the R6 commit carries the
   req-§7 checklist inline.

## Appendix A — golden JSON schema

Formal field list = §R1.4. Required: `case_id`, `created`, `r_version`,
`packages`, `data`, `model{formula,family,fit_control}`, `r_call`,
`vcov`, `ci_convention{dist,level}`, `quantities{coefficients,estimate,
std_error}`. Optional: `labels`, `quantities.conf_low/conf_high/
vcov_matrix`, `tolerances` (per-quantity overrides; presence = ledger
entry), `notes`.

## Appendix B — ledger templates

In §G2. Ledgers are append-only; an entry is never edited, only followed
by a superseding entry.

## Appendix C — decisions pinned by this guide (not by the plan)

Visible so the user can overrule; each carries its rationale. If any
proves wrong mid-build, stop and ask — don't silently deviate.

1. **Oracle data convention:** datasets generated in Python, committed as
   CSV, read by R — bit-identical via shortest-roundtrip float repr.
   (Avoids cross-language RNG entirely.)
2. **Fit-alignment gate:** every R golden stores R's β̂; Python tests gate
   on `TOL_COEF=1e-8` before comparing effects. (Separates misalignment
   from defects — plan trap 6 made mechanical.)
3. **`method="auto"` κ rule:** resolve delta iff differentiable and
   worst-case posture κ ≤ `KAPPA_BORDERLINE` (0.3); tier-2 via
   disagreement ≤ 5%. (Mirrors the legacy runtime threshold default,
   moved to compile time; design §5.2/§11.8 name the mechanism but not
   the constant.)
4. **`gradient_backend`/`fd_step`/weights-fingerprint enter the Plan**
   (they change numbers ⇒ analysis-defining).
5. **Spec-form outcome spelling:** `outcome="y ~ x"` (OLS) or
   `outcome=("y ~ x", sm.families.Binomial())` — no new `family=`
   constructor kwarg. (Keeps the constructor at the req-§7 surface.)
6. **Module homes:** `resolve_scale` and query builders in
   `_engine/_queries.py`; `WiringFacts` extraction in `_graph/_compile.py`;
   config building in `_engine/_queries.py`; bank wiring in
   `_engine/_execute.py`.
7. **Tolerance constants:** `TOL_COEF 1e-8 / TOL_EST 1e-6 / TOL_SE 1e-5 /
   TOL_CI 1e-5 / TOL_ANALYTIC 1e-10`, rtol-only, atol per-case-by-ledger.
8. **Regression-golden format:** `.npz` per matrix cell + json manifest;
   recorder refuses overwrite without `--force`.
9. **Seed-derivation goldens are literals pasted into the test** after one
   verified run (small sizes), not external files. (Reviewable in diff.)
10. **`block_type` declares at `steps.input(block=, block_type="moving")`**
    and flows through wiring facts into the engine config. (Req §7
    dissolves `bootstrap_config` but gives block *type* no home; design
    §4.1 says the input node owns ALL dependence declarations — block
    type is a property of the block declaration. `steps.input` gains the
    kwarg at R5/R6; `_graph` records it in node params so it is
    plan-hashed. The `bca` `acceleration` override has no new home —
    ledger it if a ported test needs it.)

## Appendix D — Post-audit forward checklist (R1 → R2)

Items recorded by the R1 audit that do not block R2 but must be resolved in
later workstreams:

1. **Survey aggregation convention (R6).** The two survey goldens currently
   pin the *unweighted* estimand (legacy's default warning is in force). When
   `steps.input(design=)` lands at R6, decide whether the survey posture
   weights the aggregation and add a weighted twin golden if needed.

2. **D4 regression hook (R5/R6).** The cluster oracle tests were re-spelled to
   the explicit `vcov={"type":"cluster","groups":g}` spelling; nothing now
   exercises the broken `cluster=` declaration path. Add a
   cluster-declared-at-`steps.input` oracle or anchor case proving the new
   engine's `vcov_spec` resolution does not reproduce D4.

3. **Re-point Margins-direct oracle tests at R6.** Four tests still call
   `Margins` directly because `GComputation` lacks `weights=` / `survey_design=`
   routing; they are marked with `TODO(R6)` comments in
   `tests/oracle/test_analytic.py` and `tests/oracle/test_r_golden.py`.

## R2 audit follow-up

2026-06-12. R2 workstream (`dfcad23`) passed functional parity but needed
process and coverage repairs. This section records the findings and the
follow-up commit.

### Citation review (I3″)

Root `_estimands.py` and `margins/_atoms.py` were read and traced during the
port. No formula defects were found; the atom kernels match their docstring
formulas. The review produced the following legacy-behavior findings, recorded
as defect-ledger entries D12–D15.

### Findings fixed in the follow-up commit

- F4 `build_inference_config`: `n_sim=plan.n_sim or 4000` and
  `n_boot=plan.B or 1000` were replaced with pass-through values; a
  `# TODO(R5)` marks `gradient_backend`/`fd_step` placeholders pending Plan
  fields (appendix C #4).
- F5 Multi-estimand `label=`: restored the legacy `UserWarning` in
  `compile_query` when `atexog` or `over` produces multiple estimands.

### Findings recorded for R5/R6 decision

- **D12 — `weights=` + `over=` crash (legacy and new).** Full-length weights
  are not subset per over-group in `_build_prediction_query` (and legacy),
  producing a shape-mismatch `TypeError`. Fix requires oracle anchoring of the
  intended weighted-group semantics.

- **D13 — `contrasts()`/`evaluate()` ignore declared weights in per-scenario
  aggregation.** `scenario_weights` are never passed to
  `make_linear_combination_estimand`/`make_evaluate_estimand`, so weighted and
  unweighted sessions yield identical contrast values. Connects to the R6
  survey-aggregation convention.

- **D14 — 2D contrast-matrix normalization has no home.** Legacy session
  normalizes matrix/list-of-lists contrasts into named dicts; the new builder
  receives raw weights. Numbers are correct (`jnp.dot` handles 2D), but labels
  are `['contrast']` for a k-row estimand — a silent label/shape mismatch for
  R4's result wrap.

- **D15 — WTP spelling deviation.** `wtp()` composes slope estimands at the h
  level rather than literally through `make_evaluate_estimand`; this satisfies
  the design intent (no result-level division) but is an undeclared deviation
  worth one ledger line.

### D12–D14 decisions (2026-06-12, user verdicts)

All three stop-and-asks from the R2 audit were decided and implemented in the
new engine only (legacy stays frozen until R7):

- **D12 → weighted mean within group** (ledger D16). `weights=` + `over=`
  subsets the per-observation weights to each group's rows positionally;
  the group estimand is the weighted group mean, matching `marginaleffects`
  `by=`+`wts=` and Stata `margins, over() [pw=]`. R golden
  (`avg_predictions(by =, wts =)`) lands with the R5/R6 case matrix.
- **D13 → contrasts()/evaluate() honor declared weights** (ledger D17).
  Per-scenario aggregation is weighted, consistent with predict/dydx and
  `avg_comparisons(wts =)`. Changes shipped 0.3.0 numbers → R8 Corrections.
  Non-aligned scenario rows (data-override/grid) under weights= refuse with
  a clear ValueError.
- **D14 → normalization in the builder, now** (ledger D18). Matrix /
  list-of-lists / dict / vector contrast forms normalize and validate in
  `_build_contrast_query`; labels are correct before R3/R4 consume them.
- **Appendix D.1 DECIDED: survey design weights the aggregation.** When
  `steps.input(design=)` lands at R6, design weights drive estimand
  aggregation (population-representative), matching `marginaleffects`'
  automatic use of svyglm weights and Stata `svy: margins`. Weighted twin
  goldens land at R6; the existing unweighted survey goldens become
  explicit-unweighted corroboration or retire (ledger entry at R6).

## R3 completion note

2026-06-12. R3 workstream implemented.

- New module: `pymargins/_engine/_execute.py` with `execute_query` — the
  doctrine dispatch/executor entry point.
- Behavior pinned:
  - `method = plan.method_resolved`; no recomputation, no fallback branches.
  - delta: `is_jax_differentiable` probe first; non-differentiable estimand
    raises `CompileError` steering to `method="simulation"` (§6.1).
  - delta: `kappa_threshold=float("inf")` ⇒ κ is recorded but never steers.
  - simulation: `banks.sim_draws` injected into `InferenceConfig.sim_draws`.
  - bootstrap: resampling declaration resolved from `wiring_facts`
    (matching → cluster → survey-design PSU/strata); `BankSet` indices and
    states injected into `config` before `_run_bootstrap`.
  - bootstrap: replicate-failure rate thresholding via
    `REPLICATE_FAILURE_NOTE` / `REPLICATE_FAILURE_WARN`; notes recorded in
    `estimand_metadata["diagnostics"]`, warnings emitted as `SoundnessWarning`.
  - unreachable method ⇒ `AssertionError`.
- `pymargins/_engine/_queries.py`: `build_inference_config` now accepts a
  pre-computed `frozen_cov` so Σ̂ is resolved once per estimator.
- Tests: `tests/test_engine_execute.py` — 12 tests covering no-fallback
  attributes, non-differentiable refusal, κ non-steering vs legacy,
  survey/cluster/block resampler routing, bank replay across queries, and
  replicate-failure warnings.
- Gate: `pytest -m "not slow" -q` — 1686 passed, 3 skipped;
  `ruff check .` — green.

### R3 audit fixes (post-commit)

2026-06-12. Seam issues found during R3 review; fixed before R5 wiring.

1. **n_sim=0 delta plan crash** — `_run_delta` now skips the
   delta-simulation disagreement diagnostic when `config.n_sim == 0` and also
   catches `IndexError` from `np.quantile`. Prevents the `compile()` default
   (`n_sim=0`) from crashing once R5 wires `compile()` to `execute_query()`.
2. **Metadata mutation on replay** — `_record_replicate_failures` now copies
   `estimand_metadata` before appending diagnostics, so re-executing the same
   compiled query no longer accumulates duplicate entries on the frozen query
   object.
3. **training_data access guard** — `_resolve_resample_source` mirrors the
   legacy glue's `(NotImplementedError, AttributeError, TypeError)` guard and
   raises the same bootstrap-specific `NotImplementedError` when no resample
   source is available.

Regression tests added to `tests/test_engine_execute.py` for issues 1 and 2.
Updated gate: `pytest -m "not slow" -q` — 1688 passed, 3 skipped.

### R3 audit follow-up: inference-budget invariant (root cause of issue 1)

2026-06-12. Issue 1 above patched the delta *symptom*. The root cause is that
the new plan layer dropped the legacy session's `n_sim >= 1` / `n_boot >= 1`
invariant (`margins/_session.py` validated both and defaulted them to
4000 / 1000). `compile()` and `GComputation` instead defaulted both to **0**,
so the simulation path (`n_sim=0` → empty draws → `IndexError`) and bootstrap
path (`B=0` → "All bootstrap replicates failed") carried the identical latent
crash, reachable once R5 wires `compile()` → `execute_query()`. The lie was
masked today only because `_extract_legacy_kwargs` forwards `n_sim`/`B` to the
session **only when > 0**, so the session silently substituted its own
defaults.

Fix: `compile()` now defaults `B=1000`, `n_sim=4000` (matching the legacy
session) and validates both as positive integers (`CompileError` otherwise);
`GComputation.__init__` defaults updated to match so the Plan records the
budget the engine actually uses. `Plan`/`build_inference_config` left as pure
pass-throughs (their pass-through tests rely on direct construction).
Tests: `tests/test_compile.py` — parametrized default-positivity and
non-positive rejection across `{delta, simulation, bootstrap}`.
