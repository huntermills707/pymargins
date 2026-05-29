# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Internal

- CI: GitHub Actions pipeline for lint (ruff check + format), a test matrix
  across Python 3.10–3.14 (trimmed to 3.10/3.12/3.14 on PRs), a bare-install
  job verifying optional backends stay lazily imported, a minimum-dependency-
  version job (`uv --resolution lowest-direct`), per-backend isolation runs,
  and a cache-backed docs build.
- CI: release workflow publishing to PyPI on `v*` tags via Trusted Publishing
  (OIDC, no stored token).
- Dependabot: weekly updates for GitHub Actions and pip dependencies.
- Repo-wide ruff lint/format pass and ruff configuration in `pyproject.toml`.

## [0.1.1] — 2026-05-26

### Added

- `Margins.from_posterior()` for constructing sessions from Bayesian posterior
  draw banks (MCMC samples treated as simulation draws).
- `MarginsResult.contrast(C)` for testing linear hypotheses `C @ theta = 0`
  on vector-valued results.
- `MarginsResult.influence()` exposing per-observation jackknife influence
  measures (DFBETA-style) for sensitivity diagnostics.
- `pymargins.adjust()` for multiple-comparison correction on result tables
  (Holm, Bonferroni, Benjamini–Hochberg FDR, and others via `statsmodels`).
- Elasticity convenience methods on `MarginsResult`: `eyex()`, `eydx()`,
  and `dyex()` for semi-elasticity and elasticity transformations.
- `MarginsResult.to_disk()` / `MarginsResult.from_disk()` for lightweight
  pickle-based persistence of result objects.
- `Margins.rmst()` survival convenience wrapper for restricted mean survival
  time contrasts.
- New how-to guide: `docs/howto/influence.md` covering jackknife influence
  diagnostics and leave-one-out sensitivity analysis.
- Expanded how-to guides for elasticities, simultaneous confidence intervals,
  and exporting results (Excel/pickle).

### Fixed

- Refreshed Jupyter notebook execution cache for ReadTheDocs builds.

## [0.1.0] — 2026-05-23

### Added

- **Session-level inference-distribution caching.** Bootstrap and
  simulation random objects are now materialized once per session and
  reused across every subsequent call:
  - **Bootstrap**: resample indices and refitted-model states are
    harvested at first bootstrap-method use; later calls evaluate the
    estimand over the cached states rather than re-fitting. A 10-point
  survival curve now costs ~1 bootstrap pass, not 10.
  - **Simulation**: β* draws are generated once and reused.
  - Sessions freeze inference parameters (`method`, `n_boot`, `rng_seed`,
    `n_sim`, `cluster`, `block_size`, `bootstrap_config`, `matching`)
    after the cache is built; mutating them raises `RuntimeError`.
  - Adapter-drift detection: if the underlying model is re-fitted after
    the cache exists, the next call raises `RuntimeError`.
  - `MarginsResult` exposes `n_boot_effective` and `n_boot_failed` so
    callers know how many replicates succeeded.
- **Multi-time prediction** for survival and other time-indexed adapters.
  Scenarios can now carry a `prediction_time` key; the lifelines Cox-PH
  survival adapter supports it via `with_prediction_time` shallow clones.
  `contrasts()` and `evaluate()` correctly route each scenario through
  its own predict function. The Rossi recidivism demo shows a
  counterfactual survival-curve grid computed in one bootstrap pass.
- Six end-to-end demos in `docs/demos/` using bundled datasets: Mroz LFP
  (logit), Fair affairs (logit vs Poisson), Spector PSI (small-n
  inference), Rossi recidivism (Cox PH), wage panel (entity FE), and
  ANES96 (multinomial logit). Notebook-style (`myst-nb`), executed at
  docs-build time. Surfaced as a top-level "Demos — end-to-end
  analyses" section in the sidebar; the Williams (2012) replication
  scripts move to a hidden archive page.
- Optional `progress_bar=True` on `Margins(..., method="bootstrap")` to show
  a TQDM progress bar during bootstrap refitting and evaluation.
- Public `pymargins.adapters` module exposing every concrete adapter class
  (e.g. `StatsmodelsGLMAdapter`, `LifelinesCoxPHAdapter`,
  `SklearnBootstrapAdapter`) via lazy PEP 562 resolution.  This keeps
  `import pymargins` cheap — optional third-party dependencies are loaded
  only when an adapter is actually accessed.
- Base adapter shapes (`ModelAdapter`, `GLMAdapter`,
  `LinearPredictionAdapter`, `WrappedFDAdapter`, `BootstrapOnlyAdapter`),
  `VariableInfo`, `InferenceMethod`, and `register_adapter` are also
  re-exported from `pymargins.adapters` so custom-adapter authors have a
  single import target.
- `tests/test_adapters_facade.py` ensuring every concrete adapter is mapped
  and resolved correctly.
- `tests/test_session_bank_cache.py` covering session-level bootstrap-state
  banks, simulation-draw banks, resample-index banks, cache invalidation,
  adapter-drift detection, and frozen-attribute enforcement.

### Fixed

- Broken MyST cross-references in `docs/explanations/session_precommitment.md`
  (links to `howto/bootstrap.md`, `howto/cluster_block_bootstrap.md`, and
  `howto/matching.md`).
- ReadTheDocs notebook execution now defaults to `nb_execution_mode="cache"`
  so already-executed notebooks do not re-run on every RTD build.

## [0.0.1] — 2026-05-17

Initial public release on PyPI and Read the Docs.

### Added

- `Margins` session API with explicit analytical pre-commitment:
  scale, variance estimator, confidence level, default evaluation
  point, and inference method are fixed at construction.
- Scale-specific constructors: `linear_scale`, `log_scale`,
  `logit_scale`, `correlation_scale`, and `from_formula`.
- Estimands: adjusted predictions, slopes (`dydx`), contrasts,
  difference-in-differences, and arbitrary differentiable
  `evaluate` expressions.
- Scenario helpers: `pairwise`, `reference`, `at_levels`, `grid`,
  `did`, `diff`, `all_pairwise`.
- Inference: JAX-native delta method, Krinsky–Robb simulation, and
  nonparametric bootstrap with cluster and block variants
  (percentile/BCa/normal CIs, parallel refitting).
- κ (kappa) nonlinearity diagnostic with automatic simulation
  fallback; `Margins.diagnose()` pre-flight reporting.
- Auto-detected model adapters for statsmodels (OLS/WLS/GLS, GLM,
  discrete/count, zero-inflated, MNLogit, ordered, GEE, MixedLM, RLM,
  QuantReg, PHReg), linearmodels (IV/2SLS, panel, absorbing,
  Fama–MacBeth), lifelines (CoxPH, time-varying Cox, AFT families,
  generalized gamma, piecewise exponential, cubic-spline), and
  scikit-learn (via bootstrap).
- `register_adapter` extension point for custom model backends.
- pysmatch propensity-matching integration (`PysmatchClient`).
- Polars input support.
- Result objects with `summary()`, `to_frame()`, hypothesis testing
  (`test`, `joint_test`), and `materialize()` for memory release.
- Documentation site with tutorials, how-to guides, API reference,
  and theory/design explanations.

[Unreleased]: https://github.com/huntermills707/pymargins/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/huntermills707/pymargins/releases/tag/v0.1.1
[0.1.0]: https://github.com/huntermills707/pymargins/releases/tag/v0.1.0
[0.0.1]: https://github.com/huntermills707/pymargins/releases/tag/v0.0.1
