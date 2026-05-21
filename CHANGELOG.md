# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/huntermills707/pymargins/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/huntermills707/pymargins/releases/tag/v0.0.1
