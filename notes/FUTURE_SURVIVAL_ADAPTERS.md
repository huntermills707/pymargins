# Plan: Survival Model Adapters (statsmodels PHReg + lifelines)

## Context

marginaleffects supports survival models by restricting delta-method inference to the hazard-ratio scale and using bootstrap for survival-probability estimands. We can follow the same playbook in pymargins.

Two packages to support:
1. **statsmodels `PHReg`** — Cox proportional hazards with `predict(params, exog, pred_type=...)`
2. **lifelines `CoxPHFitter`** and parametric AFT models (`WeibullAFTFitter`, `LogNormalAFTFitter`, etc.)

## Key Design Decisions

### 1. Three adapter shapes

**Cox PH on hazard ratio / linear predictor scale** — pure JAX, exact autodiff, all inference methods (delta, simulation, bootstrap). The baseline hazard does not enter the hazard ratio, so delta-method SEs are valid.
- statsmodels: `pred_type="lhr"` (log hazard ratio) or `"hr"` (hazard ratio)
- lifelines: `predict_partial_hazard()` or `predict_log_partial_hazard()`

**Cox PH on survival probability scale** — requires baseline hazard. Delta-method ignores variability in the estimated baseline hazard, producing anti-conservative CIs. marginaleffects recommends bootstrap (`vcov="rsample"`) for this scale. We follow the same approach: bootstrap-only for survival-probability estimands on Cox PH.
- statsmodels: `pred_type="surv"` or `"cumhaz"`
- lifelines: `predict_survival_function()` or `predict_cumulative_hazard()`

**Parametric AFT models** — pure JAX for ALL scales. Unlike Cox PH, parametric AFT models have no nonparametric nuisance function. The baseline hazard is fully parameterized (Weibull shape/scale, LogNormal μ/σ, etc.). Delta-method SEs are valid for survival probabilities, cumulative hazards, expected survival, and median survival.

For this plan: **hazard-ratio-scale Cox PH + full-scale parametric AFT**. Survival-probability Cox PH is Phase 2.

### 2. Parametric AFT models are pure JAX

Weibull, LogNormal, LogLogistic AFT models have closed-form survival functions. We can implement `predict()` entirely in JAX:

```python
# Weibull AFT
lambda_x = jnp.exp(X @ beta_lambda)  # scale
rho      = jnp.exp(beta_rho)          # shape
S(t|x)   = jnp.exp(-(t / lambda_x) ** rho)
```

This gives exact autodiff, delta-method SEs, all prediction types (survival, cumhaz, median, expectation).

### 3. No formula API for statsmodels PHReg

statsmodels `PHReg` does not have a `from_formula` constructor. Array-fit only. Training data must be provided explicitly.

### 4. lifelines has formula support

lifelines `CoxPHFitter.fit(df, duration_col="T", event_col="E", formula="x1 + x2")` supports patsy-like formulas. We can extract the design matrix from `cph.regressors` and reconstruct training data from `cph.durations`, `cph.event_observed`, and the original DataFrame.

### 5. Covariance flavors

- **statsmodels PHReg**: `cov_params()` only. No HC/cluster refit support in the underlying model.
- **lifelines CoxPHFitter**: `variance_matrix_` (default), `cluster_col` for cluster-robust, `robust=True` for robust SEs. We can support these by checking what was used during fit.
- **lifelines parametric AFT**: `variance_matrix_` only.

### 6. Censoring and bootstrap refit

Bootstrap resampling must preserve the `(time, status, X)` structure. For both packages, refit means calling `.fit()` on a resampled DataFrame with the same `duration_col`, `event_col`, etc.

## Phase 1: Hazard-Ratio Cox PH + Parametric AFT

### 1. `StatsmodelsPHRegAdapter` — new file

Inherits from `ModelAdapter` directly (not GLMAdapter because the "link" is log but the baseline hazard is nonparametric).

**`predict(beta, X, offset=None)`** — hazard ratio scale only:
```python
return jnp.exp(jnp.asarray(X) @ beta)
```

**`coefficients()`** — `jnp.asarray(self.results.params)`

**`covariance()`** — `jnp.asarray(self.results.cov_params())` for default only. No HC/cluster.

**`design_matrix_from_df()`** — array-fit only. Build from `exog_names`.

**`refit(resampled_data, *, index=None)`** — `PHReg(endog, exog, status=status).fit()` on resampled data.

**Auto-detection** — `PHRegResults` (module `statsmodels.duration.hazard_regression`)

### 2. `LifelinesCoxPHAdapter` — new file

Inherits from `ModelAdapter` directly.

**`predict(beta, X, offset=None)`** — partial hazard:
```python
X_centered = jnp.asarray(X) - self._x_mean
return jnp.exp(X_centered @ beta)
```

**Key**: lifelines centers covariates by default. `_x_mean` must be stored from the original fit.

**`coefficients()`** — `jnp.asarray(self.results.params_.values)`

**`covariance()`** — `jnp.asarray(self.results.variance_matrix_)` for default. Check if cluster/robust was used.

**`design_matrix_from_df()`** — formula-fit: use `cph.regressors` to map columns. Array-fit: align by covariate names.

**`refit(resampled_data, *, index=None)`** — `CoxPHFitter().fit(resampled, duration_col=..., event_col=..., formula=...)` preserving all kwargs.

**Auto-detection** — `CoxPHFitter` (check module `lifelines.`)

### 3. `LifelinesWeibullAFTAdapter` — new file

Inherits from `ModelAdapter` directly.

**`predict(beta, X, offset=None)`** — parametric survival probability at a fixed time `t`:
```python
# beta = [beta_lambda..., beta_rho]
p = X.shape[1]
beta_lambda = beta[:p]
beta_rho = beta[p]  # scalar
lambda_x = jnp.exp(jnp.asarray(X) @ beta_lambda)
rho = jnp.exp(beta_rho)
t = self._prediction_time  # e.g., median observed survival time
return jnp.exp(-(t / lambda_x) ** rho)
```

**Why fixed-time survival probability?** The `predict()` contract returns `(n_obs,)`. Expected survival time and median survival time are also valid scalar predictions, but survival probability at a clinically relevant time (e.g., 1-year, 5-year, or median follow-up) is the most common estimand in applied survival analysis. The adapter stores `_prediction_time` (default: median observed event time) and exposes it as a configurable property.

Users who want expected survival or median survival can use `evaluate()` with a custom estimand function. Users who want survival probabilities at other times can construct a new adapter with a different `_prediction_time`.

For Weibull: `S(t|x) = exp(-(t/λ(x))^ρ)` where `λ(x) = exp(Xβ_λ)` and `ρ = exp(β_ρ)`.

**`coefficients()`** — flatten the multi-index params: `[lambda_x1, lambda_x2, ..., lambda_intercept, rho_intercept]`

**`covariance()`** — `variance_matrix_` reshaped to match flattened params.

**`refit(resampled_data, *, index=None)`** — `WeibullAFTFitter().fit(resampled, ...)`

**Auto-detection** — `WeibullAFTFitter`

### 4. Register in auto-detection

Update `pymargins/_adapters/__init__.py`:
- `PHRegResults` → `StatsmodelsPHRegAdapter`
- `CoxPHFitter` → `LifelinesCoxPHAdapter`
- `WeibullAFTFitter` → `LifelinesWeibullAFTAdapter`

## Phase 1 Files to Modify / Create

| File | Action |
|------|--------|
| `pymargins/_adapters/__init__.py` | Register new adapters |
| `pymargins/_adapters/statsmodels_phreg.py` | **New** — statsmodels PHReg adapter |
| `pymargins/_adapters/lifelines_coxph.py` | **New** — lifelines CoxPH adapter |
| `pymargins/_adapters/lifelines_weibull_aft.py` | **New** — lifelines Weibull AFT adapter |
| `tests/test_adapter_statsmodels_phreg.py` | **New** |
| `tests/test_adapter_lifelines_coxph.py` | **New** |
| `tests/test_adapter_lifelines_weibull_aft.py` | **New** |

## Phase 1 Test Strategy

For each adapter:
1. **Auto-detection**
2. **Coefficients shape** matches cov_params shape
3. **Predictions match** native package predictions (hazard ratios for Cox, survival probability at fixed time for AFT)
4. **JAX differentiability** — grad of mean prediction w.r.t. beta
5. **Covariance** default
6. **Bootstrap refit** round-trip
7. **End-to-end** via Margins session: `predict()`, `dydx()`

## Phase 1 Risks / Open Questions

1. **lifelines covariate centering**: CoxPHFitter centers by default. Must store `_norm_mean` and apply in predict. What if user passes `normalize=False`? Need to detect.
2. **Parametric AFT multi-index params**: `params_` is a MultiIndex Series. Flattening must be deterministic and match `variance_matrix_` ordering.
3. **Prediction time for AFT**: The adapter chooses a default time `t` for survival probability prediction (median observed event time). Users can override via adapter constructor argument.
4. **Strata in Cox PH**: Both statsmodels and lifelines support stratified Cox models. Stratification affects baseline hazard estimation but not the coefficient interpretation. For hazard-ratio-scale adapter, strata are irrelevant. For survival-probability scale (Phase 2), strata matter.
5. **Ties handling**: Efron/Breslow/Exact ties methods affect the likelihood but not the predict semantics. No adapter impact.

---

## Phase 2: Survival-Probability Cox PH + Additional AFT Models

### Motivation

Phase 1 only supports hazard ratios for Cox PH. Applied survival analysis often needs:
- Survival probabilities at fixed times (e.g., 1-year survival)
- Cumulative hazards
- Median survival times
- Restricted mean survival time (RMST)
- Time-varying covariate effects

For **parametric AFT**, these are already supported in Phase 1 because the baseline hazard is fully parameterized.

For **Cox PH**, these require the baseline hazard, which is nonparametric. The delta method ignores baseline hazard uncertainty, producing anti-conservative SEs. marginaleffects handles this by recommending bootstrap for survival-probability estimands.

### Phase 2 Scope

1. **Cox PH survival-probability scale** (statsmodels + lifelines)
2. **LogNormal AFT adapter** (lifelines)
3. **LogLogistic AFT adapter** (lifelines)
4. **Restricted mean survival time (RMST)** estimand support
5. **Time-varying coefficient Cox models** (lifelines `CoxTimeVaryingFitter`)

### Phase 2 Design: Cox PH Survival-Probability Scale

#### Option A: `WrappedFDAdapter` + native predict (recommended)

Use the same pattern as `StatsmodelsOrderedAdapter`:

```python
class LifelinesCoxPHSurvivalAdapter(WrappedFDAdapter):
    def __init__(self, cph_fitter, training_data, prediction_time=None):
        self.results = cph_fitter
        self._training_data = training_data
        self._prediction_time = prediction_time or cph_fitter.durations.median()
        self._x_mean = cph_fitter._norm_mean.values

    def native_predict(self, beta_np, X):
        # Reconstruct the fitted model's baseline hazard
        # and compute survival probabilities at prediction_time
        # ... (uses lifelines internals)
        pass
```

**Pros**: Exact predictions using lifelines' native baseline hazard estimation.
**Cons**: Finite-difference gradients are slow and noisy. Not suitable for delta method.

**Inference restriction**: `supported_inference_methods = {"bootstrap"}`. Delta method is invalid because SEs ignore baseline hazard uncertainty. Simulation also inherits the same problem.

#### Option B: Pure JAX parametric baseline hazard approximation (research-grade)

Approximate the baseline hazard with a parametric function (e.g., spline or Weibull) and fit it alongside the Cox coefficients. This is what `flexsurv` does in R.

**Pros**: Exact autodiff, delta method valid.
**Cons**: Not a standard Cox PH model anymore. Different estimand.

**Decision**: Use Option A (WrappedFDAdapter + bootstrap-only). This matches marginaleffects' approach.

### Phase 2 Implementation: `LifelinesCoxPHSurvivalAdapter`

```python
class LifelinesCoxPHSurvivalAdapter(WrappedFDAdapter):
    def __init__(self, cph_fitter, training_data=None, prediction_time=None):
        self.results = cph_fitter
        self._training_data = extract_training_data(cph_fitter, training_data)
        self._prediction_time = prediction_time or self._compute_default_time()
        self._exog_names = list(cph_fitter.params_.index)
        self._x_mean = getattr(cph_fitter, '_norm_mean', None)
        self._duration_col = cph_fitter.duration_col
        self._event_col = cph_fitter.event_col
        self._formula = getattr(cph_fitter, 'formula', None)

    def _compute_default_time(self):
        # Use median observed event time as default
        durations = self.results.durations
        events = self.results.event_observed
        return np.median(durations[events])

    def native_predict(self, beta_np, X):
        """Compute survival probability at prediction_time.

        This requires reconstructing the baseline hazard from the
        fitted model. For lifelines, we can use the baseline survival
        function stored on the fitter and raise it to the partial hazard.
        """
        X_np = np.asarray(X)
        # Center if needed
        if self._x_mean is not None:
            X_np = X_np - self._x_mean
        # Compute partial hazard
        ph = np.exp(X_np @ beta_np)
        # Get baseline survival at prediction_time
        S0_t = self._baseline_survival_at(self._prediction_time)
        # Survival probability
        return S0_t ** ph

    def _baseline_survival_at(self, t):
        # Use lifelines' stored baseline survival function
        S0 = self.results.baseline_survival_
        # Interpolate to time t
        # ... (handle edge cases)
        pass

    @property
    def supported_inference_methods(self):
        return {"bootstrap"}  # Delta method invalid for survival probabilities

    def refit(self, resampled_data, *, index=None):
        # Preserve all original fit kwargs
        kwargs = {
            'duration_col': self._duration_col,
            'event_col': self._event_col,
        }
        if self._formula is not None:
            kwargs['formula'] = self._formula
        new_cph = CoxPHFitter()
        new_cph.fit(resampled_data, **kwargs)
        return LifelinesCoxPHSurvivalAdapter(
            new_cph, training_data=resampled_data,
            prediction_time=self._prediction_time
        )
```

### Phase 2 Implementation: LogNormal AFT Adapter

LogNormal AFT: `S(t|x) = 1 - Φ((log(t) - μ(x)) / σ)` where `μ(x) = X β_μ` and `σ = exp(β_σ)`.

```python
class LifelinesLogNormalAFTAdapter(ModelAdapter):
    def predict(self, beta, X, offset=None):
        p = X.shape[1]
        beta_mu = beta[:p]
        beta_sigma = beta[p]
        mu = jnp.asarray(X) @ beta_mu
        sigma = jnp.exp(beta_sigma)
        t = self._prediction_time
        from jax.scipy.special import ndtr
        return 1.0 - ndtr((jnp.log(t) - mu) / sigma)
```

### Phase 2 Implementation: LogLogistic AFT Adapter

LogLogistic AFT: `S(t|x) = (1 + (t / α(x))^β)^(-1)` where `α(x) = exp(X β_α)` and `β = exp(β_β)`.

```python
class LifelinesLogLogisticAFTAdapter(ModelAdapter):
    def predict(self, beta, X, offset=None):
        p = X.shape[1]
        beta_alpha = beta[:p]
        beta_beta = beta[p]
        alpha = jnp.exp(jnp.asarray(X) @ beta_alpha)
        beta_param = jnp.exp(beta_beta)
        t = self._prediction_time
        return 1.0 / (1.0 + (t / alpha) ** beta_param)
```

### Phase 2 Implementation: Restricted Mean Survival Time (RMST)

RMST at time τ: `E[min(T, τ) | x] = ∫₀^τ S(t|x) dt`

This is an `evaluate()` estimand, not a `predict()` scale. Users would define:

```python
m = Margins(weibull_model, adapter=adapter)
result = m.evaluate(lambda S: jnp.trapezoid(S, dx=dt))
```

Where `S` is the survival function evaluated on a grid of times. The adapter would need a method to return the full survival curve, not just a point estimate.

**Decision**: Defer RMST to a separate `SurvivalCurveAdapter` interface that returns `(n_obs, n_times)` survival matrices. This is a significant architectural extension beyond the current `(n_obs,)` predict contract.

### Phase 2 Files to Modify / Create

| File | Action |
|------|--------|
| `pymargins/_adapters/__init__.py` | Register new adapters |
| `pymargins/_adapters/lifelines_coxph_survival.py` | **New** — lifelines CoxPH survival-probability adapter (WrappedFDAdapter) |
| `pymargins/_adapters/statsmodels_phreg_survival.py` | **New** — statsmodels PHReg survival-probability adapter (WrappedFDAdapter) |
| `pymargins/_adapters/lifelines_lognormal_aft.py` | **New** — lifelines LogNormal AFT adapter |
| `pymargins/_adapters/lifelines_loglogistic_aft.py` | **New** — lifelines LogLogistic AFT adapter |
| `tests/test_adapter_lifelines_coxph_survival.py` | **New** |
| `tests/test_adapter_lifelines_lognormal_aft.py` | **New** |
| `tests/test_adapter_lifelines_loglogistic_aft.py` | **New** |

### Phase 2 Risks / Open Questions

1. **Baseline hazard interpolation**: For Cox PH survival probabilities, we need to evaluate `S_0(t)` at arbitrary times `t`. Lifelines stores it at event times only; interpolation is needed for user-specified times.
2. **Bootstrap for survival probabilities**: Each replicate re-estimates the baseline hazard, which is computationally expensive. May need `n_boot=200` rather than `n_boot=1000`.
3. **Time-varying coefficients**: `CoxTimeVaryingFitter` has a different data format (start/stop intervals). The training data structure changes significantly.
4. **Competing risks**: `AalenJohansenFitter` handles competing risks with a completely different prediction structure (cumulative incidence functions).

---

## Recent Bug Fixes to Account For (Phase 1 Review)

### 1. `refit()` signature change

All adapters now use `refit(self, resampled_data, *, index=None)`. The `index` parameter is passed from `_run_bootstrap` and must be accepted even if unused. Survival adapters must follow this signature.

### 2. Bootstrap resampling strategies

`_run_bootstrap` now supports cluster bootstrap (`cluster=`) and block bootstrap (`block_size=`, `bootstrap_config={"block_type": ...}`). For survival models:
- **Cluster bootstrap**: Valid for Cox PH with clustered data (e.g., patients in hospitals). The `cluster_col` from lifelines should align with the `cluster=` parameter.
- **Block bootstrap**: Generally not applicable to survival data unless it's time-to-event with temporal correlation (rare).
- **i.i.d. bootstrap**: Valid for independent survival data. Censoring structure is preserved by resampling full rows.

### 3. `InferenceConfig` expansion

`InferenceConfig` now has `cluster`, `block_size`, and `bootstrap_config` fields. These flow from `Margins.__init__()` → `_inference_config()` → `_run_bootstrap()`. No adapter changes needed.

### 4. Strict mode validation

`Margins.__init__()` strict mode now validates `cluster`, `block_size`, and `bootstrap_config`. Tests must pass these explicitly under strict mode.

### 5. Mutual exclusion validation

`cluster` and `block_size` are mutually exclusive in `Margins.__init__()`. For survival models with clustered data, users would use `cluster=` (not `block_size=`).

### 6. `_run_bootstrap` passes `index` to `refit`

Line 492: `new_adapter = adapter.refit(resampled, index=idx)`. All `refit` implementations must accept `index`.

### 7. `np.isnan` vs `pd.isna`

In `_run_bootstrap`, cluster validation uses `pd.isna(cluster_ids)` rather than `np.isnan()` to handle mixed types. No adapter impact.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-08 | Hazard-ratio scale only for Cox PH Phase 1 | Delta method is valid for HR; survival probability needs baseline hazard |
| 2026-05-08 | Fixed-time survival probability for AFT default | Most common applied estimand; `predict()` returns `(n_obs,)` |
| 2026-05-08 | Bootstrap-only for Cox PH survival probability Phase 2 | marginaleffects uses same approach; delta method ignores baseline hazard uncertainty |
| 2026-05-08 | WrappedFDAdapter for Cox PH survival probability | Exact predictions using native baseline hazard; no parametric approximation |
| 2026-05-08 | Defer RMST to future architectural extension | Requires `(n_obs, n_times)` survival matrix, beyond current `(n_obs,)` contract |
