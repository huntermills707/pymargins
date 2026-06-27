# How `pymargins` differs from `marginaleffects` and `margins`
This page is for users coming from the R `marginaleffects` package
or Stata's `margins` command. The mathematics is the same — the
implementations converge because the math forces them to. The
differences are in posture.

## 1. Estimator pre-commitment

`marginaleffects` and `margins` let you choose `vcov`, scale, level,
and aggregation per call. `pymargins` makes those choices
estimator-level. The constructor *is* the methods section; switching
any of them requires a new estimator.

This is more rigid by design. See [](plan_pre_registration.md).

## 2. Curvature awareness

Both R and Stata always do the delta method. They do not tell you
when the delta method is suspect.

`pymargins` can compute Skovgaard's relative curvature κ to flag
estimands where the delta-method linearization may be unreliable.
When κ is large, switch to `method="simulation"` or
`method="bootstrap"`. See [](kappa_diagnostic.md).

## 3. JAX-native autodiff

`marginaleffects` uses numerical derivatives. `pymargins` is built
on JAX: exact gradients for any model whose predict can be expressed
in JAX, and a custom-JVP bridge for models where it cannot. See
[](gradient_backend.md).

## 4. Inference scale as a first-class object

`marginaleffects` offers `type=` (link / response) and a couple of
hard-coded transforms. `pymargins` takes `phi`, `phi_inv` as
arbitrary JAX callables; the same chain-rule machinery handles
log-RR, Fisher-z, lift, or any user-defined scale. See
[](inference_scale.md).

## 5. Design-based survey inference

`pymargins` does complex-survey inference natively. Declare the design
once with a `SurveyDesign` (sampling weights, PSUs/clusters, strata, and
an optional finite-population correction) and pass it to the constructor:

```python
from pymargins import GComputation, SurveyDesign, steps  # 0.4.0: Margins -> GComputation
d = SurveyDesign(weights=w, psu=psu, strata=strat)
m = GComputation(steps.input(data, design=d), outcome=fit, weights=w)
m.dydx("x")   # design-weighted AME with Taylor-linearization SE
```

The point estimate is design-weighted and the standard error is the
stratified, clustered Taylor-linearization sandwich — the same quantity
R's `survey::svyglm` + `marginaleffects` produce, matched numerically.
Because the design-based covariance flows through the same frozen-cov
chokepoint as every other `vcov`, the delta *and* simulation paths are
both design-based with no extra wiring; the bootstrap path resamples PSUs
*within* strata. See [](../tutorials/survey_design.md).

## 6. Narrower scope

`pymargins` does adjusted predictions, slopes, contrasts, and
differentiable compositions. It does not do:

- model averaging,
- counterfactual prediction with model re-solving,
- post-estimation likelihood transformations.

If you need those, `marginaleffects` has broader coverage and a
gentler learning curve, and it is the right tool.

## API mapping for R users

If you are coming from `marginaleffects`, the conceptual mapping is:

| `marginaleffects` (R) | `pymargins` (Python) | Notes |
|-----------------------|----------------------|-------|
| `predictions()`       | `m.predict()`        | Adjusted predictions (AAP / APM / APR) |
| `slopes()`            | `m.dydx()`           | Marginal effects (AME / MEM / MER) |
| `comparisons()`       | `m.contrasts()`      | Linear contrasts with joint covariance |
| `avg_predictions()`   | `m.predict(at="overall")` | AAP is the default `at` |
| `avg_slopes()`        | `m.dydx(at="overall")` | AME is the default `at` |
| `svydesign()` + `svyglm()` | `GComputation(steps.input(..., design=SurveyDesign(...)), outcome=...)` | Design-based survey SEs (linearization or stratified bootstrap) |
| `plot_predictions()`  | `m.predict(...)` + `to_frame()` + matplotlib | See [](../howto/plotting.md) |
| `hypotheses()`        | `m.evaluate()`       | Nonlinear compositions |
| `type = "link"`       | inference scale      | `pymargins` commits to one scale per estimator |
| `type = "response"`   | reporting scale      | CI endpoints are back-transformed automatically |

## When to pick which

| Need                                                | Tool                |
|-----------------------------------------------------|---------------------|
| Broad model coverage, fast iteration                | `marginaleffects`   |
| Audit trail and pre-registration                    | `pymargins`         |
| Curvature diagnostics                               | `pymargins`         |
| Custom inference scales                             | `pymargins`         |
| Complex-survey design-based SEs                     | both (numerically agreed) |
| Stata-style replications                            | both (numerically agreed) |