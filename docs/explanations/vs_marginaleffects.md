# How `pymargins` differs from `marginaleffects` and `margins`

This page is for users coming from the R `marginaleffects` package
or Stata's `margins` command. The mathematics is the same — the
implementations converge because the math forces them to. The
differences are in posture.

## 1. Session pre-commitment

`marginaleffects` and `margins` let you choose `vcov`, scale, level,
and aggregation per call. `pymargins` makes those choices session-
level. The constructor *is* the methods section; switching any of
them requires a new session.

This is more rigid by design. See [](session_precommitment.md).

## 2. The κ fallback

Both R and Stata always do the delta method. They do not tell you
when the delta method is suspect.

`pymargins` computes Skovgaard's relative curvature κ as a routine
diagnostic and *auto-falls-back* to simulation when κ exceeds the
session threshold. The fallback is loud — every result records the
realized inference method and the reason it was used. See
[](kappa_diagnostic.md).

## 3. JAX-native autodiff

`marginaleffects` uses numerical derivatives. `pymargins` is built
on JAX: exact gradients (and exact Hessians for κ) for any model
whose predict can be expressed in JAX, and a custom-JVP bridge for
models where it cannot. See [](gradient_backend.md).

## 4. Inference scale as a first-class object

`marginaleffects` offers `type=` (link / response) and a couple of
hard-coded transforms. `pymargins` takes `phi`, `phi_inv` as
arbitrary JAX callables; the same chain-rule machinery handles
log-RR, Fisher-z, lift, or any user-defined scale. See
[](inference_scale.md).

## 5. Narrower scope

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
| `plot_predictions()`  | `m.predict(...)` + `to_frame()` + matplotlib | See [](../howto/plotting.md) |
| `hypotheses()`        | `m.evaluate()`       | Nonlinear compositions |
| `type = "link"`       | `phi_inv` scale      | `pymargins` commits to one scale per session |
| `type = "response"`   | `phi` back-transform | CI endpoints are back-transformed automatically |

## When to pick which

| Need                                                | Tool                |
|-----------------------------------------------------|---------------------|
| Broad model coverage, fast iteration                | `marginaleffects`   |
| Audit trail and pre-registration                    | `pymargins`         |
| Curvature-aware fallback                            | `pymargins`         |
| Custom inference scales                             | `pymargins`         |
| Stata-style replications                            | both (numerically agreed) |
