# Gradient backend: autodiff vs wrapped-FD vs FD

`pymargins` computes the delta-method Jacobian by JAX
autodifferentiation when possible, by autodiff over a custom-JVP FD
primitive when the model is a black box, and by full finite
differences as a last resort.

| Backend       | When picked                                     | Pros                                | Cons                                          |
|---------------|-------------------------------------------------|-------------------------------------|-----------------------------------------------|
| `autodiff`    | predict can be expressed in JAX (GLMs, OLS)     | exact gradient and Hessian          | model must be JAX-implementable               |
| `wrapped_fd`  | black-box predict, but `η = X β` is accessible  | exact gradient outside the boundary | one FD call per parameter at the boundary     |
| `fd`          | full black box                                  | works on anything                   | Hessian quality compounds poorly (bad for κ)  |

The session argument `gradient_backend="auto"` picks the best
available path per adapter; the choice is sticky for the session.

## The custom-JVP bridge

For a black-box predict `f(β, X)`, the wrapped-FD path wraps the
*predict boundary itself* in a JAX primitive with a custom JVP that
does central differences. Once the primitive is registered, all
downstream estimand math (averaging over rows, applying `phi`,
forming contrasts) is autodiff. The FD compounding is bounded to one
primitive call.

This is the recommended path for adapter implementers when JAX
reimplementation would be error-prone but the linear predictor
`η = X β` is exposed by the fitted result. The helpers
`make_predict_with_fd_jvp` and `make_glm_jvp_wrapper` in
`pymargins._gradients` factor out the boilerplate.

## When full FD is unavoidable

For models with no exposed `η = X β` structure (rare, but it
happens — some bespoke fitters, certain mixture models), the full-FD
backend differentiates the entire estimand through the model's
predict function. Gradient quality is acceptable; *Hessian* quality
is poor, which means κ is noisier and the fallback decision becomes
less reliable. In these cases, prefer a bootstrap or simulation
session — and consider whether you want to expose the linear
predictor to upgrade the adapter to `wrapped_fd`.

See [](adapter_pattern.md) for the adapter contract that decides the
backend.
