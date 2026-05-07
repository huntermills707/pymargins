# Architectural Review of pymargins

## Overall posture

The architecture is genuinely well-conceived. The PRIMER nails the core
commitments — session-level pre-commitment, the `(h, phi, phi_inv)` estimand
triple, three orthogonal axes (estimand × aggregation × inference), and clean
layered dependencies. The numerical kernels are textbook-correct. The adapter
shapes (`GLMAdapter`, `LinearPredictionAdapter`, `WrappedFDAdapter`,
`BootstrapOnlyAdapter`) are a good factoring of how prediction differs across
model classes.

That said, the API has bugs, mostly in the user-facing layer where
scale-handling crosses module boundaries. Find them now, before downstream
tests hide them as test-level workarounds.

> **Post-fix changelog** — The definite bugs B1, B2/B3, B4, B6, B10, B14,
> B15, B27 and the design issues A1, A6 identified in this review have
> been resolved. B13 ("dydx doesn't block discrete var_type") is moot:
> `_infer_variable_type` no longer emits `"discrete"`, and `dydx()` now
> uses data-side FD (see "Behavioral changes" below) so discrete-vs-
> continuous classification at the design-matrix level no longer affects
> slope semantics.
>
> Remaining open items: B16, B17, A4, A5, A8, A11, B12, B18.
>
> **Behavioral changes since this review:**
> - `dydx(v)` now returns the **total** derivative ∂μ/∂v computed via a
>   data-side central difference: the source DataFrame's column `v` is
>   perturbed by ±ε, the design matrix is rebuilt through
>   `adapter.design_matrix_from_df`, and patsy regenerates every
>   interaction, polynomial, spline, and `I(...)` transform. Matches
>   Stata's `margins, dydx()` and R's `marginaleffects::slopes()`.
>   The previous column-wise partial silently underreported marginal
>   effects for any model with interactions or transforms.
> - `column_index_of_variable` is now a type guard for `dydx()` only;
>   its return value is unused. Categorical/binary variables still raise
>   here with a "use contrasts() instead" message.
> - `is_jax_differentiable` strengthened to probe `jax.vmap` and
>   `jax.hessian`, matching the trace patterns the engine actually uses
>   (simulation draws and the κ diagnostic). The previous single-point
>   `jax.grad` probe missed `TracerBoolConversionError` cases.
> - κ computed at β̂ on all three inference paths (delta, simulation,
>   bootstrap) when h is JAX-differentiable. Previously only delta
>   reported it.
> - `delta_simulation_disagreement` works for vector estimands.
> - `LogC` link inverse and derivative corrected; `Power(0)` link maps
>   to the log link as statsmodels does.
> - Auto-fallback from delta to simulation surfaces on the result via
>   `fallback_triggered` / `fallback_reason`, propagating through
>   composition.


## Definite bugs

**B1 — Missing `jax` import breaks every `dydx()` call** *(FIXED)*
`pymargins/_estimands.py:191`
`make_slope_estimand` calls `jax.vmap(...)` but the module imports only
`jax.numpy as jnp` (line 28). `jax` is not in this module's namespace. First
call to `dydx()` raises `NameError`. (`is_jax_differentiable` correctly
imports `jax` inside the function — line 422 — confirming this was an
oversight, not a pattern.)

**B2/B3 — `weights` silently dropped in the "overall" aggregate path** *(FIXED)*
`pymargins/_estimands.py:96-98, 278-279`
Both `make_prediction_estimand` and `make_linear_combination_estimand` accept
a `weights` kwarg. The "overall" branch ignores it; only "weighted" honors
it. Margins always passes `aggregate="overall"` for `at="overall"`
(margins.py:802), so survey-weighted AME silently uses uniform weights.
Either fold "weighted" into "overall" or document that "overall" means
unweighted.

**B4 — Cholesky NaN passthrough on non-PSD `Σ̂`** *(FIXED)*
`pymargins/_kappa.py:124-133`
`jnp.linalg.cholesky` on JAX returns a NaN-filled matrix for non-PSD inputs
*without raising*. The `try/except Exception` block won't fire; `L` becomes
NaN and propagates silently into κ. Need an explicit `jnp.isnan(L).any()`
check, then route to ridge.

**B6 — Simulation path is a Python for-loop over `h`** *(FIXED)*
`pymargins/_inference.py:283`
`h_draws_inf = np.array([np.asarray(h(jnp.asarray(b))) for b in draws_beta])`
runs `n_sim=4000` independent JAX traces. With κ-driven auto-fallback firing
for borderline estimands, this becomes the hot path and will be unusable for
nontrivial models. Use `jax.vmap(h)(draws_beta)` for the JAX path.

**B10 — `MarginsResult.__mul__` (and therefore `scaled()`) is wrong on
non-identity scales** *(FIXED)*
`pymargins/_result.py:573-599`
For a log-scale result with `estimate=1.5` (RR=1.5), `result * 2` returns
`estimate=3.0`. The mathematically correct value is `1.5² = 2.25` (since
doubling the log-RR exponentiates as squaring the RR). The CI-bound swap
also assumes identity-scale arithmetic. `__mul__`/`scaled()` should either
restrict to identity scale or apply `phi(scalar * phi_inv(estimate))`
properly. PRIMER §7 explicitly says scaling is cosmetic; the implementation
breaks that promise off identity scale.

**B14 — No reproducibility seed plumbed through** *(FIXED)*
`pymargins/margins.py:723`
`_inference_config()` hardcodes `rng_seed=None`. There is no session-level
`rng_seed`; running the same `m.contrasts(...)` twice gives different draws.
Add `rng_seed` as a session-level argument.

**B15 — `Σ̂` is cached lazily, not at session construction** *(FIXED)*
`pymargins/margins.py:736-738`
`_frozen_cov` only computes Σ̂ on first method call. Between `Margins(...)`
and the first `.predict(...)`, mutating the underlying model changes what
Σ̂ gets frozen. PRIMER and result-layer comments imply Σ̂ is "frozen at
session creation" — make it eager.

**B16 — `_build_prediction_estimand` slicing depends on undocumented layout**
`pymargins/margins.py:799-800`
`X[i*rows_per : (i+1)*rows_per]` assumes `expand_scenario` concatenated grid
blocks contiguously. Currently true but not in the contract; if
expand_scenario ever interleaves rows, this silently produces wrong design
matrices. Better: have `expand_scenario` return a list of per-grid-point
DataFrames.

**B17 — `over=` requires pandas; not in adapter contract**
`pymargins/margins.py:839` (also `diagnose()` at 600)
`_enumerate_groups` calls `base_data.groupby(...)`. Adapters that expose
non-pandas `training_data` will fail here. Either make pandas part of the
adapter contract or guard explicitly.

**B27 — Wasted gradient trace** *(FIXED)*
`pymargins/_inference.py:153, 197`
`is_jax_differentiable(h, beta)` traces through `jax.grad(h)` to test, then
`_run_delta` immediately calls `gradient(h, beta)` again. For known-autodiff
adapters, skip the check entirely (use `adapter.supports_jax_autodiff`).

**A6 — `atexog` accepts dict-or-DataFrame in docs, only dict in
implementation** *(FIXED)*
`pymargins/margins.py:302`, `pymargins/_scenarios.py:94-106`
Margins.predict puts atexog into `scenario["atexog"]`. `expand_scenario`
checks `scenario["data"]` for the DataFrame path. The DataFrame override
never fires from Margins. Either route DataFrame atexog → `scenario["data"]`
or update the docstring.


## Design issues / latent fragility

- **A1 — Session weakref leaks into result reporting.** *(FIXED)*
  `MarginsResult.summary()`, `to_frame()`, and `test()` all call
  `self._session_obj()` to get `phi`/`phi_inv`. If the session is GC'd,
  results become partially unusable. Capture `phi`, `phi_inv`, scale label
  on the result at construction. Same-session check for composition is the
  only reason to keep the weakref.

- **A4 — `over=` logic duplicated.** `_scenarios.expand_with_over` exists;
  `margins.py:_enumerate_groups` reimplements it. Pick one.

- **A5 — `h_factory` constructed unconditionally.** Only the bootstrap path
  uses it; delta/simulation closes over the adapter. Build it lazily when
  `method="bootstrap"`.

- **A8 — Kappa fallback throws away the gradient.** `_run_delta` computes
  `gradient`, then computes `kappa` (which traces gradient and Hessian
  itself), then if κ > threshold calls `_run_simulation` from scratch.
  Restructure to compute κ first.

- **A11 — No third-party adapter registry.** `_detect_adapter_class` is
  hardcoded. PRIMER invites custom adapters; needs a `register()` hook.

- **B12 — `result.test(kind=...)` ignores `kind`.** Only Wald is
  implemented. Either drop the parameter or stub LR/score and raise
  NotImplementedError.

- **B13 — `dydx` doesn't block "discrete" var_type.** *(MOOT)*
  `_infer_variable_type` no longer emits `"discrete"`; integer columns
  with few unique values are classified `"continuous"` (or `"binary"`
  when there are exactly 2 uniques). `dydx()` uses a data-side central
  difference through `design_matrix_from_df`, so the underlying
  semantics work for any numeric continuous column regardless of
  classification.

- **B18 — Subtle semantic distinction between `contrasts` and `evaluate`.**
  `contrasts` does `Σ wᵢ φ⁻¹(pᵢ)`; `evaluate` does `φ⁻¹(compose(p))`. Both
  correct, both useful, but undocumented. A worked example in `evaluate`'s
  docstring showing the difference would save users from confusion.


## Architecturally sound, ready to build on

The following modules are tight enough to write tests against and start
exercising end-to-end:

1. **`_gradients.py`** — Clean three-backend dispatch. Custom-JVP wrappers
   are correct. The FD Hessian's O(n²) cost is a known limitation, not a
   bug. Ready for IMPLEMENTATION_GUIDE 0.1 tests.

2. **`_delta.py`** — Pure numerical kernel; minimal API surface. Math is
   textbook; straightforward to validate against statsmodels. Ready for
   IMPLEMENTATION_GUIDE 0.2.

3. **`_kappa.py`** — Whitening-via-Cholesky implementation is correct.
   Modulo B4 (NaN check), the κ math, classification, session-level
   diagnostic, and disagreement-vs-simulation cross-check are all in place.

4. **`_adapter.py`** — Abstract interface and four shapes are well-factored.
   `attach()` validation hook is a clean idiom. The interface contract
   documented in IMPLEMENTATION_GUIDE matches the code.

5. **`__init__.py`** — Public API surface is small and intentional.


## Needs work before dependent code stabilizes

1. **`_estimands.py`** — ~~B1 blocks `dydx`; B2/B3 silently drop weights. Fix
   both before any AME testing.~~ Both fixed.

2. **`_inference.py`** — ~~Functional but inefficient (B6, A8).~~ B6 fixed.
   A8 (kappa fallback throws away gradient) remains open. Bootstrap path
   now exists with `h_factory` (per inner IMPLEMENTATION_GUIDE), which is
   good — but Margins should pass it conditionally.

3. **`_result.py`** — ~~Composition arithmetic is correct on identity scale;
   broken on non-identity (B10). Capture `phi`/`phi_inv` on the result
   instead of dereferencing the session.~~ Both fixed.

4. **`margins.py`** — ~~Eager Σ̂ (B15), `rng_seed` plumbing (B14), DataFrame
   atexog routing (A6)~~ — fixed. pandas-coupling for `over=` (B17) remains
   open. The class is doing a lot; consider extracting `_build_*_estimand`
   into a separate "session compiler" once the bugs are out.

5. **`StatsmodelsGLMAdapter`** — Skeleton works for formula-fit logit.
   `column_index_of_variable`'s factor heuristic is fragile (acknowledged
   in IMPLEMENTATION_GUIDE 0.3). Cluster vcov requires refit (acknowledged).


## Recommended order of operations

*Completed:* B1, B2/B3, B4, B6, B10, B14, B15, A1, A6, B27.

*Remaining open items:* B16, B17, A4, A5, A8, A11, B12, B13, B18.

1. ~~Fix B1 (10-second fix; otherwise dydx fails on first call).~~ Done.
2. ~~Fix B4 (Cholesky NaN check on JAX) — same scale of fix.~~ Done.
3. Write tests for `_gradients` and `_delta` against analytical truth and
   statsmodels (IMPLEMENTATION_GUIDE 0.1, 0.2). These will surface anything
   else broken in the kernels.
4. ~~Fix B6 (vmap simulation) — needed before κ-fallback exercises anything
   bigger than toy examples.~~ Done.
5. ~~Decide and document weights semantics (B2/B3).~~ Done: "overall" now
   uses weights when provided.
6. ~~Fix B10 + capture `phi`/`phi_inv` on results (A1) — these together fix
   the scale-handling fragility in `_result.py`.~~ Done.
7. ~~Fix B14, B15 (rng seed, eager cov).~~ Done.
8. ~~End-to-end smoke test (IMPLEMENTATION_GUIDE 0.5).~~ Done.

The first four items unblock the kernel layer and the simulation hot path.
After that, `_result.py` and the user-facing layer can be tightened against
real test cases rather than against speculation.
