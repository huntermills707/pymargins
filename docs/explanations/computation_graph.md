# The computation graph

New in **pymargins 0.4.0**.

The estimator core has been rebuilt around a **functional computation graph**.
Instead of passing every analysis choice as a keyword argument to a monolithic
estimator object, you assemble a **wiring graph** from small, pure `steps.*`
verbs, compile it once into an immutable **Plan**, and read out results through
typed **estimator nouns**.

## Why a graph?

Every release added another axis to `GComputation`: matching, transforms, survey
designs, multiple imputation. Each axis landed as more kwargs on a God object.
The graph keeps the assembled depth explicit:

```python
from pymargins import GComputation, steps

prep = steps.input(df, design=design)
prep = steps.match(prep, matcher)
prep = steps.trim(prep, "x1", lower=0.0, upper=1.0)

est = GComputation(prep, outcome="y ~ treat*x1 + x2", method="delta")
```

The graph makes *what re-executes under what* derivable from position, not
from a matrix of kwargs.

## Three layers

1. **Wiring graph** (`steps.*`) — pure dataflow. Nodes are frozen,
   content-addressed values. `steps.input(df, cluster=ids, block=k)` is the
   resampling root; every downstream stage re-executes per bootstrap replicate
   by construction.
2. **Influence-function contract** (`adapter.influence()`) — tier-1 sources
   expose per-observation `ψ^β = A @ score_obs`, making analytic survey
   correction and (in future releases) AIPW composition possible.
3. **M-transforms** — bootstrap, simulation, and MI are higher-order
   transforms *over* the graph, not nodes in it. `method=` is resolved once at
   compile and never flips at runtime.

## Pre-registration

Everything that defines the analysis binds once, at construction:
`at`, `scale`, `method`, `vcov`, `ci`, `level`, `B`/`n_sim`.
Queries are readouts. The constructor returns a `Plan` whose hash is printed
on every result:

```python
est = GComputation(model, method="delta", level=0.95)
print(est.plan.hash)   # e.g. 'a7f3c21@1'
print(est.plan.describe())
```

Changing any analysis parameter creates a new estimator object with a new
hash — the anti-hacking mechanism is visibility, not prevention.

## How pymargins is validated

Correctness is anchored outside the package in four layers:

1. **Analytic suite** (`tests/oracle/test_analytic.py`) — closed-form
   identities such as "OLS `dydx("x")` estimate equals `β_x` and SE equals
   `sqrt(cov_params[x,x])`".
2. **R reference suite** (`tests/oracle/test_r_golden.py`) — goldens generated
   by `marginaleffects`, `survey`, and `sandwich` in R, committed as JSON, and
   compared in the default test lane.
3. **Regression goldens** (`tests/golden/`) — byte-exact recordings of anchor
   cells from the validated engine, protecting deterministic resampling streams
   that no external oracle can check.
4. **Calibration lane** (`tests/test_calibration_slow.py`, weekly slow lane) —
   Monte Carlo coverage simulations confirm that delta and simulation CIs reach
   their nominal level on correctly-specified DGPs, and that simulation and
   bootstrap SEs agree with the analytic delta SE on smooth cases.

Disagreement between independent oracles beyond recorded tolerance is a
stop-and-fix event, not a tolerance tweak.

## Migration from 0.3.x

`Margins` is removed. The mapping is:

| 0.3.x | 0.4.0 |
|---|---|
| `GComputation(model)` | `GComputation(model)` |
| `cluster=` / `block_size=` | `steps.input(df, cluster=...)` / `steps.input(df, block=...)` |
| `survey_design=` | `steps.input(df, design=...)` |
| `matching=` | `steps.match(node, matcher)` |
| `transforms=` | `steps.trim` / `steps.drop_outliers` / `steps.reimpute` chain |
| `bootstrap_config=` | `ci=` + `B=` + `seed=` + `steps.input(..., block_type=...)` |
| `phi=` / `phi_inv=` | `scale=` (named or callable pair) |
| `GComputation(m, scale="log")` | `GComputation(m, scale="log")` |

See [](plan_pre_registration.md) for the level/CI doctrine, and
[](../howto/kappa_fallback.md) for how κ is handled under decide-once semantics.
