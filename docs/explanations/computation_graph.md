# The computation graph
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

New in **pymargins 0.4.0**.

The estimator core has been rebuilt around a **functional computation graph**.
Instead of passing every analysis choice as a keyword argument to a monolithic
session object, you assemble a **wiring graph** from small, pure `steps.*`
verbs, compile it once into an immutable **Plan**, and read out results through
typed **estimator nouns**.

## Why a graph?

Every release added another axis to `Margins`: matching, transforms, survey
designs, multiple imputation. Each axis landed as more kwargs on a God object.
The graph keeps the assembled depth explicit:

```python
prep = steps.input(df, design=design)
prep = steps.impute(prep, imputer, m=20)
prep = steps.match(prep, matcher)
est = GComputation(prep, outcome="y ~ treat*x1 + x2", ...)
```

The graph makes *what re-executes under what* derivable from position, not
from a matrix of kwargs.

## Three layers

1. **Wiring graph** (`steps.*`) — pure dataflow. Nodes are frozen,
   content-addressed values.
2. **Influence-function contract** (`adapter.influence()`) — tier-1 sources
   expose per-observation ψ, making analytic survey correction and AIPW
   composition possible.
3. **M-transforms** — bootstrap, simulation, and MI are higher-order
   transforms *over* the graph, not nodes in it.

## Pre-registration

Everything that defines the analysis binds once, at construction:
`at`, `scale`, `method`, `vcov`, `ci`, `level`, `B`/`n_sim`.
Queries are readouts.  The constructor returns a `Plan` whose hash is
printed on every result:

```python
est.plan.hash   # 'a7f3c21@1'
```

Changing any analysis parameter creates a new estimator object with a new
hash — the anti-hacking mechanism is visibility, not prevention.

## Migration

`Margins` continues to work unchanged.  It is internally rewritten as a shim
over the same engine, and the existing test suite is its correctness proof.
The new surface (`GComputation`, `steps`) is documented as the preferred API
starting in 0.4.0.