# Delta vs simulation vs bootstrap

Three inference paths, one session knob (`method=`). Picking is a
function of curvature, sample structure, and refit cost.

## Delta

- First-order Taylor of `g(β)` around `β̂`.
- Cheap: one Jacobian-vector product per estimand.
- Accurate when κ is small (see [](kappa_diagnostic.md)).
- Symmetric Wald CI on the inference scale; can cross natural
  bounds when back-transformed.

## Krinsky–Robb simulation

- Draw `β_s ∼ N(β̂, V̂)`, evaluate `g(β_s)`, read inference off the
  empirical distribution.
- No refits — just `S` forward evaluations.
- Reported point estimate stays analytic `g(β̂)` (matches Stata's
  `vce(simulation)`).
- Pointwise CIs default to empirical quantiles, so asymmetric CIs
  for extreme probabilities / ratios are natural.
- Used as the κ fallback target.

## Bootstrap

- Refit on `B` resamples, read inference off the bootstrap
  distribution.
- Expensive (refit cost × `B`) but robust to model misspecification
  *and* to estimand curvature.
- Three resampling schemes (`pairs`, `cluster`, `block`); see
  [](../howto/cluster_block_bootstrap.md).
- Required when the asymptotic variance is suspect (small clusters,
  long-memory time series, mis-specified GLM family).

## Decision rule

```text
┌──────────────────────────────────────────────────────┐
│ Curvature high (κ ≥ 0.3) ──────────► simulation      │
│ Cluster correlation ──────────────► cluster bootstrap │
│ Heavy time-series dependence ─────► block bootstrap   │
│ Otherwise ───────────────────────► delta              │
└──────────────────────────────────────────────────────┘
```

`pymargins` automates the first row through the κ fallback. The
others are user-decisions encoded at session construction.
