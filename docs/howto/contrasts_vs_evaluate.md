# Contrasts vs `evaluate` — choosing the right tool
> **Migration note (0.4.0):** the `Margins` session class has been removed. Use `GComputation` instead. This tutorial will be fully rewritten in R8.

`pymargins` offers two ways to combine scenario predictions:

| Tool | What it does | When to use |
|------|--------------|-------------|
| `contrasts` | **Linear** combination on the inference scale | Risk differences, ratios (via log scale), odds ratios, lift, DiD |
| `evaluate` | **Nonlinear** composition on the response scale | Raw ratios, NNT, reciprocals, custom utility functions |

The rule of thumb is simple: if your estimand can be written as a
weighted sum, use `contrasts`.  If it cannot, use `evaluate`.

## The mathematical distinction

**`contrasts`**:

```
result = φ( Σᵢ wᵢ · φ⁻¹(pᵢ) )
```

The weights `wᵢ` are fixed scalars.  The combination happens **on the
inference scale**, before back-transformation.  Because the operation is
linear in `φ⁻¹(p)`, the delta method is exact for the combination step
itself — curvature only enters through the individual predictions.

**`evaluate`**:

```
result = φ( φ⁻¹( compose(p₁, p₂, …, p_k) ) )
```

`compose` is an arbitrary function of the response-scale predictions.
The combination happens **on the response scale**, and `phi_inv` is
applied to the *output* of `compose`.  Nonlinearity in `compose`
propagates directly into the delta-method Jacobian, which usually
produces a larger κ and a wider CI.

## Decision flowchart

```text
Is the estimand a weighted sum of predictions?
├─ Yes ─────────────────────────────► contrasts
│   (risk diff, log-ratio, odds ratio, DiD, etc.)
│
└─ No ──────────────────────────────► evaluate
    (raw ratio, NNT, reciprocal, nested nonlinear, custom utility)
```

## Side-by-side: the same ratio two ways

### Preferred: ratio via `contrasts` on log scale

```python
m = Margins.log_scale(fit, at="overall")
scen, w = pairwise("treated", [1, 0])

res = m.contrasts(scenarios=scen, contrasts=w)
```

- Inference scale: `log(p₁) − log(p₀)`  
- Delta method: exact for the combination step  
- κ: usually small  
- Audit trail: `[+1, −1]` weight is explicit  
- Reporting: back-transformed to `p₁ / p₀`

### Fallback: ratio via `evaluate` on linear scale

```python
m = Margins.linear_scale(fit, at="overall")

res = m.evaluate(
    scenarios=scen,
    compose=lambda p: p[0] / p[1],
)
```

- Inference scale: `p₁ / p₀` directly  
- Delta method: approximate (ratio is nonlinear)  
- κ: usually larger  
- Audit trail: `compose` function must be inspected to know what was computed  
- Reporting: same point estimate, different (often wider) CI

Use the `evaluate` version only when your field or journal requires
inference on the **raw ratio scale** rather than the log-ratio scale.

## When `contrasts` is strictly better

| Estimand | Scale | Why `contrasts` wins |
|----------|-------|----------------------|
| Risk difference | `linear_scale` | `w = [+1, −1]` is exact; no curvature from a ratio |
| Risk ratio | `log_scale` | `log(p₁) − log(p₀)` is linear; delta is exact |
| Odds ratio | `logit_scale` | `logit(p₁) − logit(p₀)` is linear |
| Lift (RR − 1) | `log_scale` | Compute RR with `contrasts`, subtract 1 |
| DiD | `linear_scale` | Four-cell `[+1, −1, −1, +1]` is linear |
| Reference contrasts | any | Weight matrix is linear |

## When `evaluate` is required

| Estimand | Why `evaluate` is needed |
|----------|--------------------------|
| Raw ratio on linear scale | `p₁ / p₀` cannot be written as `Σ wᵢ · φ⁻¹(pᵢ)` for any standard `φ` |
| Number needed to treat | `1 / (p₁ − p₀)` is a reciprocal, not a weighted sum |
| Emax-style parameter | `(high − placebo) / (low − placebo)` is a ratio of differences |
| Custom utility | `√p₁ − √p₂` or any bespoke function |
| Ratio of ratios | `(p₁/p₀) / (p₃/p₂)` is nested nonlinear |

## Composability and audit trail

`contrasts` produces an audit-friendly result because the weight vector
is stored in `estimand_metadata` and visible in `summary()`.  A reviewer
reads `[+1, −1]` and knows exactly what was computed.

`evaluate` buries the logic inside a `compose` callable.  The result
records that `evaluate` was used, but the reviewer must inspect the
source code to verify the formula.  This is acceptable for complex
custom estimands, but it is a reason to prefer `contrasts` whenever the
two approaches agree numerically.

## Performance

`contrasts` uses a dedicated fast path for linear combinations: one
Jacobian evaluation per scenario, then matrix multiplication for the
weights.  `evaluate` differentiates through `compose`, which adds
overhead and can force an auto-route to simulation if `compose` is not
JAX-differentiable.

## Summary

1. **Start with `contrasts`**.  If your estimand is a difference, ratio
   (via log scale), odds ratio (via logit scale), or DiD, `contrasts`
   is faster, more transparent, and usually more accurate.

2. **Reach for `evaluate`** only when the estimand is genuinely
   nonlinear: reciprocals, raw-scale ratios, nested ratios, or custom
   utility functions.

3. **When in doubt**, try both.  If the point estimates agree and the
   `contrasts` CI is narrower with a smaller κ, the contrast is the
   better tool.