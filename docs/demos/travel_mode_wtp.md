---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Travel mode choice — the value of travel time (WTP)
When a commuter chooses between modes, they trade money against time.
The rate at which they are willing to substitute one for the other —
how many dollars an hour of travel time is worth to them — is the
**value of travel time savings (VTTS)**, the single most-used number
in transport appraisal. It is a *willingness to pay*: the marginal
rate of substitution between a time attribute and the cost (price)
attribute in a discrete-choice model.

`pymargins` exposes this directly. For a model with a continuous time
regressor and a continuous cost regressor, `m.wtp(attribute, price)`
returns

$$
\text{WTP} \;=\; -\,\frac{\partial U/\partial\,\text{attribute}}
                          {\partial U/\partial\,\text{price}},
$$

with the standard error propagated jointly through *both* slopes — a
ratio, so the uncertainty is not just the two marginal SEs bolted
together. This demo estimates VTTS on the classic Greene–Hensher
travel-mode data and shows why the ratio's interval rewards a
simulation cross-check.

```{code-cell} python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymargins import GComputation  # 0.4.0: Margins -> GComputation

# The TravelMode data ships in long form: one row per (traveller, mode),
# 210 travellers × 4 modes (air / train / bus / car).
long = pd.read_csv("data/travelmode.csv")
print(long.head(8).to_string(index=False))
```

## 1. A binary choice with attribute *differences*

`statsmodels` multinomial logit takes traveller-level regressors, but
cost and time are **alternative-specific** — each mode has its own.
The textbook device that turns alternative-specific attributes into a
chooser-level regression is the *difference specification*: restrict
to two alternatives and regress the choice on the *difference* in each
attribute between them. The coefficient on a difference is the
utility weight on that attribute, which is all VTTS needs.

We take the two ground modes, **car vs train**, and difference their
in-vehicle time (`travel`, minutes) and out-of-pocket cost (`vcost`,
dollars). A full appraisal would fit a conditional logit over all four
modes; the two-alternative difference model is the version that fits a
chooser-level GLM, and it is plenty to exercise `wtp()` on real data:

```{code-cell} python
wide = long.pivot(index="individual", columns="mode",
                  values=["vcost", "travel"])
wide.columns = [f"{attr}_{mode}" for attr, mode in wide.columns]

chosen = long[long["choice"] == "yes"].set_index("individual")[["mode", "income"]]
df = wide.join(chosen)

# Restrict to travellers who chose car or train, and build the binary outcome.
df = df[df["mode"].isin(["car", "train"])].copy()
df["car"] = (df["mode"] == "car").astype(int)
df["cost_diff"] = df["vcost_car"] - df["vcost_train"]   # dollars
df["time_diff"] = df["travel_car"] - df["travel_train"]  # minutes

print(f"{len(df)} travellers; car share = {df['car'].mean():.2f}")
print(df[["car", "cost_diff", "time_diff", "income"]].describe().round(1))
```

## 2. Fit the choice model

```{code-cell} python
fit = smf.glm(
    "car ~ cost_diff + time_diff + income",
    data=df,
    family=sm.families.Binomial(),
).fit()
print(fit.summary().tables[1])
```

Both attribute coefficients are negative and significant: a car that
costs more, or takes longer, *relative to the train* is chosen less
often — exactly the sign theory demands. Cost and time are individually
well identified; the question is what their **ratio** says, and how
sure we are of it.

## 3. Average marginal effects

On the probability scale, each extra dollar of relative car cost and
each extra minute of relative car time both lower P(choose car):

```{code-cell} python
m = GComputation(fit, vcov="HC3", at="overall", scale="identity")

print(m.dydx("cost_diff").summary())
print(m.dydx("time_diff").summary())
```

## 4. Willingness to pay — the value of travel time

`wtp` forms the ratio with joint inference. Because `time_diff` is a
*nuisance* attribute (more time lowers utility), the WTP for one more
minute is negative — travellers would need to be *compensated* to
accept it. The interpretable headline number is its negation: the
value of travel-time **savings**, in dollars per hour.

```{code-cell} python
wtp_minute = m.wtp("time_diff", "cost_diff")
print(wtp_minute.summary())

vtts_per_hour = -float(wtp_minute.estimate) * 60
print(f"\nValue of travel time savings ≈ ${vtts_per_hour:.2f} per hour")
```

A value in the mid-teens of dollars per hour is squarely in the range
transport economists report for this dataset — a sanity check that the
difference specification recovers a sensible number, not just a
significant coefficient.

## 5. Why the ratio wants a simulation cross-check

The two slopes are each tightly estimated, but their *ratio* is a
nonlinear function of β, and the denominator (`cost_diff`) is the
noisier of the two. That curvature is exactly what the κ diagnostic
watches. Re-running the same WTP under simulation shows how much the
delta-method interval understates the asymmetry:

```{code-cell} python
m_sim = GComputation(fit, vcov="HC3", at="overall",
    method="simulation", n_sim=4000, seed=0, scale="identity")
wtp_sim = m_sim.wtp("time_diff", "cost_diff")

def ci_str(res):
    lo, hi = (float(x) for x in res.conf_int())
    return f"[{lo:+.3f}, {hi:+.3f}]  width {hi - lo:.3f}"

print(f"delta      WTP/min = {float(wtp_minute.estimate):+.3f}  95% CI {ci_str(wtp_minute)}")
print(f"simulation WTP/min = {float(wtp_sim.estimate):+.3f}  95% CI {ci_str(wtp_sim)}")
```

The simulation interval is wider and skewed — the right behaviour for
a ratio whose denominator is uncertain. When a WTP, elasticity, or any
other ratio is the deliverable, report the simulation interval (or let
the κ guard trip the fallback automatically); the delta interval is a
linearization of something visibly curved.

```{note}
`wtp()` builds the ratio from two `dydx` calls and composes them.
If subgroup κ values straddle the fallback threshold (so one slice
falls back to simulation while another stays on the delta method),
composition refuses to mix inference methods. Pin the method
explicitly — `method="simulation"` on the estimator — whenever you
compute WTP across subgroups.
```

## Where to next

- [](../tutorials/mnlogit.md) — the multinomial logit tutorial, where
  `wtp()` is introduced and applied per alternative.
- [](../howto/elasticities.md) — the other ratio-of-derivatives
  estimand, with the same joint-inference treatment.
- [](../explanations/kappa_diagnostic.md) — what κ measures and when
  it forces the simulation fallback.