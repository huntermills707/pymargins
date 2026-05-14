# Robust and clustered standard errors

Pass the variance estimator at session construction. The session
treats `vcov=` as a session-level commitment — every subsequent call
inherits it.

```python
from pymargins import Margins

# Heteroskedastic-robust HC3
m = Margins.log_scale(fit, vcov="HC3")

# Cluster-robust
m = Margins.log_scale(fit, vcov={"type": "cluster", "groups": df["firm"]})

# HAC (Newey–West) for time series
m = Margins.linear_scale(fit, vcov={"type": "HAC", "maxlags": 4})

# Bring your own Σ̂
m = Margins.log_scale(fit, vcov=my_sandwich)
```

When the model already carries a robust covariance (e.g. a
`PanelOLS` fit with `cov_type="clustered"`), you can omit `vcov=`
and the adapter will pick up the fit-time covariance.
