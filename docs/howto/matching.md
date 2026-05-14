# Matching support

`pymargins.PysmatchClient` integrates with the
[`pysmatch`](https://pypi.org/project/pysmatch/) propensity-score
matcher. The client wraps a fitted matcher, exposes the matched
sample, and offers a `rematch` method for sensitivity analysis.

```python
import pandas as pd
from pymargins import Margins, PysmatchClient

client = PysmatchClient(matcher)         # a fitted pysmatch.Match
matched = client.rematch(df)             # a deterministic re-pull

fit = smf.glm("y ~ treatment + x1 + x2", data=matched,
              family=sm.families.Binomial()).fit()
m = Margins.log_scale(fit, vcov="HC3", at="overall")
m.contrasts(
    scenarios=[
        {"atexog": {"treatment": 1}, "label": "matched-treated"},
        {"atexog": {"treatment": 0}, "label": "matched-control"},
    ],
    contrasts=[+1, -1],
)
```

For uncertainty that reflects matching, run a cluster bootstrap with
the matched-pair ID as the cluster — see
[](cluster_block_bootstrap.md).
