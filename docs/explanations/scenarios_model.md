# The scenarios model — `at` vs `atexog`

Two knobs control *where in covariate space* a margin is evaluated.
They live at different levels of the API.

## `at` — session-level aggregation rule

`at=` is set at session construction. It controls the default
evaluation rule for variables *not* otherwise pinned:

| Value          | Per-variable behavior                                     |
|----------------|-----------------------------------------------------------|
| `"overall"`    | take the observed value on every row, then average        |
| `"typical"`    | median for continuous, mode for discrete                  |
| `"mean"`       | mean for all (errors on non-numeric)                      |
| `"median"`     | median for all                                            |
| `"mode"`       | mode for all (errors on continuous)                       |
| dict           | per-variable override, with `_default` as a fallback rule |
| callable       | `(data) -> 1-row DataFrame`, fully bespoke                |

`"overall"` corresponds to Stata's bare `margins` and gives AAP / AME.
`"typical"` corresponds to `margins, atmeans` for mixed factor /
continuous models and gives APM / MEM.

## `atexog` — per-call counterfactual pins

`atexog=` is a per-call dict (or a list of dicts wrapped as
`scenarios=`) that pins specific variables to specific values. A
list-valued entry produces a Cartesian product (a grid). Variables
not mentioned in `atexog` follow the session's `at=` rule.

```python
# AAP at age=25, 45, 65, averaging the rest over the sample
Margins.log_scale(fit, at="overall").predict(
    atexog={"age": [25, 45, 65]}
)

# APR at age=25, 45, 65, others held at typical profile
Margins.log_scale(fit, at="typical").predict(
    atexog={"age": [25, 45, 65]}
)
```

## Why split it this way?

Because the aggregation choice is a methodological commitment (AME
vs MEM is an argument; if you flip mid-analysis the audit trail
should show it) but the counterfactual pins are not (you genuinely
do want to evaluate the same AME at several age points).

This is the same logic behind keeping `phi`, `vcov`, `level`, and
`method` session-level: the analytical *posture* belongs in the
constructor; the analytical *question* belongs in the method call.

See [](session_precommitment.md).
