"""Tests for pool_imputations (Rubin pooling combinator)."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from pymargins import ImputationDiagnostic, Margins, pool_imputations
from pymargins._result import MarginsResult


def _mk(
    est,
    se,
    *,
    level=0.95,
    labels=("dydx",),
    phi=None,
    phi_inv=None,
    method="delta",
    **meta,
):
    est = np.asarray(est, float)
    md = {"labels": list(labels), "kind": "slope", "at": "overall"}
    md.update(meta)  # allow overriding kind/at/scenarios/over for guard tests
    return MarginsResult(
        estimate=est,
        std_error=np.asarray(se, float),
        conf_int_lower=est,
        conf_int_upper=est,
        method=method,
        level=level,
        phi=phi,
        phi_inv=phi_inv,
        estimand_metadata=md,
    )


# ---------------------------------------------------------------------------
# Correctness (the math)
# ---------------------------------------------------------------------------


def test_rubin_scalar_known_answer():
    ests, ses = [1.0, 1.2, 0.8, 1.1, 0.9], [0.20, 0.22, 0.19, 0.21, 0.20]
    M = len(ests)
    pooled = pool_imputations([_mk(e, s) for e, s in zip(ests, ses, strict=False)])
    W, B = np.mean(np.square(ses)), np.var(ests, ddof=1)
    T = W + (1 + 1 / M) * B
    r = (1 + 1 / M) * B / W
    df = (M - 1) * (1 + 1 / r) ** 2
    assert float(pooled.estimate) == pytest.approx(np.mean(ests))
    assert float(pooled.std_error) == pytest.approx(np.sqrt(T))
    assert float(pooled.imputation_diagnostic.df) == pytest.approx(df)
    t = stats.t.ppf(0.975, df)
    assert float(pooled.conf_int_upper) == pytest.approx(np.mean(ests) + t * np.sqrt(T))


def test_zero_between_reduces_to_single_fit():
    pooled = pool_imputations([_mk(1.0, 0.2) for _ in range(4)])
    assert float(pooled.std_error) == pytest.approx(0.2)
    assert np.isinf(pooled.imputation_diagnostic.df)
    assert float(pooled.imputation_diagnostic.fmi) == pytest.approx(0.0)
    assert float(pooled.conf_int_upper) == pytest.approx(
        1.0 + stats.norm.ppf(0.975) * 0.2
    )


def test_pools_on_inference_scale_not_reporting():
    import jax.numpy as jnp

    ratios = [1.5, 2.0, 1.8]
    pooled = pool_imputations(
        [_mk(r, 0.1, phi=jnp.exp, phi_inv=jnp.log, labels=("RR",)) for r in ratios]
    )
    assert float(pooled.estimate) == pytest.approx(np.exp(np.mean(np.log(ratios))))
    assert float(pooled.estimate) != pytest.approx(np.mean(ratios))


def test_t_not_z_for_small_M_with_between_variance():
    pooled = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    half = float(pooled.conf_int_upper - pooled.estimate)
    df = float(pooled.imputation_diagnostic.df)
    assert half == pytest.approx(stats.t.ppf(0.975, df) * float(pooled.std_error))
    assert stats.t.ppf(0.975, df) > stats.norm.ppf(0.975)


def test_vector_pooling_is_per_component():
    res = [
        _mk([1.0 + 0.1 * m, 2.0 - 0.1 * m], [0.2, 0.3], labels=("a", "b"))
        for m in range(5)
    ]
    pooled = pool_imputations(res)
    assert pooled.estimate.shape == (2,)
    assert pooled.imputation_diagnostic.fmi.shape == (2,)


def test_method_agnostic_mixes_delta_and_bootstrap():
    a = _mk(1.0, 0.20, method="delta")
    b = _mk(1.3, 0.25, method="bootstrap")
    pooled = pool_imputations([a, b, _mk(1.1, 0.21, method="simulation")])
    assert pooled.method == "pooled"


# ---------------------------------------------------------------------------
# Commensurability guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad,match",
    [
        (lambda: [_mk(1.0, 0.2, labels=("a",)), _mk(1.0, 0.2, labels=("b",))], "label"),
        (
            lambda: [
                _mk(1.0, 0.2),
                _mk(1.0, 0.2, phi=np.exp, phi_inv=np.log),
            ],
            "scale|phi",
        ),
        (lambda: [_mk(1.0, 0.2, level=0.95), _mk(1.0, 0.2, level=0.90)], "level"),
        (
            lambda: [
                _mk([1.0, 2.0], [0.2, 0.2], labels=("a", "b")),
                _mk(1.0, 0.2),
            ],
            "shape",
        ),
        # Same labels/shape/level/scale, but a different estimand (F1):
        (lambda: [_mk(1.0, 0.2), _mk(1.0, 0.2, at={"z": 5})], "at"),
        (lambda: [_mk(1.0, 0.2), _mk(1.0, 0.2, kind="prediction")], "kind"),
        (
            lambda: [
                _mk(1.0, 0.2, scenarios=[{}]),
                _mk(1.0, 0.2, scenarios=[{"a": 1}]),
            ],
            "scenario",
        ),
        (
            lambda: [_mk(1.0, 0.2, over="g"), _mk(1.0, 0.2, over="h")],
            "over",
        ),
        (lambda: [_mk(1.0, 0.2)], "at least two"),
        (lambda: [_mk(np.nan, 0.2), _mk(1.0, 0.2)], "finite"),
    ],
)
def test_incommensurable_inputs_raise(bad, match):
    with pytest.raises(ValueError, match=match):
        pool_imputations(bad())


def test_negative_se_raises():
    with pytest.raises(ValueError, match="negative"):
        pool_imputations([_mk(1.0, -0.2), _mk(1.0, 0.2)])


def test_shared_nonscalar_metadata_pools_ok():
    # Identical nested at-grid + scenarios across imputations must NOT trip the
    # estimand-identity guard (F1), even though the point estimates differ.
    md = dict(at={"z": 0.0}, scenarios=[{"x": 1}, {"x": 2}])
    res = [_mk(1.0 + 0.1 * m, 0.2, **md) for m in range(4)]
    pooled = pool_imputations(res)
    assert pooled.method == "pooled"
    assert pooled.imputation_diagnostic.n_imputations == 4


# ---------------------------------------------------------------------------
# Diagnostics, persistence, boundary
# ---------------------------------------------------------------------------


def test_fmi_increases_with_between_variance():
    tight = pool_imputations([_mk(e, 0.2) for e in (0.98, 1.0, 1.02)])
    spread = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    assert float(spread.imputation_diagnostic.fmi) > float(
        tight.imputation_diagnostic.fmi
    )
    assert 0.0 <= float(spread.imputation_diagnostic.fmi) <= 1.0


def test_summary_footer_reports_M_and_fmi():
    s = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)]).summary()
    assert "M=3" in s and "FMI" in s


def test_pooled_roundtrips_to_disk(tmp_path):
    pooled = pool_imputations([_mk(1.0 + 0.1 * m, 0.2) for m in range(4)])
    pooled.to_disk(tmp_path / "p.pkl")
    loaded = MarginsResult.from_disk(tmp_path / "p.pkl")
    assert loaded.imputation_diagnostic.n_imputations == 4


def test_pooled_result_is_a_leaf():
    pooled = pool_imputations([_mk(1.0 + 0.1 * m, 0.2) for m in range(3)])
    assert pooled.session is None and pooled.gradient is None and pooled.draws is None
    with pytest.raises(ValueError):
        _ = pooled - pooled


# ---------------------------------------------------------------------------
# Barnard–Rubin small-sample correction (complete_df)
# ---------------------------------------------------------------------------


def test_complete_df_zero_between_no_runtimewarning():
    # F2: B<=0 with complete_df set must not leak a numpy divide warning
    # (df_old=inf ⇒ inf/inf), and df must collapse to the complete-data df.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pooled = pool_imputations([_mk(1.0, 0.2) for _ in range(4)], complete_df=20.0)
    assert float(pooled.imputation_diagnostic.df) == pytest.approx(20.0)


def test_complete_df_matches_barnard_rubin_formula():
    # The corrected df equals the Barnard–Rubin (1999) combination and is
    # strictly smaller than the classic Rubin df.
    ests = (0.5, 1.0, 1.5)
    base = pool_imputations([_mk(e, 0.2) for e in ests])
    corr = pool_imputations([_mk(e, 0.2) for e in ests], complete_df=20.0)
    riv = float(base.imputation_diagnostic.riv)
    df_old = float(base.imputation_diagnostic.df)
    lam = riv / (1.0 + riv)
    nu = 20.0
    df_obs = (nu + 1.0) / (nu + 3.0) * nu * (1.0 - lam)
    expected = df_old * df_obs / (df_old + df_obs)
    assert float(corr.imputation_diagnostic.df) == pytest.approx(expected)
    assert float(corr.imputation_diagnostic.df) < df_old


# ---------------------------------------------------------------------------
# Phase 2: pooled test() / conf_int() / summary rows
# ---------------------------------------------------------------------------


def test_pooled_test_returns_t_statistic():
    pooled = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    tr = pooled.test(value=0.0, null_scale="inference")
    df = float(pooled.imputation_diagnostic.df)
    t = float(tr.statistic)
    p = float(tr.pvalue)
    assert p == pytest.approx(2 * stats.t.sf(abs(t), df))
    assert tr.method == "wald"


def test_pooled_test_honors_one_sided_alternative():
    # F3: the pooled t-test must respect alternative=, not silently force
    # two-sided. Conventions mirror delta_wald_test with Student-t.
    pooled = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    df = float(pooled.imputation_diagnostic.df)
    t = float(pooled.estimate) / float(pooled.std_error)  # null=0, identity scale
    g = pooled.test(value=0.0, null_scale="inference", alternative="greater")
    lt = pooled.test(value=0.0, null_scale="inference", alternative="less")
    two = pooled.test(value=0.0, null_scale="inference", alternative="two-sided")
    assert float(g.pvalue) == pytest.approx(float(stats.t.sf(t, df)))
    assert float(lt.pvalue) == pytest.approx(float(stats.t.cdf(t, df)))
    # Complementary one-sided tails sum to one; two-sided is twice the smaller.
    assert float(g.pvalue) + float(lt.pvalue) == pytest.approx(1.0)
    assert float(two.pvalue) == pytest.approx(
        2 * min(float(g.pvalue), float(lt.pvalue))
    )


def test_pooled_conf_int_recomputes_at_new_level():
    pooled = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    lo, hi = pooled.conf_int(level=0.90)
    df = float(pooled.imputation_diagnostic.df)
    se = float(pooled.std_error)
    est = float(pooled.estimate)
    tcrit = stats.t.ppf(0.95, df)
    assert float(lo) == pytest.approx(est - tcrit * se)
    assert float(hi) == pytest.approx(est + tcrit * se)


def test_summary_rows_show_t_for_pooled():
    pooled = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    rows = pooled._summary_rows()
    assert len(rows) == 1
    assert "statistic" in rows[0]
    assert "pvalue" in rows[0]
    assert rows[0].get("stat_label") == "t"


def test_simultaneous_ci_not_implemented_for_pooled():
    pooled = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)])
    with pytest.raises(NotImplementedError):
        pooled.conf_int(level=0.95, simultaneous=True)


# ---------------------------------------------------------------------------
# End-to-end (real fits)
# ---------------------------------------------------------------------------


def test_end_to_end_pool_widens_over_within():
    """M IterativeImputer draws → M OLS fits → pool; pooled SE exceeds mean within-SE."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["y"] = 1.0 + 0.6 * df["x1"] - 0.4 * df["x2"] + rng.normal(0, 0.5, n)
    df_nan = df.copy()
    df_nan.loc[rng.uniform(size=n) < 0.30, "x1"] = np.nan

    per_imp = []
    for s in range(5):
        imp = IterativeImputer(sample_posterior=True, random_state=s, max_iter=10)
        completed = pd.DataFrame(imp.fit_transform(df_nan), columns=df_nan.columns)
        fit = smf.ols("y ~ x1 + x2", data=completed).fit()
        per_imp.append(Margins.linear_scale(fit).dydx("x1"))

    pooled = pool_imputations(per_imp)
    within = np.mean([float(r.std_error) for r in per_imp])
    assert float(pooled.std_error) > within
    assert "M=5" in pooled.summary()


# ---------------------------------------------------------------------------
# Soft warnings
# ---------------------------------------------------------------------------


def test_same_session_warning():
    r = _mk(1.0, 0.2)
    r2 = _mk(1.0, 0.2)

    # Give them the same session object to trigger the warning
    class FakeSession:
        pass

    sess = FakeSession()
    r = dataclasses.replace(r, session=sess)
    r2 = dataclasses.replace(r2, session=sess)
    with pytest.warns(UserWarning, match="same session"):
        pool_imputations([r, r2])


# ---------------------------------------------------------------------------
# F6/F7: footer branches, persistence fidelity, headers, additive contract
# ---------------------------------------------------------------------------


def test_footer_formats_scalar_and_vector():
    # F6: both branches of ImputationDiagnostic.footer() format correctly.
    scalar = ImputationDiagnostic(
        n_imputations=5,
        fmi=0.312,
        relative_efficiency=0.941,
        df=18.4,
        within_var=0.04,
        between_var=0.01,
        total_var=0.05,
        riv=0.2,
    )
    s = scalar.footer()
    assert "M=5" in s and "FMI 0.312" in s and "df 18.4" in s and "rel. eff. 0.941" in s

    vector = ImputationDiagnostic(
        n_imputations=5,
        fmi=np.array([0.10, 0.312]),
        relative_efficiency=np.array([0.99, 0.941]),
        df=np.array([120.0, 18.4]),
        within_var=np.array([0.04, 0.05]),
        between_var=np.array([0.01, 0.02]),
        total_var=np.array([0.05, 0.07]),
        riv=np.array([0.2, 0.3]),
    )
    v = vector.footer()
    # max FMI, min df, min rel. eff. across components.
    assert "FMI max=0.312" in v
    assert "df min=18.4" in v
    assert "rel. eff. min=0.941" in v


def test_vector_pool_footer_uses_max_min():
    # The vector branch is reached through the real pooling path too.
    res = [
        _mk([1.0 + 0.1 * m, 2.0 - 0.05 * m], [0.2, 0.3], labels=("a", "b"))
        for m in range(5)
    ]
    s = pool_imputations(res).summary()
    assert "FMI max=" in s and "df min=" in s and "rel. eff. min=" in s


def test_nonpooled_result_has_no_imputation_footer():
    # Additive contract: an ordinary result is untouched by the MI machinery.
    r = _mk(1.0, 0.2)
    assert r.imputation_diagnostic is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # bare result has no stats to test
        s = r.summary()
    assert "MI pooled" not in s


def test_pooled_summary_pvalue_header_is_t_not_z():
    # F7: a pooled result reports its Student-t p-value under a "P>|t|" header.
    s = pool_imputations([_mk(e, 0.2) for e in (0.5, 1.0, 1.5)]).summary()
    assert "P>|t|" in s
    assert "P>|z|" not in s


def test_pooled_roundtrip_preserves_values(tmp_path):
    # F6: the full estimate / CI / diagnostic payload survives to_disk/from_disk.
    pooled = pool_imputations([_mk(1.0 + 0.1 * m, 0.2) for m in range(4)])
    pooled.to_disk(tmp_path / "p.pkl")
    loaded = MarginsResult.from_disk(tmp_path / "p.pkl")

    assert float(loaded.estimate) == pytest.approx(float(pooled.estimate))
    assert float(loaded.std_error) == pytest.approx(float(pooled.std_error))
    assert float(loaded.conf_int_lower) == pytest.approx(float(pooled.conf_int_lower))
    assert float(loaded.conf_int_upper) == pytest.approx(float(pooled.conf_int_upper))

    want, got = pooled.imputation_diagnostic, loaded.imputation_diagnostic
    assert got.n_imputations == want.n_imputations
    for field in ("fmi", "df", "within_var", "between_var", "total_var", "riv"):
        assert float(getattr(got, field)) == pytest.approx(float(getattr(want, field)))
