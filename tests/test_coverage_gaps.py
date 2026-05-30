"""Tests to close coverage gaps in internal modules."""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# _result._text
# ---------------------------------------------------------------------------


def test_summary_string_repr():
    from pymargins._result._text import SummaryString

    s = SummaryString("hello\nworld")
    assert repr(s) == "hello\nworld"


# ---------------------------------------------------------------------------
# _result._test
# ---------------------------------------------------------------------------


def test_testresult_summary_multi_statistic():
    from pymargins._result._test import TestResult

    tr = TestResult(
        statistic=np.array([2.0, 3.0]),
        pvalue=np.array([0.05, 0.01]),
        df=2,
        null_value=np.array([0.0, 0.0]),
        alternative="two-sided",
        method="joint_wald",
        estimand_metadata={},
    )
    summary = tr.summary()
    assert "joint_wald" in summary
    assert "[0] stat=2.0000" in summary
    assert "[1] stat=3.0000" in summary


def test_adjustedresults_summary():
    from pymargins._result._test import AdjustedResults

    adj = AdjustedResults(
        results=None,
        p_raw=np.array([0.01, 0.05]),
        p_adj=np.array([0.02, 0.10]),
        reject=np.array([True, False]),
        method="holm",
        alpha=0.05,
    )
    summary = adj.summary()
    assert "holm" in summary
    assert "0.01" in summary


# ---------------------------------------------------------------------------
# _scenarios
# ---------------------------------------------------------------------------


def resolver(data, meta):
    return data


def test_expand_scenario_explicit_data():
    from pymargins._scenarios import expand_scenario

    base = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    meta = {
        "x": type("V", (), {"var_type": "continuous", "name": "x"})(),
        "y": type("V", (), {"var_type": "continuous", "name": "y"})(),
    }
    custom = pd.DataFrame({"x": [10, 20], "y": [40, 50]})
    expanded, md = expand_scenario({"data": custom}, base, resolver, meta)
    assert len(expanded) == 2
    assert md["strategy"] == "explicit"


def test_expand_scenario_unknown_variable_raises():
    from pymargins._scenarios import expand_scenario

    base = pd.DataFrame({"x": [1, 2, 3]})
    meta = {
        "x": type("V", (), {"var_type": "continuous", "name": "x"})(),
    }
    with pytest.raises(ValueError, match="Unknown variable"):
        expand_scenario({"atexog": {"z": 1}}, base, resolver, meta)


def test_expand_scenario_empty_training_data_raises():
    from pymargins._scenarios import expand_scenario

    base = pd.DataFrame({"x": []})
    meta = {}
    with pytest.raises(ValueError, match="empty training data"):
        expand_scenario({}, base, resolver, meta)


def test_expand_scenario_polars_backend():
    polars = pytest.importorskip("polars")
    from pymargins._scenarios import expand_scenario
    from pymargins._tabular import PolarsTabular

    base = PolarsTabular(polars.DataFrame({"x": [1.0, 2.0, 3.0]}))
    meta = {
        "x": type("V", (), {"var_type": "continuous", "name": "x"})(),
    }
    expanded, md = expand_scenario({"atexog": {"x": [10, 20]}}, base, resolver, meta)
    assert len(expanded) == 6  # 3 rows * 2 grid values
    assert md["n_grid_points"] == 2


def test_expand_scenario_polars_no_grid():
    polars = pytest.importorskip("polars")
    from pymargins._scenarios import expand_scenario
    from pymargins._tabular import PolarsTabular

    base = PolarsTabular(polars.DataFrame({"x": [1.0, 2.0, 3.0]}))
    meta = {
        "x": type("V", (), {"var_type": "continuous", "name": "x"})(),
    }
    expanded, md = expand_scenario({}, base, resolver, meta)
    assert len(expanded) == 3


def test_make_aggregation_resolver_callable():
    from pymargins._scenarios import make_aggregation_resolver

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    meta = {
        "x": type("V", (), {"var_type": "continuous", "name": "x"})(),
    }
    resolver = make_aggregation_resolver(lambda data: data.iloc[:1])
    result = resolver(df, meta)
    assert len(result) == 1


def test_resolve_var_spec_dict_with_default():
    from pymargins._scenarios import _resolve_var_spec

    info = type("V", (), {"var_type": "continuous", "name": "x"})()
    assert _resolve_var_spec({"y": "mean", "_default": "median"}, "x", info) == "median"
    assert _resolve_var_spec({"x": "mean", "_default": "median"}, "x", info) == "mean"


def test_summarize_column_min_max():
    from pymargins._scenarios import _summarize_column

    info = type("V", (), {"var_type": "continuous", "name": "x"})()
    col = pd.Series([1.0, 2.0, 3.0])
    assert _summarize_column(col, "min", info, None) == 1.0
    assert _summarize_column(col, "max", info, None) == 3.0


def test_summarize_column_percentile():
    from pymargins._scenarios import _summarize_column

    info = type("V", (), {"var_type": "continuous", "name": "x"})()
    col = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _summarize_column(col, "p25", info, None) == 1.75


def test_summarize_column_callable():
    from pymargins._scenarios import _summarize_column

    info = type("V", (), {"var_type": "continuous", "name": "x"})()
    col = pd.Series([1.0, 2.0, 3.0])
    assert _summarize_column(col, np.sum, info, None) == 6.0


def test_summarize_column_unknown_spec_raises():
    from pymargins._scenarios import _summarize_column

    info = type("V", (), {"var_type": "continuous", "name": "x"})()
    col = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unknown variable summary spec"):
        _summarize_column(col, "foobar", info, None)


def test_summarize_column_mode_on_continuous_raises():
    from pymargins._scenarios import _summarize_column

    info = type("V", (), {"var_type": "continuous", "name": "x"})()
    col = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Mode requested for continuous"):
        _summarize_column(col, "mode", info, None)


def test_weighted_mean_nan_raises():
    from pymargins._scenarios import _weighted_mean

    col = pd.Series([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        _weighted_mean(col, None)


def test_weighted_mean_bad_weights_raises():
    from pymargins._scenarios import _weighted_mean

    col = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="NaN or Inf"):
        _weighted_mean(col, np.array([1.0, np.nan, 1.0]))


def test_weighted_quantile_uniform_weights():
    from pymargins._scenarios import _weighted_quantile

    col = pd.Series([1.0, 2.0, 3.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    # Uniform weights should delegate to np.quantile
    result = _weighted_quantile(col, 0.5, w)
    assert result == 2.5


def test_weighted_quantile_zero_weights_raises():
    from pymargins._scenarios import _weighted_quantile

    col = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="sum to zero"):
        _weighted_quantile(col, 0.5, np.array([0.0, 0.0, 0.0]))


def test_weighted_mode_with_weights():
    from pymargins._scenarios import _weighted_mode

    col = pd.Series(["a", "b", "a", "b"])
    w = np.array([1.0, 3.0, 1.0, 1.0])
    assert _weighted_mode(col, w) == "b"


def test_weighted_mode_nan_raises():
    from pymargins._scenarios import _weighted_mode

    col = pd.Series(["a", None, "b"])
    with pytest.raises(ValueError, match="NaN or Inf"):
        _weighted_mode(col, None)


# ---------------------------------------------------------------------------
# _tabular
# ---------------------------------------------------------------------------


def test_pandas_tabular_init_raises_on_non_dataframe():
    from pymargins._tabular import PandasTabular

    with pytest.raises(TypeError, match="pd.DataFrame"):
        PandasTabular([1, 2, 3])


def test_pandas_tabular_dtypes():
    from pymargins._tabular import PandasTabular

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tab = PandasTabular(df)
    dtypes = tab.dtypes()
    assert dtypes["a"] == np.dtype("int64")


def test_pandas_tabular_iloc_bool_mask():
    from pymargins._tabular import PandasTabular

    df = pd.DataFrame({"a": [1, 2, 3]})
    tab = PandasTabular(df)
    sub = tab.iloc(np.array([True, False, True]))
    np.testing.assert_array_equal(sub["a"], [1, 3])


def test_pandas_tabular_groupby():
    from pymargins._tabular import PandasTabular

    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    tab = PandasTabular(df)
    groups = list(tab.groupby(["g"]))
    assert len(groups) == 2
    keys = [g for g, _ in groups]
    # pandas may return tuples for single-key groupby
    assert set(keys) == {"a", "b"} or set(keys) == {("a",), ("b",)}


def test_polars_tabular_iloc_negative_index():
    polars = pytest.importorskip("polars")
    from pymargins._tabular import PolarsTabular

    df = polars.DataFrame({"a": [1, 2, 3]})
    tab = PolarsTabular(df)
    sub = tab.iloc(-1)
    assert len(sub) == 1
    np.testing.assert_array_equal(sub["a"], [3])


def test_polars_tabular_to_jax_dict():
    polars = pytest.importorskip("polars")
    from pymargins._tabular import PolarsTabular

    df = polars.DataFrame({"a": [1, 2, 3]})
    tab = PolarsTabular(df)
    d = tab.to_jax_dict()
    assert "a" in d
    np.testing.assert_array_equal(np.asarray(d["a"]), [1, 2, 3])


def test_polars_tabular_with_column_jax_array():
    import jax.numpy as jnp

    polars = pytest.importorskip("polars")
    from pymargins._tabular import PolarsTabular

    df = polars.DataFrame({"a": [1, 2, 3]})
    tab = PolarsTabular(df)
    tab2 = tab.with_column("a", jnp.array([10, 20, 30]))
    np.testing.assert_array_equal(tab2["a"], [10, 20, 30])


# ---------------------------------------------------------------------------
# _formula
# ---------------------------------------------------------------------------


def test_formulaspec_type_validation():
    from pymargins._formula import FormulaSpec

    with pytest.raises(TypeError, match="formula must be a string"):
        FormulaSpec(123, pd.DataFrame({"x": [1]}))

    with pytest.raises(TypeError, match="training_data must be a pandas DataFrame"):
        FormulaSpec("y ~ x", [1, 2, 3])

    with pytest.raises(ValueError, match="empty"):
        FormulaSpec("y ~ x", pd.DataFrame({"x": []}))


# ---------------------------------------------------------------------------
# _inference._dispatch
# ---------------------------------------------------------------------------


def test_run_inference_unsupported_method():
    # MagicMock imported at top
    from pymargins._inference._config import InferenceConfig
    from pymargins._inference._dispatch import run_inference

    adapter = MagicMock()
    adapter.supported_inference_methods = {"delta", "simulation"}
    adapter.coefficients.return_value = jnp.array([1.0, 2.0])
    config = InferenceConfig(method="bootstrap")
    with pytest.raises(ValueError, match="does not support method"):
        run_inference(lambda b, X: b[0], adapter, config)


def test_run_test_draws_greater():
    from pymargins._inference._dispatch import run_test

    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000,))
    stat, p = run_test(
        estimate=np.array([0.5]),
        grad=None,
        cov_params=None,
        draws=draws,
        null_value=0.0,
        alternative="greater",
    )
    assert 0 <= float(p) <= 1


def test_run_test_draws_less():
    from pymargins._inference._dispatch import run_test

    rng = np.random.default_rng(42)
    draws = rng.standard_normal((1000,))
    stat, p = run_test(
        estimate=np.array([0.5]),
        grad=None,
        cov_params=None,
        draws=draws,
        null_value=0.0,
        alternative="less",
    )
    assert 0 <= float(p) <= 1


# ---------------------------------------------------------------------------
# _formula
# ---------------------------------------------------------------------------


def test_formulaspec_verify_fails():
    # MagicMock imported at top
    from pymargins._formula import FormulaSpec

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    spec = FormulaSpec("y ~ x", df)
    adapter = MagicMock()
    adapter.results = MagicMock()
    # Return wildly different fitted values
    adapter.results.fittedvalues = np.array([100.0, 200.0, 300.0])
    adapter.results.model = MagicMock()
    adapter.results.model.exog_names = ["Intercept", "x"]
    with pytest.raises(ValueError, match="Formula verification failed"):
        spec.verify_against(adapter)


# ---------------------------------------------------------------------------
# _estimands
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# margins/_session validation errors
# ---------------------------------------------------------------------------


def test_session_n_jobs_invalid():
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit = smf.ols("y ~ x", data=df).fit()
    from pymargins import Margins

    with pytest.raises(ValueError, match="n_jobs must be a positive integer"):
        Margins.linear_scale(fit, n_jobs=0)


def test_session_level_out_of_range():
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit = smf.ols("y ~ x", data=df).fit()
    from pymargins import Margins

    with pytest.raises(ValueError, match="level must be in"):
        Margins.linear_scale(fit, level=1.5)


def test_session_n_sim_invalid():
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit = smf.ols("y ~ x", data=df).fit()
    from pymargins import Margins

    with pytest.raises(ValueError, match="n_sim must be a positive integer"):
        Margins.linear_scale(fit, method="simulation", n_sim=-1)


def test_session_n_boot_invalid():
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit = smf.ols("y ~ x", data=df).fit()
    from pymargins import Margins

    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        Margins.linear_scale(fit, method="bootstrap", n_boot=-1)


def test_session_fd_step_invalid():
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit = smf.ols("y ~ x", data=df).fit()
    from pymargins import Margins

    with pytest.raises(ValueError, match="fd_step must be a positive finite float"):
        Margins.linear_scale(fit, fd_step=-0.01)


def test_session_phi_without_phi_inv():
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    fit = smf.ols("y ~ x", data=df).fit()
    from pymargins import Margins

    with pytest.raises(ValueError, match="phi and phi_inv must be provided together"):
        Margins(fit, phi=lambda x: x)
