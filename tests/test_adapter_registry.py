"""Tests for the third-party adapter registry (A11)."""

import jax.numpy as jnp
import pytest

from pymargins import ModelAdapter, register_adapter
from pymargins._adapters import _DETECTION_REGISTRY, _detect_adapter_class

# ---------------------------------------------------------------------------
# Dummy adapter and model for testing
# ---------------------------------------------------------------------------


class DummyModel:
    """Fake fitted model for testing auto-detection."""

    pass


class DummyAdapter(ModelAdapter):
    """Fake adapter for testing registration."""

    def coefficients(self):
        return jnp.array([1.0])

    def covariance(self, vcov_spec=None):
        return jnp.array([[1.0]])

    def predict(self, beta, X, offset=None):
        return X @ beta

    def design_matrix_from_df(self, df):
        return jnp.asarray(df.values)

    def variable_metadata(self):
        return {}

    def column_index_of_variable(self, var):
        raise NotImplementedError

    def supported_inference_methods(self):
        return {"delta", "simulation", "bootstrap"}


class AnotherDummyModel:
    pass


class AnotherDummyAdapter(ModelAdapter):
    def coefficients(self):
        return jnp.array([2.0])

    def covariance(self, vcov_spec=None):
        return jnp.array([[1.0]])

    def predict(self, beta, X, offset=None):
        return X @ beta

    def design_matrix_from_df(self, df):
        return jnp.asarray(df.values)

    def variable_metadata(self):
        return {}

    def column_index_of_variable(self, var):
        raise NotImplementedError

    def supported_inference_methods(self):
        return {"delta", "simulation", "bootstrap"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear the detection registry before each test and restore after."""
    orig = list(_DETECTION_REGISTRY)
    _DETECTION_REGISTRY.clear()
    yield
    _DETECTION_REGISTRY.clear()
    _DETECTION_REGISTRY.extend(orig)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_adapter_with_predicate():
    register_adapter(
        DummyAdapter,
        predicate=lambda m: isinstance(m, DummyModel),
        hint_modules=["tests."],
        hint_names=["DummyModel"],
        description="Dummy model for testing",
    )
    assert _detect_adapter_class(DummyModel()) is DummyAdapter


def test_register_adapter_with_hints():
    # Use the actual module prefix at runtime (pytest may import as
    # "test_adapter_registry" rather than "tests.test_adapter_registry").
    actual_module = type(AnotherDummyModel()).__module__
    register_adapter(
        AnotherDummyAdapter,
        hint_modules=[actual_module],
        hint_names=["AnotherDummyModel"],
        description="Another dummy model",
    )
    assert _detect_adapter_class(AnotherDummyModel()) is AnotherDummyAdapter


def test_register_adapter_order_matters():
    """Earlier registrations take precedence."""
    register_adapter(
        DummyAdapter,
        predicate=lambda m: isinstance(m, DummyModel),
        hint_modules=["tests."],
        hint_names=["DummyModel"],
    )
    register_adapter(
        AnotherDummyAdapter,
        predicate=lambda m: isinstance(m, DummyModel),
        hint_modules=["tests."],
        hint_names=["DummyModel"],
    )
    # First registration wins
    assert _detect_adapter_class(DummyModel()) is DummyAdapter


def test_register_adapter_requires_predicate_or_hints():
    with pytest.raises(ValueError, match="predicate"):
        register_adapter(DummyAdapter)


def test_register_adapter_appears_in_suggestions():
    from pymargins._adapters import _suggest_adapters

    register_adapter(
        DummyAdapter,
        predicate=lambda m: isinstance(m, DummyModel),
        hint_modules=["tests."],
        hint_names=["DummyModel"],
        description="Dummy model for testing",
    )
    suggestion = _suggest_adapters("DummyModel", "tests.test_adapter_registry")
    assert "DummyAdapter" in suggestion
    assert "Dummy model for testing" in suggestion


def test_register_adapter_can_override_builtin():
    """Registered adapters are checked before built-in dispatch."""
    import pandas as pd
    import statsmodels.api as sm

    df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    fitted = sm.OLS.from_formula("y ~ x", data=df).fit()

    # Register an adapter that claims OLS results
    register_adapter(
        DummyAdapter,
        predicate=lambda m: (
            type(m).__module__.startswith("statsmodels.")
            and "RegressionResultsWrapper" in type(m).__name__
        ),
        hint_modules=["statsmodels."],
        hint_names=["RegressionResultsWrapper"],
    )

    assert _detect_adapter_class(fitted) is DummyAdapter


def test_register_adapter_description_defaults_to_class_name():
    from pymargins._adapters import _REGISTERED_ADAPTERS

    register_adapter(
        DummyAdapter,
        predicate=lambda m: isinstance(m, DummyModel),
        hint_modules=["tests."],
        hint_names=["DummyModel"],
    )
    entry = _REGISTERED_ADAPTERS[-1]
    assert entry["description"] == "DummyAdapter"
    assert entry["name"] == "DummyAdapter"
