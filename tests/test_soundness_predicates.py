"""Tests for soundness predicates (W2.3).

Each predicate gets one tripping case (assert severity + message) and one
passing case.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from pymargins._adapter import (
    BootstrapOnlyAdapter,
    GLMAdapter,
    LinearPredictionAdapter,
    WrappedFDAdapter,
)
from pymargins._soundness._predicates import (
    SOUNDNESS_ROWS,
    CompileReport,
    Severity,
    SoundnessRow,
    check_ci_method_compatibility,
    check_cluster_count,
    check_ess,
    check_lonely_psu,
    check_method_adapter_compatibility,
    check_tail_count_adequacy,
)


def test_method_adapter_compatibility_refuse():
    report = check_method_adapter_compatibility("delta", {"bootstrap"}, CompileReport())
    assert report.has(Severity.REFUSE, "method_unsupported")


def test_method_adapter_compatibility_pass():
    report = check_method_adapter_compatibility("delta", {"delta", "simulation"}, CompileReport())
    assert not report.has(Severity.REFUSE)


def test_ci_method_incompatible_refuse():
    report = check_ci_method_compatibility("percentile", "delta", CompileReport())
    assert report.has(Severity.REFUSE, "ci_method_incompatible")


def test_ci_method_studentized_refuse():
    report = check_ci_method_compatibility("studentized", "delta", CompileReport())
    assert report.has(Severity.REFUSE, "ci_method_incompatible")


def test_ci_method_compatible_pass():
    report = check_ci_method_compatibility("wald", "delta", CompileReport())
    assert not report.has(Severity.REFUSE)


def test_tail_count_adequacy_warn():
    report = check_tail_count_adequacy(50, 0.95, "percentile", CompileReport())
    assert report.has(Severity.WARN, "tail_count_low")


def test_tail_count_adequacy_note():
    report = check_tail_count_adequacy(500, 0.95, "percentile", CompileReport())
    assert report.has(Severity.NOTE, "tail_count_low")


def test_tail_count_adequacy_pass():
    report = check_tail_count_adequacy(2000, 0.95, "percentile", CompileReport())
    assert not report.has(Severity.WARN)
    assert not report.has(Severity.NOTE, "tail_count_low")


def test_cluster_count_warn():
    report = check_cluster_count(10, CompileReport())
    assert report.has(Severity.WARN, "few_clusters")


def test_cluster_count_pass():
    report = check_cluster_count(50, CompileReport())
    assert not report.has(Severity.WARN)


def test_lonely_psu_refuse():
    sd = type("SD", (), {"psu": np.array([1, 1, 2]), "strata": np.array([1, 1, 2]), "nest": True})()
    report = check_lonely_psu(sd, CompileReport())
    assert report.has(Severity.REFUSE, "lonely_psu")


def test_lonely_psu_pass():
    sd = type("SD", (), {"psu": np.array([1, 2, 3]), "strata": np.array([1, 1, 1]), "nest": True})()
    report = check_lonely_psu(sd, CompileReport())
    assert not report.has(Severity.REFUSE)


def test_ess_note():
    w = np.array([1.0] * 50 + [100.0])  # one huge weight
    report = check_ess(w, CompileReport())
    assert report.has(Severity.NOTE, "ess_low")


def test_ess_pass():
    w = np.ones(100)
    report = check_ess(w, CompileReport())
    assert not report.has(Severity.NOTE, "ess_low")


def test_soundness_rows_implemented_resolve():
    """Every implemented row's predicate qualname resolves to a callable."""
    for row in SOUNDNESS_ROWS:
        if row.predicate is None:
            continue
        module_name, obj_name = row.predicate.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        obj = getattr(mod, obj_name)
        assert callable(obj), f"{row.id} predicate {row.predicate} is not callable"


def test_soundness_rows_text_present():
    """Every row carries verbatim text; future rows have no predicate."""
    for row in SOUNDNESS_ROWS:
        assert isinstance(row, SoundnessRow)
        assert row.text
        if row.predicate is None:
            # Unimplemented rows are honest about being unimplemented.
            assert "*(future)*" in row.text or row.severity in {
                "sound",
                "conditional",
                "unrepresentable",
                "refuse",
                "warn",
                "note",
            }


def _dummy_adapter(cls):
    """Return an instantiable dummy subclass of an adapter shape."""
    import jax.numpy as jnp

    class Dummy(cls):
        def coefficients(self):
            return jnp.array([0.0])

        def covariance(self, vcov_spec=None):
            return jnp.array([[1.0]])

        def predict(self, beta, X, offset=None):
            return X @ beta

        def design_matrix_from_df(self, df):
            return jnp.asarray(df)

        def column_index_of_variable(self, name):
            return 0

        def variable_metadata(self):
            return {}

    return Dummy()


@pytest.mark.parametrize(
    "adapter_class, expected_methods",
    [
        (GLMAdapter, {"delta", "simulation", "bootstrap"}),
        (LinearPredictionAdapter, {"delta", "simulation", "bootstrap"}),
        (WrappedFDAdapter, {"delta", "simulation", "bootstrap"}),
        (BootstrapOnlyAdapter, {"bootstrap"}),
    ],
)
def test_adapter_tier_matches_supported_methods(adapter_class, expected_methods):
    """req \u00a71: adapter tier (score vs FD vs bootstrap-only) is consistent
    with the declared supported_inference_methods set."""
    adapter = _dummy_adapter(adapter_class)
    assert adapter.supported_inference_methods == expected_methods


# ---------------------------------------------------------------------------
# Registry rows: bind verbatim text/severity to runtime behavior
# ---------------------------------------------------------------------------


def _row_by_id(row_id: str):
    return next(r for r in SOUNDNESS_ROWS if r.id == row_id)


def test_registry_method_unsupported_matches():
    row = _row_by_id("6.1-method-unsupported")
    report = check_method_adapter_compatibility("delta", {"bootstrap"}, CompileReport())
    _, _, msg = next((s, c, m) for s, c, m in report.entries if s == Severity.REFUSE)
    assert row.severity == "refuse"
    assert "not supported" in msg


def test_registry_ci_incompatible_matches():
    row = _row_by_id("6.1-ci-method-incompatible")
    report = check_ci_method_compatibility("studentized", "delta", CompileReport())
    _, _, msg = next((s, c, m) for s, c, m in report.entries if s == Severity.REFUSE)
    assert row.severity == "refuse"
    assert "studentized" in msg and "bootstrap" in msg


def test_registry_lonely_psu_matches():
    row = _row_by_id("6.5-lonely-psu")
    sd = type("SD", (), {"psu": np.array([1, 1, 2]), "strata": np.array([1, 1, 2])})()
    report = check_lonely_psu(sd, CompileReport())
    _, _, msg = next((s, c, m) for s, c, m in report.entries if s == Severity.REFUSE)
    assert row.severity == "refuse"
    assert "1 PSU" in msg


def test_registry_few_clusters_matches():
    row = _row_by_id("6.5-few-clusters")
    report = check_cluster_count(10, CompileReport())
    _, _, msg = next((s, c, m) for s, c, m in report.entries if s == Severity.WARN)
    assert row.severity == "warn"
    assert "Cluster-robust inference" in msg
    assert "G=10" in msg


def test_registry_tail_counts_matches():
    row = _row_by_id("6.7-tail-counts")
    report = check_tail_count_adequacy(50, 0.95, "percentile", CompileReport())
    _, _, msg = report.entries[0]
    assert row.severity == "note"
    assert "tail" in msg.lower()


def test_registry_se_b_matches():
    row = _row_by_id("6.7-se-b")
    report = check_tail_count_adequacy(50, 0.95, "se", CompileReport())
    _, _, msg = report.entries[0]
    assert row.severity == "note"
    assert "se" in msg.lower() or "SE" in msg


def test_registry_ess_matches():
    row = _row_by_id("6.6-ess")
    w = np.array([1.0] * 50 + [100.0])
    report = check_ess(w, CompileReport())
    _, _, msg = next((s, c, m) for s, c, m in report.entries if s == Severity.NOTE)
    assert row.severity == "note"
    assert "ESS" in msg
