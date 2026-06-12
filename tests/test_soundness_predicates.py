"""Tests for soundness predicates (W2.3).

Each predicate gets one tripping case (assert severity + message) and one
passing case.
"""

from __future__ import annotations

import numpy as np

from pymargins._soundness._predicates import (
    CompileReport,
    Severity,
    check_ci_method_compatibility,
    check_cluster_count,
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
