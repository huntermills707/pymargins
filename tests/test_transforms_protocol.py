"""Tests for the Stage protocol and IdentityStage (Phase 0)."""

from __future__ import annotations

import pandas as pd
import pytest

from pymargins._transforms import IdentityStage, Stage


class _MalformedStage:
    """Missing required attributes — should fail protocol check."""

    def prepare(self, data):
        return data

    def prepare_resample(self, data):
        return data


class _CustomStage:
    requires_resampling = True
    alters_rows = True
    emits_columns = ("aux",)
    source_data = None

    def prepare(self, data):
        return data

    def prepare_resample(self, data):
        return data


def test_identity_stage_attribute_defaults():
    s = IdentityStage()
    assert s.requires_resampling is False
    assert s.alters_rows is False
    assert s.emits_columns == ()
    assert s.source_data is None


def test_identity_stage_prepare_returns_input():
    s = IdentityStage()
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert s.prepare(df) is df


def test_identity_stage_prepare_resample_returns_input():
    s = IdentityStage()
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert s.prepare_resample(df) is df


def test_malformed_stage_missing_attr_fails_protocol():
    s = _MalformedStage()
    # Missing 'requires_resampling', 'alters_rows', etc.
    assert not isinstance(s, Stage)


def test_custom_stage_passes_protocol():
    s = _CustomStage()
    assert isinstance(s, Stage)


def test_protocol_duck_typing():
    """Stage is a runtime-checkable Protocol — duck typing works."""
    s = IdentityStage()
    assert isinstance(s, Stage)
    assert hasattr(s, "prepare")
    assert hasattr(s, "prepare_resample")
    assert hasattr(s, "requires_resampling")
    assert hasattr(s, "alters_rows")
    assert hasattr(s, "emits_columns")
    assert hasattr(s, "source_data")
