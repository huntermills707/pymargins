"""Tests for graph node hashing and contracts (W2.1)."""

from __future__ import annotations

import pandas as pd
import pytest

from pymargins._graph._node import Node, _fingerprint


def test_hash_stability_same_spelling():
    a = Node(kind="input", params=(("name", "df"),))
    b = Node(kind="input", params=(("name", "df"),))
    assert a.hash == b.hash


def test_hash_param_order_independence():
    a = Node(kind="input", params=(("a", 1), ("b", 2)))
    b = Node(kind="input", params=(("b", 2), ("a", 1)))
    assert a.hash == b.hash


def test_hash_differs_when_params_differ():
    a = Node(kind="input", params=(("a", 1),))
    b = Node(kind="input", params=(("a", 2),))
    assert a.hash != b.hash


def test_hash_differs_when_inputs_differ():
    parent_a = Node(kind="input")
    parent_b = Node(kind="input", params=(("x", 1),))
    child_a = Node(kind="trim", inputs=(parent_a,))
    child_b = Node(kind="trim", inputs=(parent_b,))
    assert child_a.hash != child_b.hash


def test_hash_stability_across_processes():
    """L4: compare against a recorded constant for a fixed toy graph."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    root = Node(kind="input", _payload=df)
    child = Node(kind="trim", inputs=(root,), params=(("lower", 0),))
    # Recorded constant for recipe version 1 — if this changes, the hash
    # recipe changed and the recipe-bump rule (plan W2.1) is triggered.
    expected = "507e868de0c3896435ed9f8a6d181d5930f398e9c83cc9a9d6dfa3e9af0dc130"
    assert child.hash == expected


def test_payload_mutation_impossible():
    """Frozen dataclass prevents mutation."""
    n = Node(kind="input")
    with pytest.raises(AttributeError):
        n.kind = "other"


def test_fingerprint_dataframe():
    df = pd.DataFrame({"x": [1, 2, 3]})
    fp1 = _fingerprint(df)
    df2 = pd.DataFrame({"x": [1, 2, 3]})
    fp2 = _fingerprint(df2)
    assert fp1 == fp2
    df3 = pd.DataFrame({"x": [1, 2, 4]})
    fp3 = _fingerprint(df3)
    assert fp1 != fp3


def test_fingerprint_callable():
    def my_fn():
        pass

    fp = _fingerprint(my_fn)
    assert "callable" in fp


def test_node_with_payload_hash():
    df = pd.DataFrame({"x": [1, 2, 3]})
    n = Node(kind="input", _payload=df)
    assert n.hash
    n2 = Node(kind="input", _payload=df)
    assert n.hash == n2.hash
