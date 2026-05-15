"""Tests for pymargins.scenarios helper module."""

import pytest
import numpy as np

from pymargins.scenarios import (
    pairwise,
    reference,
    at_levels,
    grid,
    did,
    diff,
    all_pairwise,
)


# ---------------------------------------------------------------------------
# pairwise
# ---------------------------------------------------------------------------

def test_pairwise_basic():
    scenarios, contrasts = pairwise("treatment", [1, 0])
    assert len(scenarios) == 2
    assert scenarios[0]["atexog"] == {"treatment": 1}
    assert scenarios[0]["label"] == "treatment=1"
    assert scenarios[1]["atexog"] == {"treatment": 0}
    assert scenarios[1]["label"] == "treatment=0"
    assert contrasts == [+1, -1]


def test_pairwise_with_fixed():
    scenarios, contrasts = pairwise("black", [1, 0], female=0, age=40)
    assert scenarios[0]["atexog"] == {"black": 1, "female": 0, "age": 40}
    assert scenarios[1]["atexog"] == {"black": 0, "female": 0, "age": 40}
    assert contrasts == [+1, -1]


def test_pairwise_custom_label_fmt():
    scenarios, _ = pairwise("treatment", ["treated", "control"], label_fmt="{val}")
    assert scenarios[0]["label"] == "treated"
    assert scenarios[1]["label"] == "control"


def test_pairwise_too_many_values():
    with pytest.raises(ValueError, match="exactly 2 values"):
        pairwise("x", [1, 2, 3])


def test_pairwise_too_few_values():
    with pytest.raises(ValueError, match="exactly 2 values"):
        pairwise("x", [1])


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------

def test_reference_basic():
    scenarios, contrasts = reference("region", ["north", "south", "east"])
    assert len(scenarios) == 3
    assert scenarios[0]["atexog"] == {"region": "north"}
    assert scenarios[1]["atexog"] == {"region": "south"}
    assert scenarios[2]["atexog"] == {"region": "east"}

    assert "south_vs_north" in contrasts
    assert "east_vs_north" in contrasts
    assert contrasts["south_vs_north"] == [-1, +1, 0]
    assert contrasts["east_vs_north"] == [-1, 0, +1]


def test_reference_explicit_ref():
    scenarios, contrasts = reference(
        "region", ["north", "south", "east"], ref_level="south"
    )
    assert scenarios[0]["atexog"] == {"region": "south"}
    assert "north_vs_south" in contrasts
    assert contrasts["north_vs_south"] == [-1, +1, 0]


def test_reference_with_fixed():
    scenarios, contrasts = reference("region", ["A", "B"], year=2020)
    assert scenarios[0]["atexog"] == {"region": "A", "year": 2020}
    assert scenarios[1]["atexog"] == {"region": "B", "year": 2020}


def test_reference_empty_levels():
    with pytest.raises(ValueError, match="at least one level"):
        reference("x", [])


def test_reference_invalid_ref():
    with pytest.raises(ValueError, match="not found"):
        reference("x", ["A", "B"], ref_level="C")


# ---------------------------------------------------------------------------
# at_levels
# ---------------------------------------------------------------------------

def test_at_levels_basic():
    scenarios = at_levels("region", levels=["north", "south", "east"])
    assert len(scenarios) == 3
    assert scenarios[0]["atexog"] == {"region": "north"}
    assert scenarios[0]["label"] == "region=north"


def test_at_levels_with_fixed():
    scenarios = at_levels("treatment", levels=[0, 1], female=0)
    assert scenarios[0]["atexog"] == {"treatment": 0, "female": 0}
    assert scenarios[1]["atexog"] == {"treatment": 1, "female": 0}


def test_at_levels_empty():
    with pytest.raises(ValueError, match="at least one level"):
        at_levels("x", levels=[])


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_grid_basic():
    scenarios = grid(age=[30, 50], treatment=[0, 1])
    assert len(scenarios) == 4
    labels = {s["label"] for s in scenarios}
    assert labels == {
        "age=30, treatment=0",
        "age=30, treatment=1",
        "age=50, treatment=0",
        "age=50, treatment=1",
    }


def test_grid_custom_label():
    scenarios = grid(label_fmt="{age}-{treatment}", age=[30, 50], treatment=[0, 1])
    labels = {s["label"] for s in scenarios}
    assert labels == {"30-0", "30-1", "50-0", "50-1"}


def test_grid_empty():
    with pytest.raises(ValueError, match="at least one variable"):
        grid()


# ---------------------------------------------------------------------------
# did
# ---------------------------------------------------------------------------

def test_did_basic():
    scenarios, contrasts = did("treat", "post")
    assert len(scenarios) == 4
    assert scenarios[0]["atexog"] == {"treat": 1, "post": 1}
    assert scenarios[1]["atexog"] == {"treat": 1, "post": 0}
    assert scenarios[2]["atexog"] == {"treat": 0, "post": 1}
    assert scenarios[3]["atexog"] == {"treat": 0, "post": 0}
    assert contrasts == {"did": [+1, -1, -1, +1]}


def test_did_custom_levels():
    scenarios, contrasts = did(
        "treat", "post",
        treated_level="yes", control_level="no",
        post_level="after", pre_level="before",
    )
    assert scenarios[0]["atexog"] == {"treat": "yes", "post": "after"}
    assert scenarios[3]["atexog"] == {"treat": "no", "post": "before"}


def test_did_with_fixed():
    scenarios, _ = did("treat", "post", female=0)
    assert scenarios[0]["atexog"] == {"treat": 1, "post": 1, "female": 0}


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_basic():
    assert diff(2) == [+1, -1]
    assert diff(4) == [+1, -1, 0, 0]
    assert diff(5) == [+1, -1, 0, 0, 0]


def test_diff_too_small():
    with pytest.raises(ValueError, match="n >= 2"):
        diff(1)


# ---------------------------------------------------------------------------
# all_pairwise
# ---------------------------------------------------------------------------

def test_all_pairwise_basic():
    scenarios, contrasts = all_pairwise(
        ["treatment", "sex"],
        [[1, 0], ["M", "F"]],
    )
    assert len(scenarios) == 4
    labels = [s["label"] for s in scenarios]
    assert labels[0] == "treatment=1, sex=M"
    assert labels[1] == "treatment=1, sex=F"
    assert labels[2] == "treatment=0, sex=M"
    assert labels[3] == "treatment=0, sex=F"

    assert len(contrasts) == 3
    # Each non-reference vs reference (first scenario)
    for w in contrasts.values():
        assert w[0] == -1.0
        assert sum(w) == 0.0


def test_all_pairwise_with_fixed():
    scenarios, _ = all_pairwise(
        ["treatment"],
        [[1, 0]],
        age=30,
    )
    assert scenarios[0]["atexog"] == {"treatment": 1, "age": 30}
    assert scenarios[1]["atexog"] == {"treatment": 0, "age": 30}


def test_all_pairwise_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        all_pairwise(["a", "b"], [[1, 0]])


def test_all_pairwise_empty():
    # itertools.product(*[]) returns [()], so we need at least one variable
    # with an empty values list to get an empty product
    with pytest.raises(ValueError, match="at least one combination"):
        all_pairwise(["x"], [[]])


def test_all_pairwise_single_variable_shorthand():
    """all_pairwise should accept a single string and flat list for one variable."""
    scenarios, contrasts = all_pairwise("region", ["N", "S", "E", "W"])
    assert len(scenarios) == 4
    assert scenarios[0]["atexog"] == {"region": "N"}
    assert "region=S_vs_region=N" in contrasts
    assert len(contrasts["region=S_vs_region=N"]) == 4
