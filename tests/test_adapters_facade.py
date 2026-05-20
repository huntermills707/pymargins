"""Tests for the public pymargins.adapters facade (lazy re-exports)."""

import glob
import importlib
import os

import pytest

import pymargins.adapters as facade
from pymargins.adapters import _ADAPTER_MODULES


def _concrete_adapter_classes():
    """(class_name, submodule) for every *Adapter class under _adapters/."""
    here = os.path.dirname(importlib.import_module("pymargins._adapters").__file__)
    pairs = []
    for path in sorted(glob.glob(os.path.join(here, "*.py"))):
        sub = os.path.splitext(os.path.basename(path))[0]
        if sub.startswith("_"):
            continue
        with open(path) as fh:
            for line in fh:
                if line.startswith("class ") and "Adapter" in line:
                    name = line[len("class "):].split("(")[0].split(":")[0].strip()
                    if name.endswith("Adapter"):
                        pairs.append((name, sub))
    return pairs


@pytest.mark.parametrize("name,sub", _concrete_adapter_classes())
def test_every_concrete_adapter_is_exported(name, sub):
    """The facade maps every concrete adapter to its real submodule."""
    assert _ADAPTER_MODULES.get(name) == sub, name
    assert name in facade.__all__


def test_mapping_has_no_stale_entries():
    """No mapping entry points at a class that no longer exists."""
    real = dict(_concrete_adapter_classes())
    assert set(_ADAPTER_MODULES) == set(real)


@pytest.mark.parametrize("name", sorted(_ADAPTER_MODULES))
def test_lazy_attribute_resolves_to_named_class(name):
    obj = getattr(facade, name)
    assert isinstance(obj, type)
    assert obj.__name__ == name
    # Cached on the module after first access (skips __getattr__ next time).
    assert name in vars(facade)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        facade.NotAnAdapter


def test_importing_pymargins_does_not_pull_optional_deps():
    """`import pymargins` must not eagerly import statsmodels/lifelines/etc."""
    import subprocess
    import sys

    code = (
        "import sys, pymargins, pymargins.adapters; "
        "bad=[m for m in ('statsmodels','lifelines','linearmodels','sklearn') "
        "if m in sys.modules]; "
        "print(bad)"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "[]", out
