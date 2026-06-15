"""I6: the new noun must not import the legacy Margins session."""

from __future__ import annotations

import ast
from pathlib import Path


def _source_imports_margins(path: Path) -> bool:
    text = path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Match the legacy ``pymargins.margins`` package, not ``pymargins`` itself.
            if module == "margins" or module.endswith(".margins") or module.startswith("pymargins.margins"):
                return True
            if any(alias.name == "Margins" for alias in node.names):
                return True
        if isinstance(node, ast.Import):
            if any(alias.name == "Margins" for alias in node.names):
                return True
    return False


def test_gcomputation_module_has_no_margins_import():
    base = Path(__file__).parent.parent / "pymargins" / "estimators" / "_base.py"
    assert not _source_imports_margins(base), (
        "pymargins/estimators/_base.py must not import the legacy Margins session"
    )
