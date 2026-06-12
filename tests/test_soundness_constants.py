"""Tests that every soundness constant carries a citation docstring (W2.3)."""

from __future__ import annotations

import ast
import inspect

import pymargins._soundness._constants as _c


def _get_module_source(module):
    try:
        return inspect.getsource(module)
    except OSError:
        return None


def test_all_constants_have_citation_docstrings():
    source = _get_module_source(_c)
    assert source is not None
    tree = ast.parse(source)

    # Map: name -> docstring (string literal following the assignment)
    docs = {}
    body = tree.body
    for i, node in enumerate(body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Check if next node is an Expr with a Constant str
                    if i + 1 < len(body) and isinstance(body[i + 1], ast.Expr):
                        expr = body[i + 1]
                        if isinstance(expr.value, ast.Constant) and isinstance(
                            expr.value.value, str
                        ):
                            docs[target.id] = expr.value.value

    missing = []
    for name in dir(_c):
        if name.startswith("_"):
            continue
        # Only check ALL_CAPS constants and public functions
        if not (name.isupper() or name in ("m_out_of_n", "block_length_fallback")):
            continue
        obj = getattr(_c, name)
        if callable(obj):
            doc = inspect.getdoc(obj)
            if not doc or "Basis:" not in doc:
                missing.append(name)
        else:
            if name not in docs or "Basis:" not in docs.get(name, ""):
                missing.append(name)

    assert not missing, f"Constants missing citation docstrings: {missing}"
