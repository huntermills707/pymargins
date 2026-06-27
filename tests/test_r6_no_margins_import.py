"""I6: the new engine must not import the legacy Margins session or result."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Paths named by I6 (design §3.9 / implementation guide G1.6).
_I6_PATHS = [
    Path(__file__).parent.parent / "pymargins" / "_engine",
    Path(__file__).parent.parent / "pymargins" / "estimators",
    Path(__file__).parent.parent / "pymargins" / "_result" / "_graphresult.py",
    Path(__file__).parent.parent / "pymargins" / "_result" / "_intervals.py",
]


def _i6_source_files() -> list[Path]:
    files: list[Path] = []
    for p in _I6_PATHS:
        if p.is_dir():
            files.extend(p.rglob("*.py"))
        else:
            files.append(p)
    return files


def _source_imports_legacy(path: Path) -> list[str]:
    """Return human-readable descriptions of any I6-violating imports in *path*."""
    text = path.read_text()
    tree = ast.parse(text)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # pymargins.margins package (and relative imports targeting it)
            if (
                module == "margins"
                or module.endswith(".margins")
                or module.startswith("pymargins.margins")
            ):
                violations.append(
                    f"{path}: imports from legacy margins package: {module}"
                )
            # pymargins._result._margins module
            if (
                module == "_margins"
                or module.endswith("._margins")
                or module == "pymargins._result._margins"
            ):
                violations.append(
                    f"{path}: imports from legacy _margins module: {module}"
                )
            # MarginsResult symbol
            if any(alias.name == "MarginsResult" for alias in node.names):
                violations.append(f"{path}: imports MarginsResult from {module}")
            # run_inference from the legacy dispatch module
            if module == "pymargins._inference._dispatch" and any(
                alias.name == "run_inference" for alias in node.names
            ):
                violations.append(f"{path}: imports run_inference from {module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "MarginsResult":
                    violations.append(f"{path}: imports MarginsResult")
                if alias.name in (
                    "pymargins.margins",
                    "pymargins._result._margins",
                    "pymargins._inference._dispatch",
                ):
                    violations.append(f"{path}: imports legacy module {alias.name}")
    return violations


@pytest.mark.parametrize(
    "path", _i6_source_files(), ids=lambda p: str(p.relative_to(p.parent.parent.parent))
)
def test_i6_no_legacy_imports(path: Path):
    violations = _source_imports_legacy(path)
    assert not violations, "\n".join(violations)
