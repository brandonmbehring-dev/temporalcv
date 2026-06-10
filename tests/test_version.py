"""Version-consistency gate (A5).

The release version lives in four hand-maintained copies: ``pyproject.toml``,
``temporalcv.__version__``, ``docs/conf.py`` (Sphinx ``release``), and
``CITATION.cff``. Nothing tied them together before the 2.0.0 release prep
(review finding) — a missed site is silent drift visible only to readers of
the published docs. These tests parse the FILES (not ``importlib.metadata``,
which lags under a stale editable install) so a future bump that misses a
site fails loud.
"""

from __future__ import annotations

import re
from pathlib import Path

import temporalcv

_ROOT = Path(__file__).resolve().parents[1]


def _extract(path: str, pattern: str) -> str:
    text = (_ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text, re.MULTILINE)
    assert match is not None, f"version pattern {pattern!r} not found in {path}"
    return match.group(1)


def test_dunder_version_matches_pyproject() -> None:
    assert temporalcv.__version__ == _extract("pyproject.toml", r'^version = "(.+)"$')


def test_sphinx_release_matches_package_version() -> None:
    assert temporalcv.__version__ == _extract("docs/conf.py", r'^release = "(.+)"$')


def test_citation_cff_version_matches() -> None:
    assert temporalcv.__version__ == _extract("CITATION.cff", r'^version: "(.+)"$')


def test_specification_header_matches() -> None:
    # SPECIFICATION.md is a release-facing top-level doc; its version header
    # sat at 1.0.0-rc1 through two releases before this gate existed.
    assert temporalcv.__version__ == _extract("SPECIFICATION.md", r"\*\*Version\*\*: (\S+)")
