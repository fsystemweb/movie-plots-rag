"""Fixtures for the guardrail hook tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from _hookutil import git


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo on `main` with one commit, used as CLAUDE_PROJECT_DIR."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "init")
    (r / ".claude" / "state").mkdir(parents=True)
    (r / "reports" / "qa").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(r))
    return r
