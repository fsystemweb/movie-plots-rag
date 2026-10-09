"""Helpers for the hook tests: load a hook module from .claude/hooks, run git, build events."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HOOKS = Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def load_hook(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"hook_{name}", HOOKS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def event(tool: str, role: str | None = None, cwd: Path | None = None, **tool_input: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"tool_name": tool, "tool_input": tool_input, "hook_event_name": "PreToolUse"}
    if role:
        data["agent_type"] = role
    if cwd:
        data["cwd"] = str(cwd)
    return data
