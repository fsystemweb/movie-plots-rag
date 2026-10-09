"""Tests for .claude/hooks/stop_gates.py."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from _hookutil import load_hook

sg = load_hook("stop_gates")


def ev(agent: str, agent_id: str = "a1") -> dict[str, Any]:
    return {"hook_event_name": "SubagentStop", "agent_type": agent, "agent_id": agent_id}


def test_qa_gate_requires_recent_verdict(repo: Path) -> None:
    code, msg = sg.evaluate(ev("qa-validator"))
    assert code == 2 and "VERDICT" in msg
    report = repo / "reports" / "qa" / "pr-3.md"
    report.write_text("# QA\nno verdict yet\n")
    assert sg.evaluate(ev("qa-validator", "a2"))[0] == 2
    report.write_text("# QA\n\nVERDICT: FAIL\n\n")
    assert sg.evaluate(ev("qa-validator", "a3")) == (0, "")
    old = time.time() - 4 * 3600
    os.utime(report, (old, old))
    assert sg.evaluate(ev("qa-validator", "a4"))[0] == 2


def test_builder_gate_runs_make(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    rc = {"v": 1}

    def fake_run(args: list[str], *a: Any, **kw: Any) -> Any:
        seen.append(args)
        return subprocess.CompletedProcess(args, rc["v"], "x" * 5000 + "TAIL", "")

    monkeypatch.setattr(sg.subprocess, "run", fake_run)
    code, msg = sg.evaluate(ev("builder"))
    assert code == 2 and msg.endswith("TAIL") and len(msg) < 3300
    assert seen[0] == ["make", "lint", "typecheck", "test"]
    rc["v"] = 0
    assert sg.evaluate(ev("builder", "b2")) == (0, "")


def test_builder_gate_timeout(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(args: list[str], *a: Any, **kw: Any) -> Any:
        raise subprocess.TimeoutExpired(args, 540, output="partial")

    monkeypatch.setattr(sg.subprocess, "run", boom)
    code, msg = sg.evaluate(ev("builder"))
    assert code == 2 and "timed out" in msg


def test_lets_agent_stop_after_three_blocks(repo: Path) -> None:
    codes = [sg.evaluate(ev("qa-validator", "same"))[0] for _ in range(5)]
    assert codes == [2, 2, 2, 0, 0]
    assert sg.evaluate(ev("qa-validator", "other"))[0] == 2  # counter is per agent_id


def test_other_agents_pass(repo: Path) -> None:
    assert sg.evaluate(ev("Explore")) == (0, "")
