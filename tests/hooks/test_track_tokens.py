"""Tests for .claude/hooks/track_tokens.py — transcript discovery, dedup by message.id, rendering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _hookutil import HOOKS, git, load_hook

tt = load_hook("track_tokens")


def msg_line(mid: str, model: str, inp: int, out: int, cw: int = 0, cr: int = 0) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {"id": mid, "model": model, "usage": {
            "input_tokens": inp, "output_tokens": out,
            "cache_creation_input_tokens": cw, "cache_read_input_tokens": cr}},
    })


def fake_transcript(path: Path) -> Path:
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
        # one API message streamed over three lines (text, tool_use, tool_use) -> counted once
        msg_line("msg_1", "claude-opus-5-5", 10, 5, 100, 1000),
        msg_line("msg_1", "claude-opus-5-5", 10, 5, 100, 1000),
        msg_line("msg_1", "claude-opus-5-5", 10, 7, 100, 1000),
        "not json at all",
        msg_line("msg_2", "claude-sonnet-5-5", 20, 30, 0, 500),
        json.dumps({"type": "summary"}),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_dedup_by_message_id(tmp_path: Path) -> None:
    usage = tt.sum_usage(fake_transcript(tmp_path / "t.jsonl"))
    assert usage["messages"] == 2
    assert usage["totals"] == {
        "input_tokens": 30, "output_tokens": 37,
        "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1500,
    }
    assert usage["by_model"]["claude-opus-5-5"]["output_tokens"] == 7
    assert usage["by_model"]["claude-sonnet-5-5"]["input_tokens"] == 20


def test_find_transcript_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main = tmp_path / "proj" / "sess.jsonl"
    main.parent.mkdir(parents=True)
    main.write_text("")
    assert tt.find_transcript({"transcript_path": str(main)}) == main

    explicit = tmp_path / "explicit.jsonl"
    explicit.write_text("")
    assert tt.find_transcript({"agent_id": "a1", "agent_transcript_path": str(explicit),
                               "transcript_path": str(main)}) == explicit

    sub = main.parent / "sess" / "subagents" / "agent-a2.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text("")
    assert tt.find_transcript({"agent_id": "a2", "session_id": "sess", "transcript_path": str(main)}) == sub

    home = tmp_path / "home"
    globbed = home / ".claude" / "projects" / "x" / "deep" / "agent-a3.jsonl"
    globbed.parent.mkdir(parents=True)
    globbed.write_text("")
    monkeypatch.setenv("HOME", str(home))
    assert tt.find_transcript({"agent_id": "a3", "session_id": "nope", "transcript_path": str(main)}) == globbed
    assert tt.find_transcript({"agent_id": "missing"}) is None


def test_current_pr_from_state_then_branch(repo: Path) -> None:
    assert tt.current_pr(repo) == "none"
    git(repo, "checkout", "-q", "-b", "pr/04-retrieval")
    assert tt.current_pr(repo) == "PR-04"
    (repo / ".claude" / "state" / "current_pr").write_text("PR-05\n")
    assert tt.current_pr(repo) == "PR-05"


def run(args: list[str], repo: Path, stdin: str = "") -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo)}
    return subprocess.run([sys.executable, str(HOOKS / "track_tokens.py"), *args], input=stdin, text=True,
                          capture_output=True, env=env, cwd=repo)


def test_hook_appends_rows_and_render_keeps_last(repo: Path, tmp_path: Path) -> None:
    (repo / ".claude" / "state" / "current_pr").write_text("PR-01")
    t = fake_transcript(tmp_path / "agent.jsonl")
    ev: dict[str, Any] = {"hook_event_name": "SubagentStop", "session_id": "s1", "agent_id": "ag1",
                          "agent_type": "builder", "agent_transcript_path": str(t)}
    assert run([], repo, json.dumps(ev)).returncode == 0
    assert run([], repo, json.dumps(ev)).returncode == 0  # same agent twice -> render keeps one
    rows = [json.loads(x) for x in (repo / "docs" / "token_usage.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["pr"] == "PR-01" and rows[0]["agent"] == "builder" and rows[0]["agent_key"] == "ag1"
    assert rows[0]["output_tokens"] == 37

    main_t = fake_transcript(tmp_path / "main.jsonl")
    stop = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": str(main_t)}
    assert run([], repo, json.dumps(stop)).returncode == 0

    assert run(["--render"], repo).returncode == 0
    md = (repo / "docs" / "TOKEN_USAGE.md").read_text()
    assert "| PR-01 | builder | 1 | 30 | 37 | 100 | 1,500 |" in md
    assert "| PR-01 | orchestrator | 1 |" in md
    assert "| **Total** | | 2 | 60 | 74 | 200 | 3,000 |" in md
    assert "estimate" in md


def test_cost_uses_model_prices() -> None:
    prices = {"models": {"claude-opus-5-5": {"input": 4, "output": 20, "cache_write": 5, "cache_read": 0.2},
                         "claude-opus-5": {"input": 5, "output": 25, "cache_write": 6.25, "cache_read": 0.5}},
              "default": {"input": 1, "output": 1, "cache_write": 1, "cache_read": 1}}
    counts = {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}
    assert tt.cost_of(counts, "claude-opus-5-5", prices) == pytest.approx(24.2)  # longest prefix wins
    assert tt.cost_of(counts, "claude-opus-5", prices) == pytest.approx(30.5)
    assert tt.cost_of(counts, "mystery", prices) == pytest.approx(3.0)


def test_render_empty(repo: Path) -> None:
    assert run(["--render"], repo).returncode == 0
    assert "No runs recorded yet" in (repo / "docs" / "TOKEN_USAGE.md").read_text()


def test_never_fails(repo: Path) -> None:
    assert run([], repo, "garbage").returncode == 0
    assert run([], repo, json.dumps({"agent_id": "zzz", "session_id": "s"})).returncode == 0
    assert not (repo / "docs" / "token_usage.jsonl").exists()


def test_prices_file_is_valid() -> None:
    prices = json.loads((HOOKS / "prices.json").read_text())
    for p in [*prices["models"].values(), prices["default"]]:
        assert set(p) == {"input", "output", "cache_write", "cache_read"}
        assert all(v >= 0 for v in p.values())
