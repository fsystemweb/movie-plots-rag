#!/usr/bin/env python3
"""SubagentStop gate for builder and qa-validator (KICKOFF.md §4.4).

builder      -> `make lint typecheck test` must be green (9 min timeout); otherwise exit 2 with the output tail.
qa-validator -> a reports/qa/pr-*.md modified in the last 3 hours must end with a `VERDICT:` line.
After 3 blocks for the same agent_id the agent is let through so it can report failure.
Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_BLOCKS = 3
BUILD_TIMEOUT_S = 540
QA_WINDOW_S = 3 * 3600
TAIL_CHARS = 3000
VERDICT_RE = re.compile(r"^VERDICT: (PASS|FAIL)$")


def project_root(data: dict[str, Any]) -> Path:
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()).resolve()


def counter_path(root: Path, agent_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", agent_id or "unknown")
    return root / ".claude" / "state" / f"stop_blocks-{safe}.count"


def read_count(path: Path) -> int:
    try:
        return int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def bump(path: Path) -> int:
    n = read_count(path) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(n))
    return n


def builder_gate(root: Path) -> str | None:
    try:
        r = subprocess.run(["make", "lint", "typecheck", "test"], cwd=root, capture_output=True, text=True,
                           timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return f"`make lint typecheck test` timed out after {BUILD_TIMEOUT_S}s\n{out[-TAIL_CHARS:]}"
    except OSError as e:
        return f"could not run make: {e}"
    if r.returncode != 0:
        out = (r.stdout or "") + (r.stderr or "")
        return f"`make lint typecheck test` failed (exit {r.returncode}). Fix and finish again.\n{out[-TAIL_CHARS:]}"
    return None


def qa_gate(root: Path, now: float | None = None) -> str | None:
    now = time.time() if now is None else now
    for report in sorted((root / "reports" / "qa").glob("pr-*.md")):
        try:
            if now - report.stat().st_mtime > QA_WINDOW_S:
                continue
            lines = [ln.strip() for ln in report.read_text(encoding="utf-8", errors="replace").splitlines()
                     if ln.strip()]
        except OSError:
            continue
        if lines and VERDICT_RE.match(lines[-1]):
            return None
    return ("no reports/qa/pr-<N>.md written in the last 3 hours ending with `VERDICT: PASS` or `VERDICT: FAIL`. "
            "Write the report before stopping.")


def evaluate(data: dict[str, Any]) -> tuple[int, str]:
    root = project_root(data)
    agent = str(data.get("agent_type") or "")
    agent_id = str(data.get("agent_id") or data.get("session_id") or "unknown")
    if agent == "builder":
        reason = builder_gate(root)
    elif agent == "qa-validator":
        reason = qa_gate(root)
    else:
        return 0, ""
    if reason is None:
        return 0, ""
    cpath = counter_path(root, agent_id)
    if read_count(cpath) >= MAX_BLOCKS:
        return 0, f"stop gate still red after {MAX_BLOCKS} blocks; letting {agent} stop to report failure"
    bump(cpath)
    return 2, f"Stop gate ({agent}): {reason}"


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0
    code, msg = evaluate(data)
    if msg:
        print(msg, file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
