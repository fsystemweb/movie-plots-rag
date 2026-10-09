#!/usr/bin/env python3
"""Token accounting for the autonomous build (KICKOFF.md §4.4). Never fails the run.

Hook mode (Stop / SubagentStop): sum message.usage from the transcript, deduplicated by message.id, and append one
JSON row to docs/token_usage.jsonl.
`--render`: rebuild docs/TOKEN_USAGE.md from the last row per (session, agent_key).
Stdlib only.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
HOOK_DIR = Path(__file__).resolve().parent


def project_root(data: dict[str, Any] | None = None) -> Path:
    data = data or {}
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()).resolve()


# --------------------------------------------------------------------------------------------- transcript


def find_transcript(data: dict[str, Any]) -> Path | None:
    agent_id = data.get("agent_id")
    if not agent_id:
        tp = data.get("transcript_path")
        return Path(tp) if tp else None
    atp = data.get("agent_transcript_path")
    if atp and Path(atp).is_file():
        return Path(atp)
    tp = data.get("transcript_path")
    sid = data.get("session_id")
    if tp and sid:
        cand = Path(tp).parent / str(sid) / "subagents" / f"agent-{agent_id}.jsonl"
        if cand.is_file():
            return cand
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/**/agent-{agent_id}.jsonl"), recursive=True)
    return Path(hits[0]) if hits else None


def sum_usage(path: Path) -> dict[str, Any]:
    """Sum usage over unique message ids (the last line seen for an id wins). Also breaks totals down by model."""
    per_id: dict[str, tuple[str, dict[str, int]]] = {}
    anon = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") if isinstance(rec, dict) else None
            if not isinstance(msg, dict):
                continue
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            mid = msg.get("id")
            if not mid:
                anon += 1
                mid = f"__anon_{anon}"
            counts = {f: int(usage.get(f) or 0) for f in FIELDS}
            per_id[str(mid)] = (str(msg.get("model") or "unknown"), counts)
    totals = {f: 0 for f in FIELDS}
    by_model: dict[str, dict[str, int]] = defaultdict(lambda: {f: 0 for f in FIELDS})
    for model, counts in per_id.values():
        for f in FIELDS:
            totals[f] += counts[f]
            by_model[model][f] += counts[f]
    return {"totals": totals, "by_model": dict(by_model), "messages": len(per_id)}


def current_pr(root: Path) -> str:
    state = root / ".claude" / "state" / "current_pr"
    try:
        val = state.read_text().strip()
        if val:
            return val
    except OSError:
        pass
    try:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        branch = ""
    m = re.match(r"^pr/(\d+)-", branch)
    return f"PR-{m.group(1)}" if m else "none"


def record(data: dict[str, Any]) -> dict[str, Any] | None:
    root = project_root(data)
    transcript = find_transcript(data)
    if transcript is None or not transcript.is_file():
        return None
    usage = sum_usage(transcript)
    agent_id = data.get("agent_id")
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pr": current_pr(root),
        "session": data.get("session_id") or "unknown",
        "agent": data.get("agent_type") or ("subagent" if agent_id else "orchestrator"),
        "agent_key": agent_id or "main",
        **usage["totals"],
        "by_model": usage["by_model"],
    }
    out = root / "docs" / "token_usage.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


# --------------------------------------------------------------------------------------------- render


def load_prices(path: Path | None = None) -> dict[str, Any]:
    try:
        return json.loads((path or HOOK_DIR / "prices.json").read_text())
    except (OSError, ValueError):
        return {"models": {}, "default": {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}}


def price_for(model: str, prices: dict[str, Any]) -> dict[str, float]:
    models = prices.get("models", {})
    best = ""
    for key in models:
        if model.startswith(key) and len(key) > len(best):
            best = key
    return models[best] if best else prices.get("default", {})


def cost_of(counts: dict[str, int], model: str, prices: dict[str, Any]) -> float:
    p = price_for(model, prices)
    return (counts.get("input_tokens", 0) * p.get("input", 0)
            + counts.get("output_tokens", 0) * p.get("output", 0)
            + counts.get("cache_creation_input_tokens", 0) * p.get("cache_write", 0)
            + counts.get("cache_read_input_tokens", 0) * p.get("cache_read", 0)) / 1_000_000


def row_cost(row: dict[str, Any], prices: dict[str, Any]) -> float:
    by_model = row.get("by_model") or {}
    if by_model:
        return sum(cost_of(c, m, prices) for m, c in by_model.items())
    return cost_of(row, "unknown", prices)


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        last[(str(r.get("session")), str(r.get("agent_key")))] = r
    return list(last.values())


def render(root: Path, prices: dict[str, Any] | None = None) -> str:
    prices = prices or load_prices()
    src = root / "docs" / "token_usage.jsonl"
    rows: list[dict[str, Any]] = []
    if src.is_file():
        for line in src.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    header = ["# Token usage", "",
              "Generated by `.claude/hooks/track_tokens.py --render`. Do not edit by hand.", ""]
    if not rows:
        text = "\n".join(header + ["No runs recorded yet.", ""])
    else:
        groups: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in latest_rows(rows):
            g = groups[(str(r.get("pr")), str(r.get("agent")))]
            g["runs"] += 1
            for f in FIELDS:
                g[f] += int(r.get(f) or 0)
            g["cost"] += row_cost(r, prices)
        lines = header + [
            "| PR | Agent | Runs | Input | Output | Cache write | Cache read | Est. cost USD (estimate) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        tot: dict[str, float] = defaultdict(float)
        for (pr, agent), g in sorted(groups.items()):
            lines.append(f"| {pr} | {agent} | {int(g['runs'])} | {int(g['input_tokens']):,} | "
                         f"{int(g['output_tokens']):,} | {int(g['cache_creation_input_tokens']):,} | "
                         f"{int(g['cache_read_input_tokens']):,} | ${g['cost']:.2f} |")
            for k, v in g.items():
                tot[k] += v
        lines.append(f"| **Total** | | {int(tot['runs'])} | {int(tot['input_tokens']):,} | "
                     f"{int(tot['output_tokens']):,} | {int(tot['cache_creation_input_tokens']):,} | "
                     f"{int(tot['cache_read_input_tokens']):,} | **${tot['cost']:.2f}** |")
        lines += ["", "Costs are estimates from `.claude/hooks/prices.json` (public list prices). The orchestrator's "
                      "main-session row is cumulative and is attributed to the PR active at its last Stop.", ""]
        text = "\n".join(lines)
    out = root / "docs" / "TOKEN_USAGE.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str]) -> int:
    try:
        if "--render" in argv:
            render(project_root())
            return 0
        data = json.load(sys.stdin)
        record(data)
    except Exception as e:  # never fail the run
        print(f"track_tokens: ignored error: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
