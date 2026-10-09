#!/usr/bin/env python3
"""PreToolUse guardrail for the autonomous build.

Reads the hook JSON from stdin. Exit 2 with a message on stderr blocks the tool call; exit 0 = no objection.
Role = ``agent_type`` (``builder`` / ``qa-validator``); absent = ``orchestrator``.
Stdlib only. Rules mirror KICKOFF.md §4.4.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ORCH = "orchestrator"
BUILDER = "builder"
QA = "qa-validator"

WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
CMD_SEPARATORS = {";", "&&", "||", "|", "&", "|&", "(", ")", "\n"}
REDIRECTS = {">", ">>", "&>", "&>>", ">|", ">&"}
ENV_FILE_RE = re.compile(r"^\.env(\..+)?$")
ENV_READERS = {
    "cat", "tac", "grep", "egrep", "fgrep", "rg", "head", "tail", "less", "more", "source", ".", "bat", "nl",
    "sed", "awk", "strings", "xxd", "od", "hexdump", "cut", "sort", "uniq", "diff", "base64", "jq", "view", "vim",
    "vi", "nano", "cp", "mv", "scp", "rsync", "python", "python3",
}
SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
DANGEROUS_RM_TARGETS = {"/", "/*", "~", "~/", "~/*", "$HOME", "$HOME/", "$HOME/*", "${HOME}", "${HOME}/", "..", "../",
                        "../*"}
SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}|\blsv2_[A-Za-z0-9_]{10,}")
COV_FAIL_RE = re.compile(r"--cov-fail-under[=\s]+(\d+(?:\.\d+)?)")
NO_COVER_RE = re.compile(r"pragma:\s*no\s*cover")
SKIP_RE = re.compile(r"pytest\.mark\.(skip|xfail)|pytest\.skip\(|pytest\.xfail\(")
PYPROJECT_COV_RE = re.compile(r"^\s*(omit|fail_under)\s*=", re.MULTILINE)
PR_BRANCH_RE = re.compile(r"^pr/(\d+)-")
# Scratch/log locations every role may write (gitignored). Documented deviation, see CLAUDE.md.
ALWAYS_WRITABLE_PREFIXES = ("logs/",)
DEV_PATHS = ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")


class Block(Exception):
    """Raised with the human-readable reason a tool call is refused."""


# --------------------------------------------------------------------------------------------- helpers


def project_root(data: dict[str, Any]) -> Path:
    raw = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    return Path(raw).resolve()


def role_of(data: dict[str, Any]) -> str:
    return str(data.get("agent_type") or ORCH)


def rel_path(path: str, root: Path, cwd: Path) -> str | None:
    """Repo-relative POSIX path, or None when the path is outside the repo."""
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = cwd / p
    resolved = Path(os.path.normpath(str(p)))
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        try:
            return resolved.resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            return None


def is_env_file(name: str) -> bool:
    base = name.rstrip("/").rsplit("/", 1)[-1]
    return bool(ENV_FILE_RE.match(base)) and base != ".env.example"


def protected_reason(rel: str) -> str | None:
    if rel.startswith(".github/workflows/") or rel == ".github/workflows":
        return ".github/workflows/** is human-owned"
    if (rel == ".claude" or rel.startswith(".claude/")) and not (
        rel == ".claude/state" or rel.startswith(".claude/state/")
    ):
        return ".claude/** (except .claude/state/**) is protected"
    if rel in {"CLAUDE.md", "KICKOFF.md", "docs/TOKEN_USAGE.md", "docs/token_usage.jsonl"}:
        return f"{rel} is protected"
    if is_env_file(rel):
        return ".env files are never written by agents (only .env.example)"
    return None


def role_write_reason(rel: str, role: str) -> str | None:
    if rel.startswith(ALWAYS_WRITABLE_PREFIXES):
        return None
    if role == ORCH:
        allowed = {"docs/PR_TRACKER.md", "docs/BLOCKERS.md", "docs/BACKLOG.md"}
        if rel in allowed or rel.startswith(".claude/state/"):
            return None
        return f"orchestrator may only write PR_TRACKER/BLOCKERS/BACKLOG and .claude/state/** (not {rel}); delegate to builder"
    if role == QA:
        if rel.startswith("reports/qa/"):
            return None
        return f"qa-validator may only write reports/qa/** (not {rel})"
    if role == BUILDER and rel.startswith("reports/qa/"):
        return "builder may not write reports/qa/** (QA verdicts belong to qa-validator)"
    return None


def content_reason(rel: str, text: str) -> str | None:
    if not text:
        return None
    in_code = rel.startswith(("src/", "tests/"))
    if in_code and NO_COVER_RE.search(text):
        return "'pragma: no cover' is not allowed in src/ or tests/"
    if in_code and SKIP_RE.search(text) and "live" not in text:
        return "skip/xfail is only allowed for live tests (text must mention 'live')"
    if rel == "pyproject.toml" and PYPROJECT_COV_RE.search(text):
        return "pyproject.toml must not set coverage omit/fail_under (gate lives in --cov-fail-under=80)"
    if rel.startswith("src/") and SECRET_RE.search(text):
        return "secret-shaped string (sk-…/lsv2_…) in src/"
    m = COV_FAIL_RE.search(text)
    if m and float(m.group(1)) < 80:
        return f"--cov-fail-under {m.group(1)} is below 80"
    return None


def check_write(path: str, text: str, role: str, root: Path, cwd: Path) -> None:
    if path in DEV_PATHS or path.startswith("/dev/fd/"):
        return
    rel = rel_path(path, root, cwd)
    if rel is None:
        raise Block(f"writing outside the repo is not allowed: {path}")
    for reason in (protected_reason(rel), role_write_reason(rel, role), content_reason(rel, text)):
        if reason:
            raise Block(reason)


# --------------------------------------------------------------------------------------------- bash parsing


HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def strip_heredocs(cmd: str) -> tuple[str, str]:
    """Return (command without heredoc bodies, concatenated heredoc bodies)."""
    lines = cmd.split("\n")
    out: list[str] = []
    bodies: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        delims = [m.group(2) for m in HEREDOC_RE.finditer(line)]
        i += 1
        for d in delims:
            body: list[str] = []
            while i < len(lines) and lines[i].strip() != d:
                body.append(lines[i])
                i += 1
            i += 1  # skip the delimiter line
            bodies.append("\n".join(body))
    return "\n".join(out), "\n".join(bodies)


def tokenize(cmd: str) -> list[str]:
    cmd = cmd.replace("\n", " ; ")
    try:
        lex = shlex.shlex(cmd, posix=True, punctuation_chars=";&|()<>")
        lex.whitespace_split = True
        lex.commenters = ""
        return list(lex)
    except ValueError:
        return cmd.split()


def segments(tokens: list[str]) -> list[list[str]]:
    segs: list[list[str]] = [[]]
    for t in tokens:
        if t in CMD_SEPARATORS:
            segs.append([t])  # keep the separator as a marker at the start of the next segment
        else:
            segs[-1].append(t)
    return [s for s in segs if s]


def split_marker(seg: list[str]) -> tuple[str | None, list[str]]:
    if seg and seg[0] in CMD_SEPARATORS:
        return seg[0], seg[1:]
    return None, seg


def strip_prefix(words: list[str]) -> list[str]:
    """Drop leading VAR=val assignments and wrappers like sudo/nohup/time/env VAR=x."""
    i = 0
    while i < len(words):
        w = words[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w):
            i += 1
        elif w in {"sudo", "nohup", "time", "command", "exec", "nice"}:
            i += 1
        elif w == "env" and i + 1 < len(words) and (
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[i + 1]) or words[i + 1].startswith("-")
        ):
            i += 1
            while i < len(words) and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[i]) or words[i].startswith("-")):
                i += 1
        else:
            break
    return words[i:]


def words_and_redirects(seg: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a segment into plain words, output-redirect targets and input-redirect sources."""
    words: list[str] = []
    outs: list[str] = []
    ins: list[str] = []
    i = 0
    while i < len(seg):
        t = seg[i]
        if t in REDIRECTS or t == "<" or t == "<<" or t == "<<<" or t == "<&":
            target = seg[i + 1] if i + 1 < len(seg) else ""
            if t in REDIRECTS and target and not target.isdigit() and target != "-":
                outs.append(target)
            elif t == "<" and target:
                ins.append(target)
            i += 2
            continue
        if t.isdigit() and i + 1 < len(seg) and seg[i + 1] in REDIRECTS | {"<"}:
            i += 1  # file-descriptor number before a redirect
            continue
        words.append(t)
        i += 1
    return words, outs, ins


def git_subcommand(words: list[str]) -> tuple[str | None, list[str]]:
    i = 1
    while i < len(words):
        w = words[i]
        if w in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}:
            i += 2
        elif w.startswith("-"):
            i += 1
        else:
            return w, words[i + 1:]
    return None, []


def positionals(args: list[str], value_opts: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in value_opts:
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def current_branch(cwd: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# --------------------------------------------------------------------------------------------- bash rules


def check_git(args_words: list[str], role: str, root: Path, cwd: Path, prior: list[list[str]]) -> None:
    sub, rest = git_subcommand(args_words)
    if "--no-verify" in rest:
        raise Block("--no-verify is not allowed")
    if sub == "config":
        raise Block("git config is not allowed")
    if sub == "commit":
        if role == QA:
            raise Block("qa-validator may not commit")
        value_flags = set("mFCcSt")  # short options that consume the rest of the cluster / next arg
        skip_next = False
        for a in rest:
            if skip_next:
                skip_next = False
                continue
            if a in {"-m", "-F", "-C", "-c", "--message", "--file", "--author", "--date", "-t", "--template"}:
                skip_next = True
                continue
            if re.fullmatch(r"-[A-Za-z]+", a):
                for ch in a[1:]:
                    if ch == "n":
                        raise Block("git commit -n (skip hooks) is not allowed")
                    if ch in value_flags:
                        break
    if sub == "push":
        if role == QA:
            raise Block("qa-validator may not push")
        for a in rest:
            if a in {"-f", "--force", "--force-with-lease", "--force-if-includes", "--mirror"} or a.startswith(
                ("--force-with-lease=", "--force=")
            ) or (re.fullmatch(r"-[A-Za-z]+", a) and "f" in a[1:]):
                raise Block("force-push is not allowed")
        pos = positionals(rest, {"--repo", "-o", "--push-option", "--receive-pack", "--exec"})
        refspecs = pos[1:]
        for r in refspecs:
            if r.startswith("+"):
                raise Block("force-push (+refspec) is not allowed")
            dst = r.split(":")[-1].removeprefix("refs/heads/")
            if dst in {"main", "master"}:
                raise Block("pushing to main/master is not allowed; open a PR")
        if not refspecs and current_branch(cwd) in {"main", "master"}:
            raise Block("pushing from main/master is not allowed; open a PR")
    if sub == "merge":
        check_local_merge(rest, role, root, cwd, prior)


def check_local_merge(rest: list[str], role: str, root: Path, cwd: Path, prior: list[list[str]]) -> None:
    if any(a in {"--abort", "--continue", "--quit"} for a in rest):
        return
    pos = positionals(rest, {"-m", "-F", "-s", "--strategy", "-X", "--strategy-option", "--file", "--message"})
    target = pos[0] if pos else ""
    switching_to_main = any(
        len(s) >= 3 and s[0] == "git" and s[1] in {"checkout", "switch"} and s[-1] in {"main", "master"}
        for s in prior
    )
    into_main = switching_to_main or current_branch(cwd) in {"main", "master"}
    if not (PR_BRANCH_RE.match(target) or into_main):
        return  # e.g. builder merging main into its own branch
    if role != ORCH:
        raise Block("only the orchestrator merges into main")
    m = PR_BRANCH_RE.match(target)
    if not m:
        raise Block("local merges into main must name a pr/<NN>-<slug> branch")
    n = m.group(1)
    require_verdict_pass(n, root)
    ok = find_existing(root / ".claude" / "state", [f"ci-{n}.ok", f"ci-{int(n)}.ok", f"ci-{int(n):02d}.ok",
                                                    f"ci-{target.replace('/', '-')}.ok"])
    if ok is None:
        raise Block(f"no .claude/state/ci-{n}.ok — run `make ci` on {target} first")
    if time.time() - ok.stat().st_mtime > 3600:
        raise Block(f"{ok.name} is older than 1 hour — re-run `make ci`")
    try:
        sha = subprocess.run(["git", "rev-parse", target], cwd=root, capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = ""
    if not sha or ok.read_text().strip() != sha:
        raise Block(f"{ok.name} does not match the current commit of {target} — re-run `make ci`")


def find_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        p = base / name
        if p.is_file():
            return p
    return None


def require_verdict_pass(n: str, root: Path) -> None:
    candidates = [f"pr-{n}.md"]
    if n.isdigit():
        candidates += [f"pr-{int(n)}.md", f"pr-{int(n):02d}.md"]
    report = find_existing(root / "reports" / "qa", candidates)
    if report is None:
        raise Block(f"merge blocked: reports/qa/pr-{n}.md does not exist")
    lines = [ln.strip() for ln in report.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines or lines[-1] != "VERDICT: PASS":
        raise Block(f"merge blocked: {report.name} does not end with 'VERDICT: PASS'")


def check_gh(words: list[str], role: str, root: Path, cwd: Path) -> None:
    args = words[1:]
    if len(args) >= 2 and args[0] == "repo" and args[1] in {"delete", "edit"}:
        raise Block(f"gh repo {args[1]} is not allowed")
    if args and args[0] == "secret":
        raise Block("gh secret is not allowed")
    if len(args) >= 2 and args[0] == "pr":
        action = args[1]
        rest = args[2:]
        if action == "review" and ("--approve" in rest or "-a" in rest):
            raise Block("gh pr review --approve is not allowed")
        if role == QA and action in {"create", "edit", "close", "merge", "ready", "reopen"}:
            raise Block(f"qa-validator may not run gh pr {action}")
        if action == "merge":
            if role != ORCH:
                raise Block("only the orchestrator merges")
            if "--admin" in rest:
                raise Block("gh pr merge --admin is not allowed")
            pos = positionals(rest, {"-t", "--subject", "-b", "--body", "-F", "--body-file", "-R", "--repo",
                                     "--match-head-commit", "-A", "--author-email"})
            if not pos or not re.fullmatch(r"\d+", pos[0]):
                raise Block("gh pr merge needs an explicit PR number")
            n = pos[0]
            require_verdict_pass(n, root)
            try:
                r = subprocess.run(["gh", "pr", "checks", n], cwd=root, capture_output=True, text=True, timeout=60)
                code = r.returncode
            except (OSError, subprocess.SubprocessError):
                code = 1
            if code != 0:
                raise Block(f"merge blocked: `gh pr checks {n}` is not green")


def inplace_files(args: list[str]) -> list[str]:
    """File operands of `sed -i` / `perl -pi`: positionals minus the script (unless given via -e)."""
    files: list[str] = []
    has_e = False
    i = 0
    while i < len(args):
        a = args[i]
        if a in {"-e", "--expression", "-f", "--file"}:
            has_e = True
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        files.append(a)
        i += 1
    return files if has_e else files[1:]


def check_segment(seg: list[str], marker: str | None, prev_words: list[str], role: str, root: Path, cwd: Path,
                  prior: list[list[str]], heredoc_text: str, full_cmd: str) -> None:
    raw_words, outs, ins = words_and_redirects(seg)
    words = strip_prefix(raw_words)
    cmd = words[0].rsplit("/", 1)[-1] if words else ""

    # pipe into a shell after curl/wget
    if marker in {"|", "|&"} and cmd in SHELLS and prev_words and prev_words[0].rsplit("/", 1)[-1] in {"curl", "wget"}:
        raise Block("piping curl/wget into a shell is not allowed")

    # .env reads
    for src in ins:
        if is_env_file(src):
            raise Block("reading .env files is not allowed")
    if cmd in ENV_READERS and any(is_env_file(w) for w in words[1:] if not w.startswith("-")):
        raise Block("reading/printing .env files is not allowed")

    # environment dumps
    if cmd == "printenv":
        raise Block("printenv is not allowed (may leak secrets)")
    if cmd == "env" and all(w.startswith("-") for w in words[1:]):
        raise Block("dumping the environment with env is not allowed")
    if cmd == "export" and (len(words) == 1 or words[1:] == ["-p"]):
        raise Block("export -p dumps the environment; not allowed")
    if cmd in {"set", "declare"} and len(words) == 1:
        raise Block(f"bare `{cmd}` dumps the environment; not allowed")

    # rm -rf of dangerous targets
    if cmd == "rm":
        flags = "".join(w[1:] for w in words[1:] if re.fullmatch(r"-[A-Za-z]+", w))
        recursive = "r" in flags or "R" in flags or "--recursive" in words
        if recursive and any(w in DANGEROUS_RM_TARGETS or w.startswith("../") for w in words[1:]):
            raise Block("rm -r of /, ~, $HOME or .. is not allowed")

    # pip install (uv pip is fine)
    if cmd in {"pip", "pip3"} and len(words) > 1 and words[1] == "install":
        raise Block("pip install is not allowed; use uv (uv add / uv sync / uv pip)")
    if cmd.startswith("python") and words[1:4] == ["-m", "pip", "install"]:
        raise Block("python -m pip install is not allowed; use uv")

    if cmd == "git":
        check_git(words, role, root, cwd, prior)
    if cmd == "gh":
        check_gh(words, role, root, cwd)

    # in-place edits
    inplace_targets: list[str] = []
    if cmd == "sed" and any(w.startswith("-i") or w.startswith("--in-place") for w in words[1:]):
        inplace_targets = inplace_files(words[1:])
    if cmd == "perl" and any(re.fullmatch(r"-[A-Za-z]*", w) and "p" in w and "i" in w for w in words[1:]):
        inplace_targets = inplace_files(words[1:])
    for t in inplace_targets:
        rel = rel_path(t, root, cwd)
        if role == ORCH and rel is not None and rel.startswith(("src/", "tests/")):
            raise Block("orchestrator may not edit src/ or tests/ (sed -i / perl -pi)")

    # writes via redirect / tee / in-place edit
    targets = list(outs) + inplace_targets
    if cmd == "tee":
        targets += [w for w in words[1:] if not w.startswith("-")]
    for t in targets:
        text = heredoc_text or full_cmd
        check_write(t, text, role, root, cwd)


def check_bash(command: str, role: str, root: Path, cwd: Path) -> None:
    if re.search(r"\b(sh|bash|zsh)\s+-c\s+[\"']?\$\(\s*(curl|wget)\b", command) or re.search(
        r"\b(sh|bash|zsh)\s+<\(\s*(curl|wget)\b", command
    ):
        raise Block("executing a downloaded script is not allowed")
    m = COV_FAIL_RE.search(command)
    if m and float(m.group(1)) < 80:
        raise Block(f"--cov-fail-under {m.group(1)} is below 80")

    body_cmd, heredoc_text = strip_heredocs(command)
    segs = segments(tokenize(body_cmd))
    prior: list[list[str]] = []
    prev_words: list[str] = []
    for seg in segs:
        marker, rest = split_marker(seg)
        check_segment(rest, marker, prev_words, role, root, cwd, prior, heredoc_text, command)
        words = strip_prefix(words_and_redirects(rest)[0])
        prior.append(words)
        prev_words = words


# --------------------------------------------------------------------------------------------- entry point


def evaluate(data: dict[str, Any]) -> str | None:
    """Return a block reason, or None if the call is allowed."""
    tool = str(data.get("tool_name") or "")
    inp = data.get("tool_input") or {}
    role = role_of(data)
    root = project_root(data)
    cwd = Path(str(data.get("cwd") or root)).resolve()
    try:
        if tool == "Read":
            path = str(inp.get("file_path") or "")
            if is_env_file(path):
                raise Block("reading .env files is not allowed (only .env.example)")
        elif tool in WRITE_TOOLS:
            path = str(inp.get("file_path") or inp.get("notebook_path") or "")
            if tool == "Write":
                text = str(inp.get("content") or "")
            elif tool == "Edit":
                text = str(inp.get("new_string") or "")
            elif tool == "MultiEdit":
                text = "\n".join(str(e.get("new_string") or "") for e in inp.get("edits") or [])
            else:
                text = str(inp.get("new_source") or "")
            check_write(path, text, role, root, cwd)
        elif tool == "Bash":
            check_bash(str(inp.get("command") or ""), role, root, cwd)
    except Block as b:
        return str(b)
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0
    reason = evaluate(data)
    if reason:
        print(f"Blocked by guardrails ({role_of(data)}): {reason}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
