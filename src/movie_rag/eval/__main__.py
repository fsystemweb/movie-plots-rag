"""``python -m movie_rag.eval`` (``make eval``, ``make eval-smoke``, ``make report``).

* ``run [--mode M|all] [--no-llm]``: evaluate and write ``<reports_dir>/eval_<mode>.json`` for each mode. The LLM half
  (agent + RAGAS) runs when ``NEBIUS_API_KEY`` is set and is recorded as "pending credentials" otherwise.
* ``smoke [--llm]``: the retrieval half on the fixture for all three modes, printed, nothing written; fails when a
  mode's MRR drops below ``eval.smoke_min_mrr``. ``--llm`` adds a small agent + RAGAS sample (skipped without a key).
* ``report``: render ``<reports_dir>/EVAL_RESULTS.md`` and the results block of the README (``eval.readme_path``)
  from the JSON files.

Needs the Qdrant service (``make up``, or ``QDRANT_URL``); the fixture is ingested when it is not in the index yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import MovieRagError
from movie_rag.eval.report import EvalReport, read_reports, render_markdown, render_readme_block, update_readme
from movie_rag.eval.runner import MODES, Dependencies, RunResult, run
from movie_rag.ingest.download import say

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
ALL = "all"
RESULTS_NAME = "EVAL_RESULTS.md"


def emit(text: str) -> None:
    """User-facing CLI output on stdout."""
    sys.stdout.write(text.rstrip("\n") + "\n")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_summary(reports: Sequence[EvalReport]) -> str:
    """A fixed-width table: one row per mode with Hit@k, MRR and latency, then the state of the LLM half."""
    if not reports:
        return "(no reports)"
    ks = reports[0].retrieval.k_values
    header = ["mode", "n", *(f"Hit@{k}" for k in ks), "MRR", "p50 ms", "p95 ms", "LLM half"]
    rows = [header]
    for r in reports:
        o, lat = r.retrieval.overall, r.retrieval.latency_ms
        rows.append(
            [
                r.mode,
                str(o.n),
                *(_fmt(o.hit_at_k.get(k)) for k in ks),
                _fmt(o.mrr),
                "n/a" if lat.p50 is None else f"{lat.p50:.0f}",
                "n/a" if lat.p95 is None else f"{lat.p95:.0f}",
                r.llm.status.replace("_", " "),
            ]
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in rows)


def smoke_failures(settings: Settings, reports: Sequence[EvalReport]) -> list[str]:
    """Modes whose MRR is below ``eval.smoke_min_mrr`` (an undefined MRR counts as below)."""
    floor = settings.eval.smoke_min_mrr
    return [
        f"{r.mode}: MRR {_fmt(r.retrieval.overall.mrr)} is below eval.smoke_min_mrr {floor}"
        for r in reports
        if r.retrieval.overall.mrr is None or r.retrieval.overall.mrr < floor
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m movie_rag.eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="evaluate and write reports/eval_<mode>.json")
    run_parser.add_argument("--mode", choices=[*MODES, ALL], default=ALL)
    run_parser.add_argument("--no-llm", action="store_true", help="do not attempt the agent + RAGAS half")
    smoke = sub.add_parser("smoke", help="retrieval metrics on the fixture, nothing written")
    smoke.add_argument("--llm", action="store_true", help="also run a small agent + RAGAS sample (needs a key)")
    sub.add_parser("report", help="render reports/EVAL_RESULTS.md from the JSON files")
    return parser


def _modes(choice: str) -> tuple[RetrievalMode, ...]:
    return MODES if choice == ALL else (choice,)  # type: ignore[return-value]


def _write_all(settings: Settings, result: RunResult) -> None:
    from movie_rag.eval.report import write_report

    directory = settings.eval.resolve(settings.eval.reports_dir)
    for report in result.reports:
        say(f"wrote {write_report(report, directory)}")


def main(argv: Sequence[str] | None = None, *, deps: Dependencies | None = None) -> int:
    """CLI entry point; ``deps`` is injected by tests. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        settings = load_settings()
    except Exception as exc:  # an unreadable config is a user error: report it, never a traceback
        say(f"cannot load the configuration: {type(exc).__name__}: fix config.yaml / .env and re-run")
        logger.debug("configuration error", exc_info=exc)
        return EXIT_FAILURE
    try:
        if args.command == "report":
            return _report(settings)
        if args.command == "smoke":
            return _smoke(settings, llm=args.llm, deps=deps)
        return _run(settings, _modes(args.mode), llm=not args.no_llm, deps=deps)
    except MovieRagError as exc:
        say(f"evaluation failed: {exc}")
        return EXIT_FAILURE


def _run(settings: Settings, modes: Sequence[RetrievalMode], *, llm: bool, deps: Dependencies | None) -> int:
    result = run(settings, modes, attempt_llm=llm, deps=deps)
    _write_all(settings, result)
    emit(format_summary(result.reports))
    if llm and any(r.llm.status == "pending_credentials" for r in result.reports):
        say("LLM half skipped: pending credentials (set NEBIUS_API_KEY — see docs/CREDENTIALS.md)")
    return EXIT_FAILURE if result.failed else EXIT_OK


def _smoke(settings: Settings, *, llm: bool, deps: Dependencies | None) -> int:
    result = run(
        settings,
        MODES,
        attempt_llm=llm,
        llm_questions_per_type=settings.eval.smoke_llm_per_type,
        experiments=False,
        deps=deps,
    )
    emit(format_summary(result.reports))
    if llm and any(r.llm.status == "pending_credentials" for r in result.reports):
        say("LLM half skipped: pending credentials (set NEBIUS_API_KEY — see docs/CREDENTIALS.md)")
    problems = smoke_failures(settings, result.reports)
    for problem in problems:
        say(f"smoke failed: {problem}")
    return EXIT_FAILURE if problems or result.failed else EXIT_OK


def _report(settings: Settings) -> int:
    directory = settings.eval.resolve(settings.eval.reports_dir)
    reports = read_reports(directory)
    target = directory / RESULTS_NAME
    target.write_text(render_markdown(reports), encoding="utf-8")
    say(f"wrote {target}")
    readme = settings.eval.resolve(settings.eval.readme_path)
    readme.write_text(update_readme(readme.read_text(encoding="utf-8"), render_readme_block(reports)), encoding="utf-8")
    say(f"updated the results block of {readme}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
