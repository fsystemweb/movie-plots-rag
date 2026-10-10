"""Generate fuzzy-plot questions by paraphrasing sampled plots: ``python -m movie_rag.eval.generate``.

Needs ``NEBIUS_API_KEY`` (the chat model writes the questions); without it the command prints the credentials hint and
exits 2 before touching the network. The result goes to ``eval.generated_path``, never over the committed
``questions_v1.jsonl``, so a person (or the QA agent) reviews it before it replaces anything.

Procedure, all of it deterministic given the model's answers:

1. shuffle the films with ``ingest.random_seed`` and walk them in that order;
2. for each film ask the model for a paraphrase (versioned prompt ``prompts/<eval.paraphrase_prompt_version>.md``);
3. reject a question that shares an ``eval.ngram_size``-word run with the plot or names the title, and ask again with
   the offending phrases listed, up to ``eval.max_attempts`` times;
4. a film that never yields a clean question is recorded as rejected and the walk moves on to the next film;
5. stop after ``eval.generate_count`` accepted questions.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from movie_rag.agent.agent import make_chat_model
from movie_rag.config import Settings, load_settings
from movie_rag.errors import EvalSetError, MissingCredentialError, MovieRagError
from movie_rag.eval.overlap import contains_phrase, shared_ngrams
from movie_rag.eval.questions import EvalQuestion, fixture_records, write_questions
from movie_rag.ingest.clean import MovieRecord
from movie_rag.ingest.download import say

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
ID_PREFIX = "gen-fuzzy"


class Rejection(BaseModel):
    """A film for which no acceptable question was produced."""

    movie_id: str
    attempts: int
    reasons: list[str] = Field(description="Why the last attempt was refused.")


class GenerationReport(BaseModel):
    """Outcome of one generator run."""

    questions: list[EvalQuestion]
    rejected: list[Rejection]
    model_calls: int
    prompt_version: str
    seed: int


def load_paraphrase_prompt(settings: Settings) -> str:
    """The system prompt for ``eval.paraphrase_prompt_version`` with the guard size filled in."""
    version = settings.eval.paraphrase_prompt_version
    path = PROMPT_DIR / f"{version}.md"
    if not path.is_file():
        raise EvalSetError(f"unknown paraphrase prompt version {version!r}: expected {path}")
    return path.read_text(encoding="utf-8").format_map({"ngram_size": settings.eval.ngram_size})


def clean_model_text(text: str) -> str:
    """The first non-empty line of the model's reply without surrounding quotes."""
    for line in text.strip().splitlines():
        stripped = line.strip().strip("\"'“”")
        if stripped:
            return stripped
    return ""


def refusal_reasons(question: str, film: MovieRecord, ngram_size: int) -> list[str]:
    """Why ``question`` is not acceptable for ``film`` (empty list: it passes the guard)."""
    if not question:
        return ["the reply was empty"]
    reasons = []
    copied = shared_ngrams(question, film.plot, ngram_size)
    if copied:
        reasons.append(f"it copies {ngram_size}-word phrases from the plot: " + "; ".join(f'"{c}"' for c in copied))
    if contains_phrase(question, film.title):
        reasons.append("it contains the film's title")
    return reasons


def _human_message(film: MovieRecord, feedback: Sequence[str]) -> HumanMessage:
    text = f"Plot:\n{film.plot}"
    if feedback:
        text += "\n\nYour previous question was refused because " + " and ".join(feedback) + ". Write a new one."
    return HumanMessage(content=text)


def _reply_text(message: BaseMessage) -> str:
    return clean_model_text(message.text)


def generate_questions(
    settings: Settings, records: Sequence[MovieRecord], model: BaseChatModel | None = None
) -> GenerationReport:
    """Paraphrase sampled plots into fuzzy questions (see the module docstring).

    Raises ``MissingCredentialError`` when no ``model`` is given and ``NEBIUS_API_KEY`` is not set.
    """
    cfg = settings.eval
    chat = model if model is not None else make_chat_model(settings)
    system = SystemMessage(content=load_paraphrase_prompt(settings))
    order = list(records)
    random.Random(settings.ingest.random_seed).shuffle(order)
    report = GenerationReport(
        questions=[],
        rejected=[],
        model_calls=0,
        prompt_version=cfg.paraphrase_prompt_version,
        seed=settings.ingest.random_seed,
    )
    for film in order:
        if len(report.questions) >= cfg.generate_count:
            break
        feedback: list[str] = []
        for attempt in range(1, cfg.max_attempts + 1):
            report.model_calls += 1
            question = _reply_text(chat.invoke([system, _human_message(film, feedback)]))
            feedback = refusal_reasons(question, film, cfg.ngram_size)
            if not feedback:
                report.questions.append(
                    EvalQuestion(
                        id=f"{ID_PREFIX}-{len(report.questions) + 1:02d}",
                        type="fuzzy_plot",
                        question=question,
                        gold_movie_ids=[film.movie_id],
                        notes=f"generated: seed {report.seed}, prompt {report.prompt_version}, attempt {attempt}",
                    )
                )
                break
            logger.info("%s attempt %d refused: %s", film.movie_id, attempt, "; ".join(feedback))
        else:
            report.rejected.append(Rejection(movie_id=film.movie_id, attempts=cfg.max_attempts, reasons=feedback))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m movie_rag.eval.generate",
        description="Paraphrase sampled fixture plots into fuzzy evaluation questions (needs NEBIUS_API_KEY).",
    )
    parser.add_argument("--out", type=Path, help="output JSONL (default: eval.generated_path from config.yaml)")
    return parser


def main(argv: Sequence[str] | None = None, *, model: BaseChatModel | None = None) -> int:
    """CLI entry point; ``model`` can be injected (tests). Returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        settings = load_settings()
        report = generate_questions(settings, fixture_records(settings), model)
    except MissingCredentialError as exc:
        say(str(exc))
        return EXIT_USAGE
    except MovieRagError as exc:
        say(str(exc))
        return EXIT_FAILURE
    out = args.out or settings.eval.resolve(settings.eval.generated_path)
    write_questions(report.questions, out)
    say(
        f"wrote {len(report.questions)} question(s) to {out} "
        f"({report.model_calls} model call(s), {len(report.rejected)} film(s) rejected after "
        f"{settings.eval.max_attempts} attempts each; seed {report.seed}, prompt {report.prompt_version})"
    )
    for rejection in report.rejected:
        say(f"  rejected {rejection.movie_id}: {'; '.join(rejection.reasons)}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
