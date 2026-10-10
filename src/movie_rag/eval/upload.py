"""Upload the evaluation questions to LangSmith as a dataset: ``python -m movie_rag.eval.upload``.

Without ``LANGSMITH_API_KEY`` the command prints a skip notice and exits 0 (a missing key is not an error, see
docs/CREDENTIALS.md). With a key it creates the dataset ``eval.dataset_name`` and one example per question:

* inputs: ``question`` and the ``filters`` the question carries;
* outputs: ``gold_movie_ids`` and ``expect_abstention`` (true for unanswerable questions);
* metadata: ``question_id``, ``type`` and, for unanswerable questions, ``absent_title``.

An existing dataset of that name is left alone (re-uploading would duplicate the examples): change
``eval.dataset_name`` when the question set changes.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import Any, Literal

from langsmith import Client
from pydantic import BaseModel

from movie_rag.config import Settings, load_settings
from movie_rag.errors import MovieRagError
from movie_rag.eval.questions import EvalQuestion, load_eval_set
from movie_rag.ingest.download import say

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
SKIP_MESSAGE = "skipped: no LANGSMITH_API_KEY (set it in .env to upload the evaluation dataset, docs/CREDENTIALS.md)"


class UploadResult(BaseModel):
    """What :func:`upload_questions` did."""

    status: Literal["skipped", "exists", "created"]
    dataset_name: str
    examples: int = 0

    def describe(self) -> str:
        if self.status == "skipped":
            return SKIP_MESSAGE
        if self.status == "exists":
            return (
                f"LangSmith dataset {self.dataset_name!r} already exists: left untouched "
                "(change eval.dataset_name to upload a new version)"
            )
        return f"created LangSmith dataset {self.dataset_name!r} with {self.examples} example(s)"


def to_example(question: EvalQuestion) -> dict[str, Any]:
    """One LangSmith example for ``question``."""
    metadata: dict[str, Any] = {"question_id": question.id, "type": question.type}
    if question.absent_title is not None:
        metadata["absent_title"] = question.absent_title
    return {
        "inputs": {"question": question.question, "filters": question.filters.model_dump(exclude_none=True)},
        "outputs": {"gold_movie_ids": question.gold_movie_ids, "expect_abstention": question.type == "unanswerable"},
        "metadata": metadata,
    }


def make_client(settings: Settings) -> Client:
    """A LangSmith client for the configured key and API URL (call only when the key is set)."""
    key = settings.langsmith_api_key
    if key is None:
        raise ValueError("make_client needs LANGSMITH_API_KEY")
    return Client(api_key=key.get_secret_value(), api_url=settings.observability.langsmith_api_url)


def upload_questions(
    settings: Settings, questions: Sequence[EvalQuestion], client: Client | None = None
) -> UploadResult:
    """Create the dataset for ``questions`` unless it exists. Skips (no network) without ``LANGSMITH_API_KEY``."""
    name = settings.eval.dataset_name
    if settings.langsmith_api_key is None:
        return UploadResult(status="skipped", dataset_name=name)
    ls = client or make_client(settings)
    if ls.has_dataset(dataset_name=name):
        return UploadResult(status="exists", dataset_name=name)
    dataset = ls.create_dataset(
        name,
        description=f"Movie Plots RAG evaluation questions ({len(questions)}), see docs/EVAL_SET.md",
    )
    ls.create_examples(dataset_id=dataset.id, examples=[to_example(q) for q in questions])
    return UploadResult(status="created", dataset_name=name, examples=len(questions))


def main(*, client: Client | None = None) -> int:
    """CLI entry point; ``client`` can be injected (tests). Returns the process exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        settings = load_settings()
        questions = load_eval_set(settings)
        result = upload_questions(settings, questions, client)
    except MovieRagError as exc:
        say(str(exc))
        return EXIT_FAILURE
    say(result.describe())
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
