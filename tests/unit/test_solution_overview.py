"""The stakeholder overview: generated numbers match the reports, and the text stays short and plain.

``docs/SOLUTION_OVERVIEW.md`` is written for readers with no AI background (KICKOFF section 8): at most 1,500 words,
every sentence under 25 words, the nine sections in order, a glossary, and no metric typed by hand. Its two
``OVERVIEW`` blocks are rendered by ``make report`` from ``reports/eval_<mode>.json``.

Sentence heuristic: fenced code (the Mermaid flow), HTML comments (block markers), tables and headings are not
prose and are skipped; links keep their text only; bullets are separate sentences; other lines of a paragraph are
joined and split after ``.``, ``!`` or ``?`` followed by a space. The word limit counts everything except the
fenced Mermaid flow and the HTML comments (block markers), tables included.
"""

from __future__ import annotations

import re
from pathlib import Path

from movie_rag.config import PROJECT_ROOT, load_settings
from movie_rag.eval.report import (
    OVERVIEW_QUALITY_BEGIN,
    OVERVIEW_QUALITY_END,
    OVERVIEW_SPEED_BEGIN,
    OVERVIEW_SPEED_END,
    read_reports,
    update_overview,
)

OVERVIEW = PROJECT_ROOT / "docs" / "SOLUTION_OVERVIEW.md"
MAX_WORDS = 1500
MAX_SENTENCE_WORDS = 24  # "sentences under 25 words"
SECTIONS = [
    "The problem",
    "What it does",
    "How it works in five steps",
    "How we know it works",
    "Cost and speed",
    "Limits",
    "How it was built",
    "What's next",
    "Glossary",
]
GLOSSARY_TERMS = ["RAG", "Embedding", "Vector database", "Hybrid search", "MCP", "Evaluation"]
FENCE = re.compile(r"^```.*?^```[ \t]*$", re.DOTALL | re.MULTILINE)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
LIST_ITEM = re.compile(r"^\s*(?:[*-]|\d+\.)\s+")


def _text() -> str:
    return OVERVIEW.read_text(encoding="utf-8")


def _reports_dir() -> Path:
    settings = load_settings(env_file=None)
    return settings.eval.resolve(settings.eval.reports_dir)


def _plain(line: str) -> str:
    return line.replace("*", "").replace("`", "").strip()


def sentences(markdown: str) -> list[str]:
    """The prose sentences of ``markdown`` (see the module docstring for what is skipped)."""
    text = COMMENT.sub("", FENCE.sub("", markdown))
    text = LINK.sub(r"\1", text)
    units: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        current: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "|")):
                continue
            if LIST_ITEM.match(line):
                units += [" ".join(current)] if current else []
                current = [_plain(LIST_ITEM.sub("", line))]
            else:
                current.append(_plain(stripped))
        units += [" ".join(current)] if current else []
    return [s.strip() for unit in units for s in re.split(r"(?<=[.!?])\s+", unit) if s.strip()]


def test_the_overview_result_blocks_match_the_committed_reports() -> None:
    text = _text()
    assert update_overview(text, read_reports(_reports_dir())) == text, (
        "overview numbers are stale: run `make report` and commit docs/SOLUTION_OVERVIEW.md"
    )


def test_each_generated_block_appears_exactly_once() -> None:
    text = _text()
    for marker in (OVERVIEW_QUALITY_BEGIN, OVERVIEW_QUALITY_END, OVERVIEW_SPEED_BEGIN, OVERVIEW_SPEED_END):
        assert text.count(marker) == 1


def test_no_metric_is_typed_by_hand_outside_the_generated_blocks() -> None:
    text = _text()
    for begin, end in (
        (OVERVIEW_QUALITY_BEGIN, OVERVIEW_QUALITY_END),
        (OVERVIEW_SPEED_BEGIN, OVERVIEW_SPEED_END),
    ):
        text = text.split(begin)[0] + text.split(end)[1]
    assert not re.findall(r"(?<![\d.])[01]\.\d{2,3}(?![\d.])", text), "a two or three decimal figure outside a block"
    assert not re.findall(r"(?:Hit@\d+|MRR)\s*(?:is|=|:)?\s*[01]\.\d", text)
    assert not re.findall(r"\d+(?:\.\d+)?\s*%\s+(?:of|accuracy|faithful)", text.replace("80% test coverage", ""))


def test_the_overview_has_at_most_1500_words() -> None:
    words = len(COMMENT.sub("", FENCE.sub("", _text())).split())
    assert words <= MAX_WORDS, f"{words} words (limit {MAX_WORDS})"


def test_every_sentence_is_under_25_words() -> None:
    too_long = [(len(s.split()), s) for s in sentences(_text()) if len(s.split()) > MAX_SENTENCE_WORDS]
    assert not too_long, "\n".join(f"{n} words: {s}" for n, s in too_long)


def test_the_nine_sections_come_in_order() -> None:
    headings = re.findall(r"^## (?:\d+\. )?(.+)$", _text(), flags=re.MULTILINE)
    assert headings == SECTIONS
    numbered = re.findall(r"^## (\d+)\. ", _text(), flags=re.MULTILINE)
    assert numbered == [str(i) for i in range(1, 10)]


def test_the_flow_is_a_mermaid_diagram_with_five_steps() -> None:
    match = re.search(r"```mermaid\n(.*?)```", _text(), flags=re.DOTALL)
    assert match is not None
    assert match.group(1).lstrip().startswith("flowchart")
    assert len(re.findall(r"\b[A-Z]\[\"", match.group(1))) == 5


def test_the_glossary_defines_the_required_terms() -> None:
    glossary = _text().split("## 9. Glossary")[1]
    for term in GLOSSARY_TERMS:
        assert re.search(rf"^\* \*\*{re.escape(term)}\b", glossary, flags=re.MULTILINE), term


def test_unfinished_numbers_are_marked_pending_not_invented() -> None:
    text = _text()
    assert "pending credentials" in text
    assert re.search(r"Cost per 1,000 questions: pending credentials", text)


def test_the_sentence_splitter_skips_code_tables_comments_and_headings() -> None:
    sample = (
        "# Title\n\nOne two three. Four five?\n\n| a | b |\n|---|---|\n| c. d. | e |\n\n```mermaid\nx --> y. z\n```\n\n"
        "<!-- hidden. text -->\n\n* bullet one. More\n  words here\n* [link text](http://x.y/z.md) ends.\n"
    )
    assert sentences(sample) == ["One two three.", "Four five?", "bullet one.", "More words here", "link text ends."]
