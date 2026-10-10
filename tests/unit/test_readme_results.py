"""The README results block is the rendering of the committed reports; nothing else in the README quotes metrics."""

from __future__ import annotations

import re
from pathlib import Path

from movie_rag.config import PROJECT_ROOT, load_settings
from movie_rag.eval.report import README_BEGIN, README_END, read_reports, render_readme_block, update_readme

README = PROJECT_ROOT / "README.md"


def _reports_dir() -> Path:
    settings = load_settings(env_file=None)
    return settings.eval.resolve(settings.eval.reports_dir)


def test_the_readme_results_block_matches_the_committed_reports() -> None:
    text = README.read_text(encoding="utf-8")
    block = render_readme_block(read_reports(_reports_dir()))
    assert block in text, "README results block is stale: run `make report` and commit README.md"
    assert update_readme(text, block) == text


def test_the_readme_has_one_results_block_and_no_hand_typed_metrics_outside_it() -> None:
    text = README.read_text(encoding="utf-8")
    assert text.count(README_BEGIN) == 1 and text.count(README_END) == 1
    outside = text.split(README_BEGIN)[0] + text.split(README_END)[1]
    # a metric looks like 0.xyz / 1.000 (three decimals, not a version number) or "Hit@1 = 0.9": none typed by hand
    assert not re.findall(r"(?<![\d.])[01]\.\d{3}(?![\d.])", outside)
    assert not re.findall(r"(?:Hit@\d+|MRR)\s*(?:is|=|:)?\s*[01]\.\d", outside)
