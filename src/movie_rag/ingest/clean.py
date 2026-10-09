"""Clean the raw Kaggle "Wikipedia Movie Plots" rows into :class:`MovieRecord` objects.

Rules (KICKOFF section 1, "Preprocessing"):

* drop plots shorter than ``ingest.min_plot_words`` words (whitespace-separated);
* normalise ``genre`` and ``origin`` to lowercase; ``"unknown"`` (any case) and empty values become ``None``
  (director and cast get the same ``"unknown"`` -> ``None`` treatment, without lowercasing);
* ``movie_id`` is ``slug(title)-year-row_index`` where ``row_index`` is the 0-based position of the row in the raw
  file, counted *before* any row is dropped. It is therefore stable across runs and unaffected by changing the
  cleaning thresholds;
* any sampling uses a fixed seed (``ingest.random_seed``).
"""

from __future__ import annotations

import csv
import logging
import random
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from movie_rag.errors import DataError

logger = logging.getLogger(__name__)

RAW_COLUMNS: tuple[str, ...] = (
    "Release Year",
    "Title",
    "Origin/Ethnicity",
    "Director",
    "Cast",
    "Genre",
    "Wiki Page",
    "Plot",
)
UNKNOWN_LABELS = frozenset({"", "unknown"})
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"^\s*(\d{4})\s*$")
EMPTY_SLUG = "untitled"

DropReason = Literal["short_plot", "invalid"]


class MovieRecord(BaseModel):
    """One cleaned film. ``movie_id`` is the primary key used by ingestion, retrieval and the MCP tools."""

    model_config = ConfigDict(frozen=True)

    movie_id: str
    title: str
    release_year: int
    origin: str | None
    director: str | None
    cast: str | None
    genre: str | None
    wiki_url: str | None
    plot: str


class CleanStats(BaseModel):
    """Row accounting for one cleaning run (logged by ingestion)."""

    rows_read: int = 0
    rows_kept: int = 0
    dropped_short_plot: int = 0
    dropped_invalid: int = 0

    @property
    def rows_dropped(self) -> int:
        return self.dropped_short_plot + self.dropped_invalid


def slugify(title: str) -> str:
    """ASCII, lowercase, hyphen-separated slug; ``"Amélie!"`` -> ``"amelie"``; nothing usable -> ``"untitled"``."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("-", ascii_title.lower()).strip("-") or EMPTY_SLUG


def make_movie_id(title: str, year: int, row_index: int) -> str:
    """Stable id: ``slug(title)-year-row_index``. Pure function of its arguments."""
    return f"{slugify(title)}-{year}-{row_index}"


def normalise_label(value: str | None) -> str | None:
    """Lowercase, trim and collapse whitespace; empty and ``"unknown"`` become ``None``."""
    if value is None:
        return None
    collapsed = _WHITESPACE.sub(" ", value).strip().lower()
    return None if collapsed in UNKNOWN_LABELS else collapsed


def _clean_text(value: str | None) -> str | None:
    """Trim and collapse whitespace, keeping case; empty and ``"unknown"`` become ``None``."""
    if value is None:
        return None
    collapsed = _WHITESPACE.sub(" ", value).strip()
    return None if collapsed.lower() in UNKNOWN_LABELS else collapsed


def count_words(text: str) -> int:
    return len(text.split())


def parse_year(value: str | None) -> int | None:
    """A four-digit year, else ``None``."""
    match = _YEAR.match(value or "")
    return int(match.group(1)) if match else None


def clean_row(row: Mapping[str, str | None], row_index: int, min_plot_words: int) -> MovieRecord | DropReason:
    """Clean one raw row.

    Returns a :class:`MovieRecord`, or the drop reason ``"short_plot"`` / ``"invalid"``.
    """
    title = _WHITESPACE.sub(" ", row.get("Title") or "").strip()
    year = parse_year(row.get("Release Year"))
    plot = _WHITESPACE.sub(" ", row.get("Plot") or "").strip()
    if not title or year is None:
        return "invalid"
    if count_words(plot) < min_plot_words:
        return "short_plot"
    return MovieRecord(
        movie_id=make_movie_id(title, year, row_index),
        title=title,
        release_year=year,
        origin=normalise_label(row.get("Origin/Ethnicity")),
        director=_clean_text(row.get("Director")),
        cast=_clean_text(row.get("Cast")),
        genre=normalise_label(row.get("Genre")),
        wiki_url=_clean_text(row.get("Wiki Page")),
        plot=plot,
    )


def clean_rows(rows: Iterable[Mapping[str, str | None]], min_plot_words: int) -> tuple[list[MovieRecord], CleanStats]:
    """Clean raw rows in order. ``row_index`` is each row's position in ``rows`` (0-based)."""
    stats = CleanStats()
    records: list[MovieRecord] = []
    for row_index, row in enumerate(rows):
        stats.rows_read += 1
        result = clean_row(row, row_index, min_plot_words)
        if isinstance(result, MovieRecord):
            records.append(result)
            stats.rows_kept += 1
        elif result == "short_plot":
            stats.dropped_short_plot += 1
        else:
            stats.dropped_invalid += 1
    return records, stats


def read_raw_rows(path: Path) -> list[dict[str, str | None]]:
    """Read the raw CSV, checking that the Kaggle schema is intact."""
    if not path.is_file():
        raise DataError(
            f"dataset file not found: {path}. Run `make download`, or point at tests/fixtures/movies_sample.csv."
        )
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in RAW_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"{path} is missing expected columns: {', '.join(missing)}")
        return list(reader)


def clean_csv(path: Path, min_plot_words: int) -> tuple[list[MovieRecord], CleanStats]:
    """Read ``path`` and clean it; logs the row accounting."""
    records, stats = clean_rows(read_raw_rows(path), min_plot_words)
    logger.info(
        "cleaned %s: rows_read=%d kept=%d dropped_short_plot=%d dropped_invalid=%d",
        path.name,
        stats.rows_read,
        stats.rows_kept,
        stats.dropped_short_plot,
        stats.dropped_invalid,
    )
    return records, stats


def sample_records(records: Sequence[MovieRecord], n: int, seed: int) -> list[MovieRecord]:
    """Deterministic sample of ``n`` records (all of them if fewer); same seed, same sample, in input order."""
    if n >= len(records):
        return list(records)
    chosen = sorted(random.Random(seed).sample(range(len(records)), n))
    return [records[i] for i in chosen]
