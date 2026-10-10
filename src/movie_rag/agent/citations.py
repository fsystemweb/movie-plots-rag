"""Citations from tool results.

The model's free text is never the source of a citation's title, year or link: those come from the films the tools
returned. The text is only used to choose *which* retrieved films the answer actually mentions, by looking for a
retrieved film's exact ``Title (Year)`` label or its ``movie_id``. A film the model invented matches nothing and is
dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from langchain_core.messages import ToolMessage

from movie_rag.agent.models import Citation, RetrievedFilm

_ARTIFACT_KEY = "structured_content"  # langchain_mcp_adapters.tools.MCPToolArtifact


def _payload(message: ToolMessage) -> Any:
    """The structured result of a tool call: the MCP artifact if there is one, else JSON in the text content."""
    artifact = message.artifact
    if isinstance(artifact, Mapping) and isinstance(artifact.get(_ARTIFACT_KEY), Mapping):
        return artifact[_ARTIFACT_KEY]
    content = message.content
    blocks: Iterable[Any] = [content] if isinstance(content, str) else content
    for block in blocks:
        text = block if isinstance(block, str) else block.get("text") if isinstance(block, Mapping) else None
        if not text:
            continue
        try:
            return json.loads(text)
        except ValueError:
            continue
    return None


def _as_film(item: Any, tool: str | None) -> RetrievedFilm | None:
    if not isinstance(item, Mapping):
        return None
    movie_id, title = item.get("movie_id"), item.get("title")
    if not isinstance(movie_id, str) or not movie_id or not isinstance(title, str) or not title:
        return None
    year, url, score = item.get("release_year"), item.get("wiki_url"), item.get("score")
    return RetrievedFilm(
        movie_id=movie_id,
        title=title,
        release_year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        wiki_url=url if isinstance(url, str) and url else None,
        director=item.get("director") if isinstance(item.get("director"), str) else None,
        genre=item.get("genre") if isinstance(item.get("genre"), str) else None,
        origin=item.get("origin") if isinstance(item.get("origin"), str) else None,
        score=float(score) if isinstance(score, int | float) and not isinstance(score, bool) else None,
        snippet=item.get("snippet") if isinstance(item.get("snippet"), str) else None,
        tool=tool,
    )


def films_from_tool_message(message: ToolMessage) -> list[RetrievedFilm]:
    """Films in one tool result: ``results`` of search_movies / find_similar, or the single film of get_movie."""
    if message.status == "error":
        return []
    payload = _payload(message)
    if not isinstance(payload, Mapping):
        return []
    items: Sequence[Any] = payload["results"] if isinstance(payload.get("results"), list) else [payload]
    films = (_as_film(item, message.name) for item in items)
    return [film for film in films if film is not None]


def merge_films(films: Iterable[RetrievedFilm]) -> list[RetrievedFilm]:
    """One entry per ``movie_id`` (the first one seen), in order of retrieval."""
    seen: dict[str, RetrievedFilm] = {}
    for film in films:
        seen.setdefault(film.movie_id, film)
    return list(seen.values())


def select_citations(text: str, retrieved: Sequence[RetrievedFilm], limit: int) -> list[Citation]:
    """Retrieved films that ``text`` mentions, in order of first mention, at most ``limit``.

    A film is mentioned when its ``movie_id`` appears as a whole token, or its exact ``Title (Year)`` label appears and
    no other retrieved film has the same label (then only the id can tell them apart).
    """
    films = merge_films(retrieved)
    label_counts: dict[str, int] = {}
    for film in films:
        label_counts[film.label] = label_counts.get(film.label, 0) + 1
    found: list[tuple[int, RetrievedFilm]] = []
    for film in films:
        positions = [m.start() for m in re.finditer(rf"(?<![\w-]){re.escape(film.movie_id)}(?![\w-])", text)]
        if label_counts[film.label] == 1 and film.label in text:
            positions.append(text.index(film.label))
        if positions:
            found.append((min(positions), film))
    found.sort(key=lambda pair: pair[0])
    return [
        Citation(movie_id=f.movie_id, title=f.title, release_year=f.release_year, wiki_url=f.wiki_url)
        for _, f in found[:limit]
    ]
