"""Citations are built from tool results; the model's text only chooses among films that were retrieved."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage

from movie_rag.agent.citations import films_from_tool_message, merge_films, select_citations
from movie_rag.agent.models import RetrievedFilm


def hit(movie_id: str, title: str, year: int | None = 1999, **extra: Any) -> dict[str, Any]:
    return {
        "movie_id": movie_id,
        "title": title,
        "release_year": year,
        "wiki_url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "director": "Someone",
        "genre": "drama",
        "origin": "american",
        "score": 0.5,
        "snippet": "a plot",
        **extra,
    }


def film(movie_id: str, title: str, year: int | None = 1999) -> RetrievedFilm:
    return RetrievedFilm(movie_id=movie_id, title=title, release_year=year, wiki_url=f"https://w/{movie_id}")


def message(content: Any, *, artifact: Any = None, status: str = "success", name: str = "search_movies") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="t1", name=name, artifact=artifact, status=status)  # type: ignore[arg-type]


# --- films_from_tool_message ----------------------------------------------------------------------------------


def test_films_are_read_from_the_mcp_structured_artifact() -> None:
    payload = {"results": [hit("a-1", "Alpha"), hit("b-2", "Beta", None, score=3)]}
    films = films_from_tool_message(message("ignored text", artifact={"structured_content": payload}))
    assert [(f.movie_id, f.title, f.release_year, f.tool) for f in films] == [
        ("a-1", "Alpha", 1999, "search_movies"),
        ("b-2", "Beta", None, "search_movies"),
    ]
    assert films[0].wiki_url == "https://en.wikipedia.org/wiki/Alpha" and films[0].snippet == "a plot"
    assert films[1].score == 3.0


def test_films_fall_back_to_the_json_text_of_the_result() -> None:
    payload = json.dumps({"results": [hit("a-1", "Alpha")]})
    assert [f.movie_id for f in films_from_tool_message(message(payload))] == ["a-1"]
    blocks = [{"type": "text", "text": "not json"}, {"type": "text", "text": payload}, {"type": "image"}, ""]
    assert [f.movie_id for f in films_from_tool_message(message(blocks))] == ["a-1"]


def test_a_single_film_result_is_one_film() -> None:
    single = json.dumps({**hit("a-1", "Alpha"), "plot": "the whole plot"})
    films = films_from_tool_message(message(single, name="get_movie"))
    assert [(f.movie_id, f.tool) for f in films] == [("a-1", "get_movie")]


def test_results_without_films_yield_nothing() -> None:
    assert films_from_tool_message(message(json.dumps({"genres": ["drama"], "origins": []}), name="list_filters")) == []
    assert films_from_tool_message(message("plain text, no json")) == []
    assert films_from_tool_message(message(json.dumps([1, 2]))) == []
    assert films_from_tool_message(message(json.dumps({"results": []}))) == []
    assert films_from_tool_message(message([])) == []
    assert films_from_tool_message(message(["", {"type": "image"}])) == []


def test_error_results_and_malformed_entries_are_ignored() -> None:
    payload = json.dumps({"results": [hit("a-1", "Alpha")]})
    assert films_from_tool_message(message(payload, status="error")) == []
    broken = {
        "results": [
            "text",
            {"title": "No id"},
            {"movie_id": "", "title": "Empty id"},
            {"movie_id": "x"},
            hit("a-1", "Alpha"),
        ]
    }
    assert [f.movie_id for f in films_from_tool_message(message(json.dumps(broken)))] == ["a-1"]


def test_odd_field_types_become_none_instead_of_failing() -> None:
    odd = {"movie_id": "a-1", "title": "Alpha", "release_year": True, "wiki_url": "", "score": "high", "genre": 3}
    (parsed,) = films_from_tool_message(message(json.dumps({"results": [odd]})))
    assert (parsed.release_year, parsed.wiki_url, parsed.score, parsed.genre) == (None, None, None, None)


# --- merge_films / label ---------------------------------------------------------------------------------------


def test_merge_keeps_the_first_film_per_id_in_order() -> None:
    merged = merge_films([film("a", "Alpha"), film("b", "Beta"), film("a", "Alpha again")])
    assert [(f.movie_id, f.title) for f in merged] == [("a", "Alpha"), ("b", "Beta")]


def test_label_is_title_and_year() -> None:
    assert film("a", "Alpha", 1999).label == "Alpha (1999)"
    assert film("a", "Alpha", None).label == "Alpha"


# --- select_citations ------------------------------------------------------------------------------------------


def test_citations_are_the_mentioned_retrieved_films_in_order_of_mention() -> None:
    retrieved = [film("a-1", "Alpha"), film("b-2", "Beta"), film("c-3", "Gamma")]
    text = "Beta (1999) fits best. Also Alpha (1999) [a-1]."
    citations = select_citations(text, retrieved, limit=5)
    assert [c.movie_id for c in citations] == ["b-2", "a-1"]
    assert citations[0].wiki_url == "https://w/b-2" and citations[0].label == "Beta (1999)"


def test_a_film_the_model_invented_is_never_cited() -> None:
    retrieved = [film("a-1", "Alpha")]
    text = "Phantom (2001) [phantom-2001-9] https://en.wikipedia.org/wiki/Phantom and Alpha (1999)."
    assert [c.movie_id for c in select_citations(text, retrieved, limit=5)] == ["a-1"]
    assert select_citations("Phantom (2001) [phantom-2001-9]", retrieved, limit=5) == []


def test_a_wrong_year_is_not_a_mention_of_the_film() -> None:
    assert select_citations("Alpha (2005) is great", [film("a-1", "Alpha", 1999)], limit=5) == []


def test_ids_match_whole_tokens_only() -> None:
    retrieved = [film("film-1", "One"), film("film-10", "Ten")]
    assert [c.movie_id for c in select_citations("only film-10 matches", retrieved, limit=5)] == ["film-10"]
    assert select_citations("film-100 does not", retrieved, limit=5) == []
    assert [c.movie_id for c in select_citations("(film-1)", retrieved, limit=5)] == ["film-1"]


def test_identical_labels_need_the_id_to_tell_the_films_apart() -> None:
    retrieved = [film("remake-a", "Same", 2000), film("remake-b", "Same", 2000)]
    assert select_citations("Same (2000) is good", retrieved, limit=5) == []
    assert [c.movie_id for c in select_citations("Same (2000) [remake-b]", retrieved, limit=5)] == ["remake-b"]


def test_the_citation_limit_keeps_the_first_mentioned() -> None:
    retrieved = [film(f"f-{i}", f"Film{i}") for i in range(4)]
    text = " ".join(f"Film{i} (1999)" for i in (3, 1, 0, 2))
    assert [c.movie_id for c in select_citations(text, retrieved, limit=2)] == ["f-3", "f-1"]


def test_a_film_retrieved_twice_is_cited_once() -> None:
    retrieved = [film("a-1", "Alpha"), film("a-1", "Alpha")]
    assert len(select_citations("Alpha (1999) [a-1]", retrieved, limit=5)) == 1
