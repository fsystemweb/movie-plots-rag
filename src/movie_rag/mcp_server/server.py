"""The FastMCP server: four read-only tools on top of :class:`movie_rag.retrieval.Retriever`. No LLM calls in here.

* The retriever comes from a factory, so tests inject a fake and the real one (embedding models, Qdrant connection)
  is only created when the first tool is called.
* Every tool runs inside a LangSmith span (see :func:`movie_rag.observability.span`) carrying the mode, filters,
  latency and result count. Without a key the span is a no-op.
* Every failure a caller can act on becomes a :class:`fastmcp.exceptions.ToolError` with a plain message; anything else
  is logged and masked (``mask_error_details``) so internals never reach the model.

This module deliberately does not use ``from __future__ import annotations``: FastMCP resolves the tool signatures at
registration time and the argument limits come from the settings.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field, ValidationError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from starlette.requests import Request
from starlette.responses import JSONResponse

from movie_rag.config import RetrievalMode, Settings, load_settings
from movie_rag.errors import MovieNotFoundError, MovieRagError
from movie_rag.mcp_server.models import SearchResult, SimilarResult
from movie_rag.observability import span
from movie_rag.retrieval import FilterOptions, MovieDetail, Retriever, SearchFilters

logger = logging.getLogger(__name__)

SERVER_NAME = "movie-rag"
TOOL_NAMES = ("search_movies", "get_movie", "find_similar", "list_filters")
HTTP_NOT_FOUND = 404

RetrieverFactory = Callable[[], Retriever]


@contextmanager
def tool_errors(settings: Settings) -> Iterator[None]:
    """Translate retrieval and Qdrant failures into :class:`ToolError` with a message the caller can act on."""
    try:
        yield
    except ToolError:
        raise
    except ValidationError as exc:
        details = "; ".join(str(e["msg"]).removeprefix("Value error, ") for e in exc.errors())
        raise ToolError(f"invalid arguments: {details}") from exc
    except MovieNotFoundError as exc:
        raise ToolError(f"{exc}. Use search_movies to find a valid movie_id or title.") from exc
    except MovieRagError as exc:
        raise ToolError(str(exc)) from exc
    except UnexpectedResponse as exc:
        logger.warning("qdrant answered %s", exc.status_code)
        if exc.status_code == HTTP_NOT_FOUND:
            raise ToolError(
                f"The movie index '{settings.qdrant.collection}' does not exist yet: run `make ingest` and retry."
            ) from exc
        raise ToolError("The movie index (Qdrant) returned an error; retry later.") from exc
    except (ResponseHandlingException, ConnectionError, TimeoutError) as exc:
        logger.warning("qdrant unavailable: %s", type(exc).__name__)
        raise ToolError("The movie index (Qdrant) is unavailable; start it with `make up` and retry.") from exc


def _detail_label(film: MovieDetail) -> str:
    year = film.release_year if film.release_year is not None else "year unknown"
    return f"{film.title} ({year}) = {film.movie_id}"


def build_server(settings: Settings | None = None, retriever_factory: RetrieverFactory | None = None) -> FastMCP:
    """The ``movie-rag`` MCP server. ``retriever_factory`` is called once, lazily, on the first tool call."""
    cfg = settings or load_settings()
    mcp_cfg = cfg.mcp
    holder: list[Retriever] = []

    def retriever() -> Retriever:
        if not holder:
            holder.append(retriever_factory() if retriever_factory else Retriever(cfg))
        return holder[0]

    mcp = FastMCP(SERVER_NAME, mask_error_details=True)

    @mcp.custom_route(mcp_cfg.health_path, methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        """Liveness probe for the Docker healthcheck (does not touch Qdrant)."""
        return JSONResponse({"status": "ok", "server": SERVER_NAME})

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
    def search_movies(
        query: Annotated[
            str,
            Field(
                min_length=mcp_cfg.min_query_chars,
                max_length=mcp_cfg.max_query_chars,
                description="Natural-language description of the plot, characters or premise of the film to find.",
            ),
        ],
        mode: Annotated[
            RetrievalMode,
            Field(
                description="'hybrid' (default, best for most questions), 'dense' (meaning only, for vague "
                "descriptions) or 'sparse' (keywords only, for names and exact terms)."
            ),
        ] = cfg.retrieval.default_mode,
        top_k: Annotated[
            int, Field(ge=1, le=mcp_cfg.max_top_k, description="How many films to return (best first).")
        ] = cfg.retrieval.top_k,
        year_from: Annotated[int | None, Field(description="Only films released in or after this year.")] = None,
        year_to: Annotated[int | None, Field(description="Only films released in or before this year.")] = None,
        genre: Annotated[
            str | None, Field(description="Only films of exactly this genre (see list_filters for valid values).")
        ] = None,
        origin: Annotated[
            str | None,
            Field(description="Only films of exactly this origin/ethnicity (see list_filters for valid values)."),
        ] = None,
    ) -> SearchResult:
        """Find films whose plot matches a description. Use this first for any 'which movie is it' question.

        Returns up to top_k distinct films, best match first, each with movie_id, title, release_year, director, genre,
        origin, score, a snippet of the best matching plot passage (at most a few hundred characters) and wiki_url.
        Facts about a film must come from these results (or get_movie); cite every film as 'Title (Year)' plus its
        wiki_url. An empty result means nothing matched: relax the filters or try another mode instead of guessing.
        """
        with (
            tool_errors(cfg),
            span(
                "mcp.search_movies",
                run_type="tool",
                settings=cfg,
                inputs={"query": query},
                tool="search_movies",
                retrieval_mode=mode,
                top_k=top_k,
                year_from=year_from,
                year_to=year_to,
                genre=genre,
                origin=origin,
            ) as run,
        ):
            filters = SearchFilters(year_from=year_from, year_to=year_to, genre=genre, origin=origin)
            hits = retriever().search(query, mode=mode, top_k=top_k, filters=filters)
            note = None
            if not hits:
                note = (
                    "No film matched. Try fewer or wider filters (check valid genre/origin values with "
                    "list_filters) or another mode."
                )
            run.set(result_count=len(hits))
            return SearchResult(query=query.strip(), mode=mode, count=len(hits), results=hits, note=note)

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
    def get_movie(
        movie_id: Annotated[
            str | None, Field(description="The movie_id returned by search_movies or find_similar (preferred).")
        ] = None,
        title: Annotated[
            str | None,
            Field(description="Exact film title, as written in search results (case-sensitive). Use if no movie_id."),
        ] = None,
        year: Annotated[
            int | None, Field(description="Release year; only with title, to tell remakes with the same title apart.")
        ] = None,
    ) -> MovieDetail:
        """Get one film's full metadata and its complete plot. Use it to verify details of a film already found.

        Pass exactly one of movie_id or title. A title must match exactly; if several films share it (remakes) the
        error lists their ids and years: call again with movie_id, or with title and year. Cite the film as
        'Title (Year)' plus its wiki_url.
        """
        with (
            tool_errors(cfg),
            span(
                "mcp.get_movie",
                run_type="tool",
                settings=cfg,
                inputs={"movie_id": movie_id, "title": title, "year": year},
                tool="get_movie",
                lookup="movie_id" if movie_id else "title",
            ) as run,
        ):
            movie_id = movie_id.strip() if movie_id else None
            title = title.strip() if title else None
            if bool(movie_id) == bool(title):
                raise ToolError("pass exactly one of movie_id or title")
            if year is not None and not title:
                raise ToolError("year can only be combined with title")
            film: MovieDetail | None
            if movie_id:
                film = retriever().get_movie(movie_id)
                if film is None:
                    raise ToolError(f"no film with movie_id {movie_id!r}. Use search_movies to find a valid movie_id.")
            else:
                assert title is not None
                matches = retriever().find_by_title(title, year=year, limit=mcp_cfg.title_lookup_limit)
                if not matches:
                    raise ToolError(
                        f"no film titled {title!r}"
                        + (f" from {year}" if year is not None else "")
                        + ". Titles must match exactly; use search_movies to find the right title."
                    )
                if len(matches) > 1:
                    options = "; ".join(_detail_label(f) for f in matches)
                    raise ToolError(
                        f"{len(matches)} films are titled {title!r}: {options}. Call again with movie_id, "
                        "or with title and year."
                    )
                film = matches[0]
            run.set(movie_id=film.movie_id, result_count=1)
            return film

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
    def find_similar(
        movie_id: Annotated[str, Field(min_length=1, description="The movie_id of the film to start from.")],
        top_k: Annotated[
            int, Field(ge=1, le=mcp_cfg.max_top_k, description="How many similar films to return.")
        ] = cfg.retrieval.top_k,
    ) -> SimilarResult:
        """Find films with plots similar to a given film ('more like this'). Needs a movie_id from another tool.

        Returns the closest films by plot meaning, most similar first, never including the film itself, in the same
        format as search_movies. Cite each as 'Title (Year)' plus its wiki_url.
        """
        with (
            tool_errors(cfg),
            span(
                "mcp.find_similar",
                run_type="tool",
                settings=cfg,
                inputs={"movie_id": movie_id},
                tool="find_similar",
                retrieval_mode="dense",
                top_k=top_k,
            ) as run,
        ):
            hits = retriever().find_similar(movie_id.strip(), top_k=top_k)
            run.set(result_count=len(hits))
            return SimilarResult(movie_id=movie_id.strip(), count=len(hits), results=hits)

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
    def list_filters() -> FilterOptions:
        """List the values the search filters accept: genres, origins and the release-year range of the collection.

        Call it before filtering by genre or origin, which must match exactly (lower case). Genres and origins are
        sorted by how many films have them; truncated is true when the lists were cut.
        """
        with tool_errors(cfg), span("mcp.list_filters", run_type="tool", settings=cfg, tool="list_filters") as run:
            options = retriever().list_filters(limit=mcp_cfg.max_filter_values)
            run.set(genres=len(options.genres), origins=len(options.origins))
            return options

    return mcp
