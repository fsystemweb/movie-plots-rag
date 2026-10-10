"""Streamlit test page: rendering only. Everything it does goes through :class:`movie_rag.ui.service.MovieService`.

Run with ``make ui`` (``streamlit run``). The page needs ``make serve`` for every mode, and ``NEBIUS_API_KEY`` only
when "Retrieval only" is off.
"""

from __future__ import annotations

import streamlit as st

from movie_rag.agent.models import Answer, RetrievedFilm
from movie_rag.retrieval.search import MODES
from movie_rag.ui import service as service_module
from movie_rag.ui.service import Comparison, MovieService, RetrievalOutcome, SearchParams, Turn, UiError

ANY = "(any)"
TURNS_KEY = "turns"
FILTERS_KEY = "filter_load"
COMPARISON_KEY = "comparison"
MS_PER_S = 1000


@st.cache_resource(show_spinner=False)
def get_service() -> MovieService:
    """One service per server process (settings and a lazily-built agent; no connection is held)."""
    return service_module.build_service()


def md(text: str) -> str:
    """Escape what Streamlit markdown would interpret in data or model text (``$`` starts a formula)."""
    return text.replace("$", r"\$")


def film_label(film: RetrievedFilm) -> str:
    return f"{film.title} ({film.release_year})" if film.release_year is not None else film.title


# --- sidebar ------------------------------------------------------------------------------------------------------


def sidebar(service: MovieService) -> tuple[SearchParams, bool]:
    """Draw the controls and return the search parameters and the retrieval-only switch."""
    cfg = service.settings
    with st.sidebar:
        st.header("Search settings")
        retrieval_only = st.toggle(
            "Retrieval only (no LLM, no key needed)",
            value=cfg.nebius_api_key is None,
            key="retrieval_only",
            help="Ranks films with the MCP search tool and never calls the chat model.",
        )
        mode = st.selectbox("Retrieval mode", MODES, index=MODES.index(cfg.retrieval.default_mode), key="mode")
        top_k = st.slider("Films to retrieve (top_k)", 1, cfg.mcp.max_top_k, cfg.retrieval.top_k, key="top_k")
        st.caption("In agent mode the model picks its own top_k; the mode and the filters below are passed on.")

        if st.session_state.get(FILTERS_KEY) is None or st.session_state[FILTERS_KEY].options is None:
            st.session_state[FILTERS_KEY] = service.load_filters()
        loaded = st.session_state[FILTERS_KEY]
        year_from = year_to = genre = origin = None
        if loaded.error is not None:
            st.warning(f"Filters unavailable. {loaded.error.message}")
        else:
            options = loaded.options
            if options.year_min is not None and options.year_max is not None and options.year_min < options.year_max:
                low, high = st.slider(
                    "Release year",
                    options.year_min,
                    options.year_max,
                    (options.year_min, options.year_max),
                    key="years",
                )
                year_from = low if low > options.year_min else None  # the full range means "no filter"
                year_to = high if high < options.year_max else None
            picked_genre = st.selectbox("Genre", [ANY, *options.genres], key="genre")
            picked_origin = st.selectbox("Origin", [ANY, *options.origins], key="origin")
            genre = None if picked_genre == ANY else picked_genre
            origin = None if picked_origin == ANY else picked_origin
            if options.truncated:
                st.caption("The genre and origin lists are cut to the most frequent values.")
    params = SearchParams(mode=mode, top_k=top_k, year_from=year_from, year_to=year_to, genre=genre, origin=origin)
    return params, retrieval_only


# --- rendering ----------------------------------------------------------------------------------------------------


def render_error(error: UiError) -> None:
    if error.kind == "credentials":
        st.info(error.message)
    else:
        st.error(error.message)


def render_films(films: list[RetrievedFilm]) -> None:
    for rank, film in enumerate(films, start=1):
        details = " | ".join(v for v in (film.genre, film.origin, film.director) if v)
        score = f"score {film.score:.3f}" if film.score is not None else ""
        title = f"[{md(film_label(film))}]({film.wiki_url})" if film.wiki_url else md(film_label(film))
        st.markdown(f"**{rank}. {title}** {score}" + (f" - {md(details)}" if details else ""))
        if film.snippet:
            st.caption(md(film.snippet))


def render_outcome(outcome: RetrievalOutcome) -> None:
    st.caption(
        f"Retrieval only: {len(outcome.films)} film(s), {outcome.mode} mode, "
        f"{outcome.latency_ms / MS_PER_S:.2f}s. No LLM was called."
    )
    if not outcome.films:
        st.info(outcome.note or "No film matched.")
    with st.expander(f"Retrieved films ({len(outcome.films)})", expanded=bool(outcome.films)):
        render_films(outcome.films)


def render_answer(turn: Turn, answer: Answer) -> None:
    st.markdown(md(answer.text))
    if answer.citations:
        st.markdown("**Sources**")
        for citation in answer.citations:
            label = md(citation.label)
            st.markdown(f"- [{label}]({citation.wiki_url})" if citation.wiki_url else f"- {label}")
    with st.expander(f"Retrieved films ({len(answer.retrieved)})"):
        render_films(answer.retrieved)
    with st.expander(f"Agent steps ({len(answer.tool_calls)} tool call(s))"):
        for step, call in enumerate(answer.tool_calls, start=1):
            args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
            outcome = f"error: {call.error}" if call.error else f"{call.result_count} film(s)"
            st.markdown(f"{step}. `{call.name}({md(args)})` -> {md(outcome)}")
        if answer.blocked_tool_calls:
            st.markdown(f"{answer.blocked_tool_calls} further call(s) were refused (tool-call cap reached).")
        if answer.stopped_early:
            st.markdown("The agent stopped at its step limit before answering.")
        if not answer.tool_calls:
            st.markdown("The model answered without calling a tool.")
    tokens = answer.usage.total_tokens
    trace = f" | [LangSmith trace]({turn.trace_url})" if turn.trace_url else " | tracing off"
    st.caption(
        f"{answer.latency_ms / MS_PER_S:.1f}s | {tokens} tokens | {answer.metadata.get('chat_model', '?')} | "
        f"prompt {answer.metadata.get('prompt_version', '?')}{trace}"
    )


def render_turn(turn: Turn) -> None:
    with st.chat_message("user"):
        st.markdown(md(turn.question))
    with st.chat_message("assistant"):
        if turn.error is not None:
            render_error(turn.error)
        elif turn.answer is not None:
            render_answer(turn, turn.answer)
        elif turn.outcome is not None:
            render_outcome(turn.outcome)


def render_comparison(comparison: Comparison) -> None:
    if comparison.error is not None:
        render_error(comparison.error)
        return
    shared = set(comparison.shared_movie_ids())
    st.caption(f"{len(shared)} film(s) appear in all three modes (marked with *).")
    for column, (mode, outcome) in zip(st.columns(len(comparison.outcomes)), comparison.outcomes.items(), strict=True):
        with column:
            st.subheader(mode)
            st.caption(f"{len(outcome.films)} film(s), {outcome.latency_ms / MS_PER_S:.2f}s")
            for rank, film in enumerate(outcome.films, start=1):
                mark = " *" if film.movie_id in shared else ""
                score = f" ({film.score:.3f})" if film.score is not None else ""
                st.markdown(f"{rank}. {md(film_label(film))}{score}{mark}")


# --- page ---------------------------------------------------------------------------------------------------------


def main() -> None:
    service = get_service()
    cfg = service.settings
    st.set_page_config(page_title=cfg.ui.title, layout="wide")
    st.title(cfg.ui.title)
    params, retrieval_only = sidebar(service)
    chat_tab, compare_tab = st.tabs(["Chat", "Compare modes"])
    typed = st.chat_input("Describe a film plot...")
    turns: list[Turn] = st.session_state.setdefault(TURNS_KEY, [])

    with chat_tab:
        st.caption("Try one of these:")
        clicked: str | None = None
        for column, example in zip(st.columns(len(cfg.ui.example_questions)), cfg.ui.example_questions, strict=True):
            if column.button(example, key=f"example:{example}", use_container_width=True):
                clicked = example
        question = (typed or clicked or "").strip()
        if question:
            with st.spinner("Searching..." if retrieval_only else "Asking the agent..."):
                turns.append(service.run_turn(question, params, retrieval_only=retrieval_only))
        for turn in turns:
            render_turn(turn)

    with compare_tab:
        st.caption("The same query in dense, sparse and hybrid mode. Retrieval only: no LLM, no key needed.")
        query = st.text_input("Query", value=cfg.retrieval.demo_query, key="compare_query")
        if st.button("Compare modes", key="compare_button") and query.strip():
            with st.spinner("Searching in three modes..."):
                st.session_state[COMPARISON_KEY] = service.compare(query.strip(), params)
        if st.session_state.get(COMPARISON_KEY) is not None:
            render_comparison(st.session_state[COMPARISON_KEY])


main()
