# Demo script (about 2 minutes)

A walk through the Streamlit test page. It needs no credentials: the page opens in "Retrieval only" mode, which ranks
films without calling a language model. The films come from the synthetic fixture (300 invented plots, 291 kept), so
the "Wikipedia" links in results do not exist.

All questions below were run against a local Qdrant with `make serve` on this branch. Top results are stable; scores
and the order of films further down the list can vary slightly between runs (hybrid ties, see `docs/EVALUATION.md`).

## Before the audience arrives (once, about a minute)

```bash
make setup          # dependencies (skip if already done)
make demo           # starts Qdrant, indexes the fixture, prints one sample query in all three modes
make serve          # terminal 1: MCP server on http://127.0.0.1:8000/mcp
make ui             # terminal 2: the page on http://localhost:8501
```

`make demo` can be replaced by `make up && make ingest FIXTURE=1`. The first run downloads about 64 MB of embedding
models. If the page says the MCP server is unreachable, it names `mcp.url` and `make serve`.

## The script

| Time | Do this | Say / point at |
|---|---|---|
| 0:00 | Open the page. Point at the sidebar: "Retrieval only (no LLM, no key needed)" is on. | "No account and no key. The search is local: meaning vectors and exact-word vectors in Qdrant." |
| 0:15 | Click the first example button: **a hotel telephone operator overhears a call arranging a death at midnight**. | The top film is *The Overheard Midnight Call* (1968), score 1.00. Open **Retrieved films**: title and year linked to the source page, score, genre, origin, director and a plot snippet. "The score is only comparable within one mode." |
| 0:40 | Type a loose paraphrase: **A private eye with amnesia fears the man he was hired to track down is himself**. | *The Forgetting Hour* (1947) is first. "No title word is in the question: the match is by story." |
| 1:00 | In the sidebar set the year range to **1990-1999**, then ask **a murder seen through a glass wall by a window cleaner**. | *The Window-Cleaner's Witness* (1997) is first, and every result is from the 1990s. "Filters are applied inside the search, not afterwards." Optionally pick Origin `telugu` and ask **villagers in a dry region hijack a fuel truck to expose a rich landowner who is draining their well**: exactly one film, *Kerosene Summer* (1975). |
| 1:15 | Open the **Compare modes** tab, type **Which film did Gleb Ostrovsky-Marin direct?**, click **Compare modes**. | Dense (meaning only) misses it: its top result is *The Last Projectionist*. Sparse and hybrid put *Rails Beneath the Snow* (1963) first. "A name has no meaning to a vector model but is an exact word to BM25. Hybrid keeps both strengths." |
| 1:35 | Compare the default query again, or type **a talking dinosaur runs for president of Mars**. | "No such film exists, yet retrieval returns its nearest neighbours with low scores. Deciding to say *I don't know* is the agent's job, which needs a key." |
| 1:50 | In the sidebar switch **Retrieval only** off and ask anything. | The page shows `set NEBIUS_API_KEY — see docs/CREDENTIALS.md` and how to switch back. "Agent mode is built and tested with a scripted model; the live run is pending credentials." |

Closing line (2:00): the evaluation (`reports/EVAL_RESULTS.md`) has retrieval numbers for all three modes on this
fixture and "pending credentials" for the language-model metrics. The fixture favours exact words, so it cannot rank
the modes for real data; see [`SOLUTION_OVERVIEW.md`](SOLUTION_OVERVIEW.md).

## Terminal alternative

```bash
make demo Q="A private eye with amnesia fears the man he was hired to track down is himself"
uv run python -m movie_rag.retrieval "villagers hijack a fuel truck" --mode hybrid --origin telugu --top-k 3
```

## Once a key exists

Add `NEBIUS_API_KEY` to `.env` ([`CREDENTIALS.md`](CREDENTIALS.md)), run `make doctor`, then restart `make ui`. The
retrieval-only switch now starts off. Show:

1. The same first question in agent mode: a short answer that names *The Overheard Midnight Call (1968)* with its
   link, then **Retrieved films** and **Agent steps** (the tool calls, at most four per question).
2. **a talking dinosaur runs for president of Mars**: the agent should say that nothing in the library fits and cite no
   film. Treat this as the thing to check, not a guarantee: see "Before the first live run" in `CREDENTIALS.md`.
3. Latency and token counts under the answer; with `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`, the trace link.
4. `make ask Q="..."` for the same agent without the page (needs `make serve`).
5. The full results: `make download && make ingest && make eval MODE=dense && make eval MODE=sparse && make eval MODE=hybrid && make report`.

Stop everything with `Ctrl+C` in both terminals and `make down` (the index stays in the `qdrant_storage` volume).
