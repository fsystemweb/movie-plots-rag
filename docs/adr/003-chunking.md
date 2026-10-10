# ADR 003: Sentence-aware chunks, header on every chunk, full plot on chunk 0

* Status: Accepted
* Date: 2026-10-10

## Context

A plot summary is the unit users ask about, but dense embeddings blur long text and the dense model has a 512-token
input limit. Retrieval must still return one film per row with a citable snippet, and `get_movie` must return the whole
plot. Code: `src/movie_rag/ingest/chunk.py`, `index.py`.

## Decision

* **Size.** `ingest.chunk_tokens: 250` tokens of plot text per chunk and `ingest.chunk_overlap: 40` tokens of overlap
  (spec values). A *token* is a WordPiece token of the dense model itself, without `[CLS]`/`[SEP]`
  (`FastEmbedder.count_tokens`), so the budget is in the units the embedding model sees. A plot at or under the budget
  stays whole.
* **Boundaries.** Longer plots are cut between sentences. The overlap is the whole trailing sentences of the previous
  chunk that fit in 40 tokens (possibly none); every chunk adds at least one new sentence, so chunking terminates. A
  single sentence over 250 tokens is split at word boundaries.
* **Header.** `Title (Year) | Genre | Director` is prepended to the text of **every** chunk before both the dense and
  the BM25 embedding, on top of the 250-token budget (about 15 tokens, far below 512). The stored `text` payload is the
  plot only, so snippets are clean; the header is rebuilt from the payload.
* **Points.** One point per chunk, id `uuid5(movie_id:chunk_idx)`, batches of `ingest.batch_size: 256`. Chunk 0 also
  carries `full_plot`; `get_movie` reads it with one `retrieve` by id (no scroll, no filter).

Measured on the committed fixture (`tests/fixtures/movies_sample.csv`, 291 films survive cleaning): plots are 63 to 345
tokens (median 122), so 278 films stay whole and 13 become two chunks, giving the 304 indexed chunks the evaluation
reports; chunk sizes are 53 to 250 tokens (median 122). The real dataset's length distribution has **not** been
measured (no download yet), so the chunk count and the share of multi-chunk films on 35,000 films are unknown.

## Alternatives considered

| Option | Why not |
|---|---|
| Whole plot as one vector | Long plots exceed or saturate the 512-token limit and blur their parts; fine for short plots, which is why they stay whole. |
| Fixed windows, no sentence logic | Cuts mid-sentence, hurts both the embedding and the snippet shown to users. |
| Recursive or semantic splitter library | An extra dependency and a model or heuristics to evaluate; the regex below is small and tested. Revisit if the full dataset shows bad chunks. |
| `full_plot` on every chunk | Multiplies payload size by the chunk count for nothing. |
| `full_plot` in a second collection | One more thing to create, fill and keep in sync. |
| Header only on the dense text | BM25 would lose title, genre and director terms, which users do type ("a 1990s thriller"). |

## Consequences

* A film's chunks compete in the prefetch lists, and grouping by `movie_id` keeps the best (ADR 002). The shown
  snippet is the best chunk, not necessarily the start of the plot.
* **Known splitter limits** (backlog `[PR-03]`, `[PR-03 QA m1/m2/m4]`): sentence splitting is a regex that splits before
  an uppercase letter, digit or quote and knows a short abbreviation list. It misses boundaries such as "say no. The",
  "Plan B. The" and "U.S. There", and never splits before a lowercase-initial sentence. The overlap is 0 when the
  last sentence alone is longer than 40 tokens. Token counting uses an internal FastEmbed call with no upper version
  pin on `fastembed`. Counts saturate at 512, which only matters for an over-long single sentence (those are split anyway).
* Changing any chunking parameter or the data does not refresh existing points (their ids do not change): run
  `make ingest RECREATE=1` (backlog: a content-hash payload field would allow selective refresh).
* The plot is stored twice on chunk 0 (`text` and `full_plot`); accepted for the single-call `get_movie`.
* The cast is stored as payload but is not in the chunk text, so cast questions are not answerable by retrieval
  (backlog `[PR-08]`).

## References

* `docs/BACKLOG.md` entries `[PR-03]`, `[PR-03 QA m1..m4]`, `[PR-08]`; `tests/fixtures/README.md`.
