# Test fixtures

## `movies_sample.csv` — SYNTHETIC data

300 rows in the exact schema of the Kaggle dataset `jrobischon/wikipedia-movie-plots`:
`Release Year, Title, Origin/Ethnicity, Director, Cast, Genre, Wiki Page, Plot`.

**Every film, person, place and plot in this file is invented.** It is not real Wikipedia text and it carries no
third-party licence. No public no-auth mirror of the real dataset with a verified matching licence and schema was
available when this fixture was written (see `docs/DATASET.md`), so the fixture stands in for it.

* Titles are made up and may accidentally resemble real films. The `Wiki Page` column therefore points at
  `https://en.wikipedia.org/wiki/<Title>_(synthetic_film)` URLs, which do not exist: never treat them as citations.
* Plots are composed from hand-written story beats per genre (plus six hand-written "anchor" films, for example
  `The Forgetting Hour`, a noir about a detective who loses his memory). They are repetitive by design; they are
  good enough to exercise ingestion, retrieval, the MCP tools, the UI and the retrieval half of the smoke
  evaluation, and nothing more. Retrieval numbers measured on it say nothing about the real 35k-film dataset.

What the fixture is built to cover:

| Property | Value |
|---|---|
| Rows | 300 |
| Genres | 14 distinct plus about 13 `unknown`; mixed casing to exercise normalisation |
| Origins | 12 distinct (American, British, Bollywood, Japanese, ...) plus about 9 `Unknown` |
| Years | 1910s to 2010s |
| Short plots | 9 rows under 50 words (cleaning must drop them), including one of exactly 49 words |
| Boundary | 1 row of exactly 50 words (kept) |
| Long plots | 12 rows of about 300 words (over 250 tokens, so the chunker must split them) |

## Regenerating

`uv run python tests/fixtures/build_movies_sample.py` rewrites the CSV from a fixed seed. A unit test
(`tests/unit/test_fixture.py`) asserts the committed file is byte-identical to the generator output, so edit the
generator, never the CSV by hand.
