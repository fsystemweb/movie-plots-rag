# Test fixtures

## `movies_sample.csv` — SYNTHETIC data

300 rows in the exact schema of the Kaggle dataset `jrobischon/wikipedia-movie-plots`:
`Release Year, Title, Origin/Ethnicity, Director, Cast, Genre, Wiki Page, Plot`.

**Every film, person, place and plot in this file is invented.** It is not real Wikipedia text and it carries no
third-party licence. No public no-auth mirror of the real dataset with a verified matching licence and schema was
available when this fixture was written (see `docs/DATASET.md`), so the fixture stands in for it.

* Titles are made up and may accidentally resemble real films; person names are drawn from invented-sounding pools
  and may coincide with real people. The `Wiki Page` column therefore points at
  `https://en.wikipedia.org/wiki/<Title>_(synthetic_film)` URLs, which do not exist: never treat them as citations.
* The fixture backs the retrieval evaluation (PR-08 writes its questions from these films), so **every film has a
  premise of its own**:
  * **36 hand-written anchor films** (`story_data/anchors_a.yaml`, `story_data/anchors_b.yaml`) across 17 genres, 17 origins and
    every decade from the 1910s to the 2010s; 13 of them are long (over 250 words) and must be chunked. They include
    `The Forgetting Hour` (1947, a noir about a detective who loses his memory), the only film in the file whose
    premise is memory loss.
  * **264 generated films.** Each takes one story motif (`story_data/motifs_*.yaml`: a protagonist's situation, a central
    object and a central twist) drawn *without replacement*, and is titled after its central object (`The Rope
    Bridge`). Only connective beats and closing sentences (`story_data/shared.yaml`) may repeat between films, and the beats
    mention the film's own object, so most sentences of a plot are unique to it.
  * `tests/unit/test_fixture.py` enforces this: no shared premise sentences, at least 75% of each plot's sentences
    unique on average (and at least half for every film), nearest-neighbour TF-IDF cosine below 0.4 for every plot,
    anchors sharing under 10% of their word 4-grams with any other plot, memory loss unique to its anchor.
* Even so, the plots are short and formulaic compared with real Wikipedia summaries. They are good enough to exercise
  ingestion, retrieval, the MCP tools, the UI and the retrieval half of the smoke evaluation, and nothing more.
  Retrieval numbers measured on them say nothing about the real 35k-film dataset.

What the fixture is built to cover:

| Property | Value |
|---|---|
| Rows | 300 (291 survive cleaning) |
| Genres | 17 distinct plus 11 `unknown`; mixed casing to exercise normalisation |
| Origins | 17 distinct (American, British, Bollywood, Japanese, South_Korean, ...) plus 10 `Unknown` |
| Years | 1910s to 2010s |
| Short plots | 9 rows under 50 words (cleaning must drop them), including one of exactly 49 words |
| Boundary | 1 row of exactly 50 words (kept) |
| Long plots | 13 rows of 250 to 300 words (over 250 tokens, so the chunker must split them) |

### Suggested evaluation targets

Use the anchors for the question types that need an unmistakable gold film: fuzzy plot (for example
`The Forgetting Hour`, `Glass Harvest`, `The Ninth Bell of Varnholt`, `Atlas of Lost Birthdays`), exact entity
(any title, for example `Counterfeit Spring`, `Kimchi Wars`), and filtered questions (genre, origin or year
ranges, for example the 1949 `The Brass Band Strike` or the Russian wartime `Rails Beneath the Snow`). Generated
films are equally usable: each has a unique object and twist.

## Regenerating

`uv run python tests/fixtures/build_movies_sample.py` rewrites the CSV from a fixed seed and the YAML files in
`story_data/`. A unit test asserts the committed file is byte-identical to the generator output, so edit the generator or
the YAML, never the CSV by hand.
