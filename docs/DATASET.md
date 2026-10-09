# Dataset

## Source

[Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) on Kaggle
(`jrobischon/wikipedia-movie-plots`): about 35,000 films, English only, under 2 GB (roughly 80 MB as CSV).

| Column | Meaning | Cleaned field |
|---|---|---|
| `Release Year` | four-digit year | `release_year` (int) |
| `Title` | film title | `title` |
| `Origin/Ethnicity` | national or regional cinema, for example `American`, `Bollywood`, `South_Korean` | `origin` (lowercase; `unknown` becomes null) |
| `Director` | director name(s) | `director` (`unknown` becomes null) |
| `Cast` | principal cast | `cast` (`unknown` becomes null) |
| `Genre` | genre label(s) | `genre` (lowercase; `unknown` becomes null) |
| `Wiki Page` | Wikipedia article URL | `wiki_url` |
| `Plot` | plot summary from the article | `plot` |

## Licence and attribution

Plot text comes from Wikipedia and is licensed CC BY-SA (the Kaggle page and some catalogues list variants of the
Creative Commons share-alike licence; check the Kaggle page for the terms that apply to your use). The project keeps
the `Wiki Page` link for every film and every answer cites `Title (Year)` with that link, which is how attribution is
preserved. Do not commit the raw dataset: `data/` is gitignored and only the synthetic fixture lives in git.

## Getting the real data

```bash
# 1. Create a Kaggle API token (https://www.kaggle.com/settings, "API"), then put it in .env:
#      KAGGLE_USERNAME=...   KAGGLE_KEY=...
# 2. Download:
make download            # writes data/raw/wiki_movie_plots_deduped.csv ; FORCE=1 to download again
```

Without credentials `make download` prints what to set (`set KAGGLE_USERNAME and KAGGLE_KEY — see
docs/CREDENTIALS.md`) and exits with code 2, with no stack trace. `make doctor` reports whether the credentials are
present and valid.

**No-auth mirror: none used.** KICKOFF section 2 allows a public no-auth mirror (for example on Hugging Face) if its
licence and columns match. A search found no mirror of this exact dataset with a verified licence, and catalogue
entries disagree on the licence (CC BY-SA versus CC BY-NC-SA), so none is adopted. Development, CI and the demo use
the synthetic fixture below.

## Fixture

[`tests/fixtures/movies_sample.csv`](../tests/fixtures/README.md) is 300 invented films in the exact Kaggle schema:
several genres, origins and decades, 9 plots under 50 words (including one of exactly 49) that cleaning drops, one of
exactly 50 words that it keeps, and 12 plots of about 300 words that the chunker must split. It is generated from a
fixed seed by `tests/fixtures/build_movies_sample.py`. Retrieval numbers measured on it prove the pipeline works; they
say nothing about quality on the real 35k films.

## Cleaning rules

Implemented in [`src/movie_rag/ingest/clean.py`](../src/movie_rag/ingest/clean.py), tested on hand-built rows in
`tests/unit/test_clean.py`.

1. **Drop short plots.** A plot with fewer than `ingest.min_plot_words` (50, in `config.yaml`) whitespace-separated
   words is dropped; 50 exactly is kept. Whitespace runs in plots are collapsed first.
2. **Drop invalid rows.** A row with an empty title or a year that is not four digits is dropped (counted separately).
3. **Normalise labels.** `genre` and `origin` are trimmed, lowercased and whitespace-collapsed; `unknown` (any case)
   and empty values become null. `director`, `cast` and `wiki_url` get the same `unknown`/empty to null treatment
   but keep their case.
4. **Stable `movie_id`.** `slug(title)-year-row_index`, for example `the-matrix-1999-4181`. The slug is ASCII,
   lowercase and hyphen-separated. `row_index` is the 0-based position of the row in the raw file, counted before any
   row is dropped, so ids do not shift when cleaning thresholds change and two films with the same title and year
   never collide. Running the cleaning again on the same file always yields the same ids.
5. **Seeded sampling.** Anything that samples films uses `ingest.random_seed` (42) through `sample_records`.

Cleaning logs `rows_read`, `kept`, `dropped_short_plot` and `dropped_invalid` on every run.
