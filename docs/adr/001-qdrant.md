# ADR 001: Qdrant as the single vector and keyword store

* Status: Accepted
* Date: 2026-10-10

## Context

The product needs three retrieval modes over the same chunks (dense, sparse BM25, and a fusion of both), metadata
filters on year, genre and origin, one result per film, and an index that tests and CI can use without credentials.
The stack was fixed by the product spec (`KICKOFF.md` section 1); this record explains why the choice fits and what it
costs, it does not claim a benchmark of alternatives (none was run).

## Decision

Use Qdrant, run as the Docker image `qdrant/qdrant:v1.15.4` (`docker-compose.yml`; the same tag is pinned in CI), one
collection (`qdrant.collection: movie_plots`) with:

* a named dense vector `dense` (`embeddings.dense_dim: 384`, cosine) and a named sparse vector `bm25` with
  `Modifier.IDF`, so both signals live on the same point (`src/movie_rag/ingest/index.py`);
* payload indexes on `release_year` (integer) and `origin`, `genre`, `movie_id` (keyword), so filters and the
  `list_filters` facets are index lookups, not scans;
* deterministic point ids `uuid5(movie_id:chunk_idx)`, so a re-ingest upserts the same points.

Hybrid search is one Query API call: two prefetches fused by the server, grouped by `movie_id` (ADR 002). Dense and
sparse mode use the same call with one named vector, so there is a single retrieval code path
(`retrieval/search.py: build_request`).

## Alternatives considered

| Option | Why not (for this project) |
|---|---|
| Postgres + pgvector | Vector search is native, BM25 is not (full-text ranking differs from BM25); hybrid would be two queries and client-side fusion, and grouping per film is more SQL to own. |
| Elasticsearch / OpenSearch | Excellent BM25 and now kNN with RRF, but a JVM cluster to run for 300 to 35,000 films; heavier than the demo needs. |
| Chroma, FAISS, LanceDB | Simple to embed, but filtered hybrid with fusion, payload indexes and facets would be code we write and test ourselves. |
| Weaviate, Milvus | Capable hybrid engines; no advantage over Qdrant for this size, and the spec fixed Qdrant. |

## Consequences

* One service does dense, sparse, fusion, filters, grouping and facets; the application has no fusion code.
* Tests use `QdrantClient(":memory:")`, but qdrant-client 1.15.1's local engine **ignores the prefetch queries and
  filters of grouped hybrid queries** (observed in PR-04 and PR-08). Hybrid behaviour is therefore only verified
  against the real service: `tests/integration/test_retrieval_qdrant.py` runs when `QDRANT_URL` is set (CI), and the
  evaluator refuses the in-memory engine for hybrid numbers. Unit tests cover hybrid by request shape.
* The compose and CI tag is `v1.15.4` and `uv.lock` pins qdrant-client 1.15.1. Parametric RRF needs only a newer
  client (1.16 for `RrfQuery`): the 1.15.4 server already accepts `rrf.k` over REST (ADR 002). Weighted RRF is
  documented as of 1.17 and would need both bumped.
* Qdrant 1.15.4 also pushes a top-level filter down into prefetches. The code attaches the filter to each prefetch
  explicitly instead of relying on that undocumented behaviour.
* `get_movie` by title is an unindexed payload match (fine for the fixture, untimed on 35,000 films; backlog).

## References

* Hybrid queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
* Payload indexes: https://qdrant.tech/documentation/concepts/indexing/
* `docs/BACKLOG.md` entries `[PR-04]` and `[PR-05]`.
