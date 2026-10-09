---
name: qdrant-hybrid
description: Use when creating the Qdrant collection, ingesting chunks with dense + BM25 vectors, or writing/reviewing retrieval (dense, sparse, hybrid RRF, filters, grouping).
---
# Qdrant hybrid retrieval

**Version first:** `uv pip show qdrant-client fastembed`. The installed version's docs win over this file; record
version + doc URL in the PR.

## Docs
- Hybrid queries: https://qdrant.tech/documentation/concepts/hybrid-queries/
- Query API / search: https://qdrant.tech/documentation/concepts/search/
- Grouping: https://qdrant.tech/documentation/concepts/search/#search-groups
- Filtering & payload indexes: https://qdrant.tech/documentation/concepts/filtering/ , https://qdrant.tech/documentation/concepts/indexing/
- FastEmbed sparse / BM25: https://qdrant.tech/documentation/fastembed/fastembed-splade/ , https://qdrant.github.io/fastembed/

## Working pattern
```python
from qdrant_client import QdrantClient, models as m
from fastembed import TextEmbedding, SparseTextEmbedding

client.create_collection(
    name,
    vectors_config={"dense": m.VectorParams(size=cfg.dense_dim, distance=m.Distance.COSINE)},
    sparse_vectors_config={"bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)},
)
client.create_payload_index(name, "release_year", m.PayloadSchemaType.INTEGER)
for f in ("origin", "genre", "movie_id"):
    client.create_payload_index(name, f, m.PayloadSchemaType.KEYWORD)

dense = TextEmbedding(cfg.dense_model); sparse = SparseTextEmbedding(cfg.sparse_model)
# documents -> .embed(...) ; queries -> .query_embed(...)   (BM25 query/doc embeddings differ!)

point_id = str(uuid.uuid5(NAMESPACE, f"{movie_id}:{chunk_idx}"))
# resumable: client.retrieve(name, ids=batch_ids) -> skip existing; upsert in batches of cfg.batch_size

flt = build_filter(year_from, year_to, genre, origin)   # m.Filter(must=[...]) or None
res = client.query_points_groups(
    name,
    prefetch=[
        m.Prefetch(query=dense_q, using="dense", limit=cfg.prefetch_limit, filter=flt),
        m.Prefetch(query=m.SparseVector(indices=..., values=...), using="bm25", limit=cfg.prefetch_limit, filter=flt),
    ],
    query=m.FusionQuery(fusion=m.Fusion.RRF),
    group_by="movie_id", group_size=1, limit=top_k, with_payload=True,
)
# dense / sparse mode: same function, query=<vector>, using="dense"|"bm25", query_filter=flt, no prefetch.
```
One shared `search(query, mode, top_k, filters)` builds the request for all modes; mode only changes the
query/prefetch part.

## Testing
- `QdrantClient(":memory:")` in unit tests; real service only under `@pytest.mark.integration`.
- Fake or tiny embedders in unit tests where model download would be slow; assert the request shape (prefetch carries
  the filter) and results (one row per `movie_id`, filtered years/genres respected).
- Idempotency: ingest twice → identical point count.

## Review checklist
- [ ] `query_embed` used for queries, `embed` for documents (both dense and BM25)
- [ ] `bm25` sparse vector has `Modifier.IDF`
- [ ] Filters applied **inside each prefetch** in hybrid mode (not only at top level)
- [ ] One shared function serves dense/sparse/hybrid
- [ ] uuid5 ids; re-ingest leaves count unchanged (test exists)
- [ ] Payload indexes created for release_year (int), origin, genre, movie_id (keyword)
- [ ] All limits, model names, collection name, batch size from `config.yaml`
