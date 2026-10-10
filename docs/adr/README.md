# Architecture decision records

Each record states the context, the decision, the alternatives that were considered, the consequences and a status.
Values quoted from `config.yaml` are the defaults at the time of writing; the file is the source of truth.

| ADR | Decision | Status |
|---|---|---|
| [001](001-qdrant.md) | Qdrant (Docker) as the single vector and keyword store | Accepted |
| [002](002-rrf-and-parameters.md) | Server-side RRF with a fixed constant, 50-chunk prefetches, grouping by film | Accepted (revisit with qdrant-client 1.16) |
| [003](003-chunking.md) | Sentence-aware chunks of about 250 tokens, 40 overlap, header on every chunk, full plot on chunk 0 | Accepted |
| [004](004-local-embeddings.md) | Local FastEmbed models (`bge-small-en-v1.5` dense, `Qdrant/bm25` sparse), no embedding API | Accepted |
