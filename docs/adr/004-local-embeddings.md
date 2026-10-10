# ADR 004: Local embeddings with FastEmbed, no embedding API

* Status: Accepted
* Date: 2026-10-10

## Context

Both retrieval signals need a model: a dense encoder and a BM25 term-weighting scheme. The project must build, test and
run its retrieval path with no credentials (`CLAUDE.md`, credentials policy); the only remote service in the design is
Nebius Token Factory for the chat model and the evaluation judge.

## Decision

Embed locally with FastEmbed (ONNX runtime, CPU), both for documents at ingest time and for queries at search time:

* dense: `embeddings.dense_model: BAAI/bge-small-en-v1.5`, `embeddings.dense_dim: 384`, cosine;
* sparse: `embeddings.sparse_model: Qdrant/bm25`, stored with `Modifier.IDF` so Qdrant applies the inverse document
  frequency on the server.

Documents use `embed` and queries use `query_embed` for **both** models: BM25 document vectors carry term frequencies
while query vectors carry the distinct terms only, and mixing them silently degrades ranking
(`ingest/embed.py`). Models load on first use, so importing the package is free. The cache directory is FastEmbed's
(`FASTEMBED_CACHE_PATH`).

Embedding API calls are not used anywhere. RAGAS's response-relevancy metric also uses the local dense model, so the
evaluation needs one credential (the judge), not two.

## Alternatives considered

| Option | Why not |
|---|---|
| Nebius (or OpenAI-compatible) embedding endpoint | Needs a key for ingestion and for every query, which breaks the no-credentials build, adds cost and latency to each search and makes results depend on a remote model version. |
| `sentence-transformers` / PyTorch | Same models, a much heavier install; FastEmbed's ONNX models are enough and are what Qdrant documents. |
| A larger local encoder (`bge-base`, `bge-large`) | Higher quality is plausible but slower on CPU and 768 or 1024 dimensions; not evaluated, and the fixture cannot tell them apart. |
| SPLADE (learned sparse) instead of BM25 | Also available in FastEmbed; heavier at ingest and not needed to demonstrate the hybrid design. Not evaluated. |

## Consequences

* Works offline after the first model download (about 64 MB in FastEmbed's cache here: it serves `bge-small-en-v1.5` as the quantized ONNX export `Qdrant/bge-small-en-v1.5-onnx-Q`, plus a few KB of BM25 files). The ingest of
  the 304-chunk fixture took about 27 s on CPU including model load (PR-03); the full 35,000-film ingest is untimed
  (backlog: FastEmbed `parallel` if too slow).
* The `mcp-server` image downloads the models on its first query into the `fastembed_models` volume, so the very first
  query needs network (backlog `[PR-05]`).
* The model, the dimension and the point ids are coupled: changing `embeddings.*` requires `make ingest RECREATE=1`
  (the ingest refuses a collection whose dense size differs from `dense_dim`).
* `bge-small-en-v1.5` is English and limited to 512 input tokens, which is why chunks are 250 tokens (ADR 003). The
  dataset is English only.
* Retrieval quality is bounded by a small model. Whether a bigger one or an API model would help on real plots is
  unmeasured; it requires the full dataset and a question set for it.

## References

* FastEmbed: https://qdrant.github.io/fastembed/ ; BM25 and sparse models: https://qdrant.tech/documentation/fastembed/fastembed-splade/
* Installed: fastembed 0.9.0, qdrant-client 1.15.1.
