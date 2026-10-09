"""Dense and BM25 sparse embeddings through FastEmbed.

Documents go through ``embed`` and queries through ``query_embed`` for *both* models: BM25 document and query
vectors are computed differently (documents carry term frequencies, queries only the distinct terms), and mixing
them silently degrades ranking. ``FASTEMBED_CACHE_PATH`` is honoured natively by FastEmbed (the model cache
directory); nothing here overrides it.

Models are loaded on first use so that importing this module, and building a :class:`FastEmbedder`, is free.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, cast

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import models as m

from movie_rag.config import EmbeddingsConfig

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """What ingestion (and, from PR-04, retrieval) needs from an embedding backend."""

    def count_tokens(self, text: str) -> int:
        """Number of tokens of ``text`` in the units of ``ingest.chunk_tokens``."""
        ...

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        """Dense document vectors, one per text."""
        ...

    def embed_sparse(self, texts: Sequence[str]) -> list[m.SparseVector]:
        """BM25 document vectors, one per text."""
        ...

    def embed_dense_query(self, text: str) -> list[float]:
        """Dense query vector."""
        ...

    def embed_sparse_query(self, text: str) -> m.SparseVector:
        """BM25 query vector."""
        ...


def to_sparse_vector(embedding: SparseEmbedding) -> m.SparseVector:
    """Convert a FastEmbed sparse embedding to the Qdrant model."""
    return m.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


class FastEmbedder:
    """The production :class:`Embedder`: ``BAAI/bge-small-en-v1.5`` (dense) and ``Qdrant/bm25`` (sparse)."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self._config = config
        self._dense: TextEmbedding | None = None
        self._sparse: SparseTextEmbedding | None = None

    @property
    def dense_model(self) -> TextEmbedding:
        if self._dense is None:
            logger.info("loading dense model %s", self._config.dense_model)
            self._dense = TextEmbedding(self._config.dense_model)
        return self._dense

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        if self._sparse is None:
            logger.info("loading sparse model %s", self._config.sparse_model)
            self._sparse = SparseTextEmbedding(self._config.sparse_model)
        return self._sparse

    def count_tokens(self, text: str) -> int:
        """WordPiece tokens of the dense model, excluding the ``[CLS]``/``[SEP]`` markers it adds around a text.

        FastEmbed truncates at the model's 512-token limit, so counts saturate there; the chunker only compares
        against ``ingest.chunk_tokens`` (250), well below it.
        """
        (encoding,) = cast(Any, self.dense_model.model).tokenize([text])
        return int(len(encoding.ids) - sum(encoding.special_tokens_mask))

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.dense_model.embed(list(texts))]

    def embed_sparse(self, texts: Sequence[str]) -> list[m.SparseVector]:
        return [to_sparse_vector(e) for e in self.sparse_model.embed(list(texts))]

    def embed_dense_query(self, text: str) -> list[float]:
        vector: list[float] = next(iter(self.dense_model.query_embed(text))).tolist()
        return vector

    def embed_sparse_query(self, text: str) -> m.SparseVector:
        return to_sparse_vector(next(iter(self.sparse_model.query_embed(text))))
