"""The Qdrant collection: schema, payload indexes, deterministic point ids and payloads.

Payload design (what later PRs rely on)
---------------------------------------
Every point carries the film metadata (``movie_id``, ``title``, ``release_year``, ``director``, ``cast``, ``genre``,
``origin``, ``wiki_page``), its position (``chunk_idx``, ``n_chunks``) and the passage ``text`` (plot only, without the
header). **Chunk 0 additionally carries ``full_plot``**: ``get_movie`` fetches ``point_id(movie_id, 0)`` directly (one
``retrieve`` by id, no scroll or filter) and reads the whole plot from it. Storing the plot on every chunk would
multiply the payload size by the chunk count for no benefit; a separate collection would add a second thing to create,
fill and keep in sync.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as m

from movie_rag.config import Settings
from movie_rag.errors import IndexingError
from movie_rag.ingest.chunk import Chunk
from movie_rag.ingest.clean import MovieRecord

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
# Fixed forever: changing it would change every point id. Not a tunable.
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/fsystemweb/movie-plots-rag/points")
KEYWORD_INDEXES = ("origin", "genre", "movie_id")
INTEGER_INDEXES = ("release_year",)


def point_id(movie_id: str, chunk_idx: int) -> str:
    """Deterministic point id: ``uuid5(movie_id:chunk_idx)``. Re-ingesting upserts the same points."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{movie_id}:{chunk_idx}"))


def chunk_payload(record: MovieRecord, chunk: Chunk) -> dict[str, Any]:
    """The payload stored with one chunk (see the module docstring)."""
    payload: dict[str, Any] = {
        "movie_id": record.movie_id,
        "title": record.title,
        "release_year": record.release_year,
        "director": record.director,
        "cast": record.cast,
        "genre": record.genre,
        "origin": record.origin,
        "wiki_page": record.wiki_url,
        "chunk_idx": chunk.chunk_idx,
        "n_chunks": chunk.n_chunks,
        "text": chunk.text,
    }
    if chunk.chunk_idx == 0:
        payload["full_plot"] = record.plot
    return payload


def _dense_size(info: m.CollectionInfo) -> int | None:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict) and DENSE_VECTOR in vectors:
        return vectors[DENSE_VECTOR].size
    return None


def _create_payload_indexes(client: QdrantClient, collection: str) -> None:
    for field in INTEGER_INDEXES:
        client.create_payload_index(collection, field, m.PayloadSchemaType.INTEGER)
    for field in KEYWORD_INDEXES:
        client.create_payload_index(collection, field, m.PayloadSchemaType.KEYWORD)


def ensure_collection(client: QdrantClient, settings: Settings, *, recreate: bool = False) -> bool:
    """Create the collection (named vectors ``dense`` and ``bm25`` + IDF, payload indexes) if it does not exist.

    ``recreate`` drops an existing collection first. Returns ``True`` when the collection was created. An existing
    collection whose dense size differs from ``embeddings.dense_dim`` raises :class:`IndexingError`: mixing
    dimensions would corrupt the index, so the fix (``--recreate``) is explicit.
    """
    name = settings.qdrant.collection
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        size = _dense_size(client.get_collection(name))
        if size != settings.embeddings.dense_dim:
            raise IndexingError(
                f"collection {name!r} has dense size {size}, config says {settings.embeddings.dense_dim}; "
                "re-run with --recreate (make ingest RECREATE=1) to rebuild it"
            )
        _create_payload_indexes(client, name)  # idempotent: repairs a collection created without them
        return False
    client.create_collection(
        name,
        vectors_config={
            DENSE_VECTOR: m.VectorParams(size=settings.embeddings.dense_dim, distance=m.Distance.COSINE),
        },
        sparse_vectors_config={SPARSE_VECTOR: m.SparseVectorParams(modifier=m.Modifier.IDF)},
    )
    _create_payload_indexes(client, name)
    return True
