from __future__ import annotations

import uuid
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m

from movie_rag.config import Settings
from movie_rag.errors import IndexingError
from movie_rag.ingest.chunk import Chunk
from movie_rag.ingest.clean import MovieRecord
from movie_rag.ingest.index import (
    DENSE_VECTOR,
    POINT_NAMESPACE,
    SPARSE_VECTOR,
    chunk_payload,
    ensure_collection,
    point_id,
)

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

MakeSettings = Callable[..., Settings]

RECORD = MovieRecord(
    movie_id="the-heist-1999-7",
    title="The Heist",
    release_year=1999,
    origin="american",
    director="Ann Lee",
    cast="A, B",
    genre="thriller",
    wiki_url="https://example.test/wiki/The_Heist",
    plot="First sentence. Second sentence.",
)


def test_point_id_is_a_deterministic_uuid5_of_movie_and_chunk() -> None:
    expected = str(uuid.uuid5(POINT_NAMESPACE, "the-heist-1999-7:0"))
    assert point_id("the-heist-1999-7", 0) == expected
    assert point_id("the-heist-1999-7", 0) == expected
    assert uuid.UUID(expected).version == 5


def test_point_ids_differ_per_chunk_and_per_film() -> None:
    ids = {point_id("a-1-0", 0), point_id("a-1-0", 1), point_id("b-1-1", 0), point_id("b-1-1", 1)}
    assert len(ids) == 4


def test_point_id_is_pinned_so_that_the_namespace_never_changes() -> None:
    """Changing the namespace would orphan every stored point."""
    assert point_id("the-heist-1999-7", 0) == "9f839a85-8154-5578-9345-90fa1bdd3d60"


def test_chunk_zero_payload_carries_metadata_text_and_the_full_plot() -> None:
    chunk = Chunk(movie_id=RECORD.movie_id, chunk_idx=0, n_chunks=2, header="h", text="First sentence.")
    assert chunk_payload(RECORD, chunk) == {
        "movie_id": "the-heist-1999-7",
        "title": "The Heist",
        "release_year": 1999,
        "director": "Ann Lee",
        "cast": "A, B",
        "genre": "thriller",
        "origin": "american",
        "wiki_page": "https://example.test/wiki/The_Heist",
        "chunk_idx": 0,
        "n_chunks": 2,
        "text": "First sentence.",
        "full_plot": "First sentence. Second sentence.",
    }


def test_later_chunks_do_not_repeat_the_full_plot() -> None:
    chunk = Chunk(movie_id=RECORD.movie_id, chunk_idx=1, n_chunks=2, header="h", text="Second sentence.")
    payload = chunk_payload(RECORD, chunk)
    assert "full_plot" not in payload
    assert payload["chunk_idx"] == 1 and payload["text"] == "Second sentence."
    assert payload["title"] == "The Heist"


def test_payload_keeps_unknown_metadata_as_null() -> None:
    sparse_record = RECORD.model_copy(update={"genre": None, "origin": None, "director": None, "cast": None})
    chunk = Chunk(movie_id=RECORD.movie_id, chunk_idx=0, n_chunks=1, header="h", text="t")
    payload = chunk_payload(sparse_record, chunk)
    assert payload["genre"] is None and payload["origin"] is None
    assert payload["director"] is None and payload["cast"] is None


def test_ensure_collection_creates_named_vectors_with_idf(make_settings: MakeSettings) -> None:
    settings, client = make_settings(), QdrantClient(":memory:")
    assert ensure_collection(client, settings) is True
    params = client.get_collection(settings.qdrant.collection).config.params
    assert isinstance(params.vectors, dict)
    assert set(params.vectors) == {DENSE_VECTOR}
    assert params.vectors[DENSE_VECTOR].size == 384
    assert params.vectors[DENSE_VECTOR].distance == m.Distance.COSINE
    assert params.sparse_vectors is not None
    assert params.sparse_vectors[SPARSE_VECTOR].modifier == m.Modifier.IDF


def test_ensure_collection_creates_the_four_payload_indexes(make_settings: MakeSettings) -> None:
    settings = make_settings()
    client = MagicMock(spec=QdrantClient)
    client.collection_exists.return_value = False
    ensure_collection(client, settings)
    created = {call.args[1]: call.args[2] for call in client.create_payload_index.call_args_list}
    assert created == {
        "release_year": m.PayloadSchemaType.INTEGER,
        "origin": m.PayloadSchemaType.KEYWORD,
        "genre": m.PayloadSchemaType.KEYWORD,
        "movie_id": m.PayloadSchemaType.KEYWORD,
    }
    assert all(call.args[0] == settings.qdrant.collection for call in client.create_payload_index.call_args_list)


def test_ensure_collection_uses_configured_name_and_dimension(make_settings: MakeSettings) -> None:
    settings = make_settings(QDRANT__COLLECTION="other_plots", EMBEDDINGS__DENSE_DIM="8")
    client = QdrantClient(":memory:")
    ensure_collection(client, settings)
    assert client.collection_exists("other_plots")
    params = client.get_collection("other_plots").config.params
    assert isinstance(params.vectors, dict) and params.vectors[DENSE_VECTOR].size == 8


def test_ensure_collection_is_idempotent(make_settings: MakeSettings) -> None:
    settings, client = make_settings(), QdrantClient(":memory:")
    assert ensure_collection(client, settings) is True
    client.upsert(
        settings.qdrant.collection,
        points=[m.PointStruct(id=point_id("a", 0), vector={DENSE_VECTOR: [0.1] * 384}, payload={})],
    )
    assert ensure_collection(client, settings) is False
    assert client.count(settings.qdrant.collection).count == 1


def test_existing_collection_gets_missing_payload_indexes_repaired(make_settings: MakeSettings) -> None:
    settings = make_settings()
    client = MagicMock(spec=QdrantClient)
    client.collection_exists.return_value = True
    client.get_collection.return_value = MagicMock(
        config=MagicMock(params=MagicMock(vectors={DENSE_VECTOR: m.VectorParams(size=384, distance=m.Distance.COSINE)}))
    )
    assert ensure_collection(client, settings) is False
    assert client.create_payload_index.call_count == 4


def test_dimension_mismatch_raises_an_actionable_error(make_settings: MakeSettings) -> None:
    client = QdrantClient(":memory:")
    ensure_collection(client, make_settings(EMBEDDINGS__DENSE_DIM="8"))
    with pytest.raises(IndexingError, match="--recreate"):
        ensure_collection(client, make_settings(EMBEDDINGS__DENSE_DIM="384"))


def test_collection_without_a_dense_vector_is_rejected(make_settings: MakeSettings) -> None:
    settings, client = make_settings(), QdrantClient(":memory:")
    client.create_collection(
        settings.qdrant.collection, vectors_config=m.VectorParams(size=384, distance=m.Distance.COSINE)
    )
    with pytest.raises(IndexingError, match="dense size None"):
        ensure_collection(client, settings)


def test_recreate_drops_existing_points(make_settings: MakeSettings) -> None:
    settings, client = make_settings(), QdrantClient(":memory:")
    ensure_collection(client, settings)
    client.upsert(
        settings.qdrant.collection,
        points=[m.PointStruct(id=point_id("a", 0), vector={DENSE_VECTOR: [0.1] * 384}, payload={})],
    )
    assert ensure_collection(client, settings, recreate=True) is True
    assert client.count(settings.qdrant.collection).count == 0


def test_recreate_on_a_missing_collection_just_creates_it(make_settings: MakeSettings) -> None:
    settings, client = make_settings(), QdrantClient(":memory:")
    assert ensure_collection(client, settings, recreate=True) is True
    assert client.collection_exists(settings.qdrant.collection)
