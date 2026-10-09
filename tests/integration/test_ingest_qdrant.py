"""End-to-end ingestion with the real FastEmbed models.

With ``QDRANT_URL`` set (CI, or ``make up`` + ``export QDRANT_URL=http://localhost:6333``) the collection lives in the
real Qdrant service and its payload indexes are checked. Without it the same tests run against the in-process
``QdrantClient(":memory:")`` so that a plain ``make check`` still exercises the real models. Models are cached under
``FASTEMBED_CACHE_PATH`` (FastEmbed reads it itself).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m

from movie_rag.config import Settings, load_settings
from movie_rag.ingest.chunk import chunk_record
from movie_rag.ingest.clean import clean_csv
from movie_rag.ingest.embed import FastEmbedder
from movie_rag.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, point_id
from movie_rag.ingest.pipeline import IngestReport, ingest_csv

pytestmark = [pytest.mark.integration, pytest.mark.filterwarnings("ignore:Payload indexes have no effect")]

QDRANT_URL = os.environ.get("QDRANT_URL")  # read at import: the per-test environment scrubber would hide it later
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "movies_sample.csv"


@pytest.fixture(scope="module")
def client() -> Iterator[QdrantClient]:
    qdrant = QdrantClient(url=QDRANT_URL, timeout=60) if QDRANT_URL else QdrantClient(":memory:")
    yield qdrant
    qdrant.close()


@pytest.fixture(scope="module")
def embedder() -> FastEmbedder:
    return FastEmbedder(load_settings(env_file=None).embeddings)


@pytest.fixture(scope="module")
def settings(client: QdrantClient) -> Iterator[Settings]:
    """Settings pointing at a throw-away collection (dropped afterwards) so the real collection is never touched."""
    collection = f"movie_plots_it_{uuid.uuid4().hex[:8]}"
    os.environ["QDRANT__COLLECTION"] = collection
    try:
        loaded = load_settings(env_file=None)
    finally:
        del os.environ["QDRANT__COLLECTION"]
    yield loaded
    if client.collection_exists(collection):
        client.delete_collection(collection)


@pytest.fixture(scope="module")
def first_report(settings: Settings, client: QdrantClient, embedder: FastEmbedder) -> IngestReport:
    return ingest_csv(settings, FIXTURE, client=client, embedder=embedder, recreate=True)


def test_ingest_fixture_counts_match_qdrant(
    settings: Settings, client: QdrantClient, first_report: IngestReport
) -> None:
    records, stats = clean_csv(FIXTURE, settings.ingest.min_plot_words)
    assert (first_report.rows_read, first_report.rows_dropped, first_report.films) == (300, 9, 291)
    assert first_report.films == len(records) == stats.rows_kept
    assert first_report.chunks_total == first_report.chunks_written > first_report.films
    assert first_report.points_in_collection == client.count(settings.qdrant.collection, exact=True).count
    assert first_report.points_in_collection == first_report.chunks_total


def test_reingest_leaves_the_point_count_unchanged(
    settings: Settings, client: QdrantClient, embedder: FastEmbedder, first_report: IngestReport
) -> None:
    again = ingest_csv(settings, FIXTURE, client=client, embedder=embedder)
    assert again.chunks_written == 0 and again.chunks_skipped == first_report.chunks_total
    assert client.count(settings.qdrant.collection, exact=True).count == first_report.points_in_collection


def test_collection_schema_has_both_named_vectors_and_payload_indexes(
    settings: Settings, client: QdrantClient, first_report: IngestReport
) -> None:
    info = client.get_collection(settings.qdrant.collection)
    assert isinstance(info.config.params.vectors, dict)
    assert info.config.params.vectors[DENSE_VECTOR].size == 384
    assert info.config.params.sparse_vectors is not None
    assert info.config.params.sparse_vectors[SPARSE_VECTOR].modifier == m.Modifier.IDF
    if QDRANT_URL:  # local mode keeps no payload indexes
        schema = {name: field.data_type for name, field in info.payload_schema.items()}
        assert schema == {
            "release_year": m.PayloadSchemaType.INTEGER,
            "origin": m.PayloadSchemaType.KEYWORD,
            "genre": m.PayloadSchemaType.KEYWORD,
            "movie_id": m.PayloadSchemaType.KEYWORD,
        }


def test_real_tokenizer_counts_wordpieces_without_special_tokens(embedder: FastEmbedder) -> None:
    assert embedder.count_tokens("Hello, world!") == 4  # hello , world !
    assert embedder.count_tokens("tokenization") > 1  # split into word pieces
    assert embedder.count_tokens("") == 0


def test_real_chunks_stay_within_the_token_budget(settings: Settings, embedder: FastEmbedder) -> None:
    records, _ = clean_csv(FIXTURE, settings.ingest.min_plot_words)
    split_films = 0
    for record in records:
        chunks = chunk_record(
            record, settings.ingest.chunk_tokens, settings.ingest.chunk_overlap, embedder.count_tokens
        )
        split_films += len(chunks) > 1
        assert all(embedder.count_tokens(c.text) <= settings.ingest.chunk_tokens for c in chunks)
        if embedder.count_tokens(record.plot) <= settings.ingest.chunk_tokens:
            assert len(chunks) == 1
    assert split_films >= 1


def test_vectors_are_real_and_queries_find_the_expected_film(
    settings: Settings, client: QdrantClient, embedder: FastEmbedder, first_report: IngestReport
) -> None:
    collection = settings.qdrant.collection
    dense = client.query_points(
        collection, query=embedder.embed_dense_query("a detective who loses his memory"), using=DENSE_VECTOR, limit=5
    )
    assert "The Forgetting Hour" in [(p.payload or {})["title"] for p in dense.points]
    sparse = client.query_points(
        collection, query=embedder.embed_sparse_query("detective memory amnesia"), using=SPARSE_VECTOR, limit=5
    )
    assert "The Forgetting Hour" in [(p.payload or {})["title"] for p in sparse.points]


def test_chunk_zero_holds_the_full_plot_for_get_movie(
    settings: Settings, client: QdrantClient, first_report: IngestReport
) -> None:
    records, _ = clean_csv(FIXTURE, settings.ingest.min_plot_words)
    record = next(r for r in records if r.title == "The Forgetting Hour")
    (point,) = client.retrieve(settings.qdrant.collection, ids=[point_id(record.movie_id, 0)])
    assert (point.payload or {})["full_plot"] == record.plot
