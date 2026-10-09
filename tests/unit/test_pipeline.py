from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from langsmith import Client, tracing_context
from qdrant_client import QdrantClient
from qdrant_client import models as m

from fakes import FakeEmbedder
from movie_rag.config import Settings
from movie_rag.errors import IndexingError
from movie_rag.ingest.chunk import chunk_record
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection, point_id
from movie_rag.ingest.pipeline import ingest_csv, resolve_source, write_records

pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")

MakeSettings = Callable[..., Settings]


@pytest.fixture
def settings(make_settings: MakeSettings) -> Settings:
    return make_settings()


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def expected_chunks(settings: Settings, csv_path: Path) -> tuple[list[MovieRecord], int]:
    """Independent re-computation of the chunk count (same chunker, one-word-one-token fake counter)."""
    records, _ = clean_csv(csv_path, settings.ingest.min_plot_words)
    total = sum(
        len(chunk_record(r, settings.ingest.chunk_tokens, settings.ingest.chunk_overlap, FakeEmbedder().count_tokens))
        for r in records
    )
    return records, total


def count(client: QdrantClient, settings: Settings) -> int:
    return client.count(settings.qdrant.collection, exact=True).count


def test_ingest_fixture_writes_every_chunk_and_reports_matching_counts(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    records, total_chunks = expected_chunks(settings, fixture_csv)
    report = ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    assert (report.rows_read, report.rows_dropped, report.films) == (300, 9, 291)
    assert report.films == len(records)
    assert total_chunks > report.films  # long plots were split
    assert report.chunks_total == report.chunks_written == total_chunks
    assert report.chunks_skipped == 0
    assert report.points_in_collection == count(client, settings) == total_chunks
    assert report.source == "movies_sample.csv" and report.collection == "movie_plots"
    assert report.elapsed_s >= 0


def test_reingest_leaves_the_point_count_unchanged_and_embeds_nothing(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    embedder = FakeEmbedder()
    first = ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    embedded_after_first = len(embedder.embedded_texts)
    second = ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    assert count(client, settings) == first.points_in_collection == second.points_in_collection
    assert (second.chunks_written, second.chunks_skipped) == (0, first.chunks_total)
    assert len(embedder.embedded_texts) == embedded_after_first  # nothing was embedded again


def test_ingest_resumes_after_deleted_points_writing_only_the_missing_ones(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    full = ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    records, _ = clean_csv(fixture_csv, settings.ingest.min_plot_words)
    victims = [point_id(r.movie_id, 0) for r in records[:5]]
    client.delete(settings.qdrant.collection, points_selector=m.PointIdsList(points=victims))
    removed = full.points_in_collection - count(client, settings)
    assert removed == 5

    embedder = FakeEmbedder()
    resumed = ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    assert resumed.chunks_written == removed
    assert resumed.chunks_skipped == full.chunks_total - removed
    assert len(embedder.embedded_texts) == removed
    assert count(client, settings) == full.points_in_collection


class ExplodingEmbedder(FakeEmbedder):
    """Fails on the n-th dense batch, like a crash half way through a long ingest."""

    def __init__(self, fail_on_batch: int) -> None:
        super().__init__()
        self.fail_on_batch = fail_on_batch

    def embed_dense(self, texts: Any) -> list[list[float]]:
        if len(self.dense_batches) + 1 == self.fail_on_batch:
            raise RuntimeError("simulated crash")
        return super().embed_dense(texts)


def test_an_interrupted_ingest_keeps_finished_batches_and_the_rerun_finishes_the_rest(
    make_settings: MakeSettings, client: QdrantClient, fixture_csv: Path
) -> None:
    settings = make_settings(INGEST__BATCH_SIZE="100")
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_csv(settings, fixture_csv, client=client, embedder=ExplodingEmbedder(fail_on_batch=2))
    assert count(client, settings) == 100  # batch 1 was durable before the crash

    embedder = FakeEmbedder()
    report = ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    assert report.chunks_skipped == 100
    assert report.chunks_written == report.chunks_total - 100
    assert len(embedder.embedded_texts) == report.chunks_total - 100
    assert count(client, settings) == report.chunks_total


def test_batches_never_exceed_the_configured_size(
    make_settings: MakeSettings, client: QdrantClient, fixture_csv: Path
) -> None:
    settings = make_settings(INGEST__BATCH_SIZE="64")
    embedder = FakeEmbedder()
    report = ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    sizes = [len(b) for b in embedder.dense_batches]
    assert max(sizes) == 64 and sum(sizes) == report.chunks_total
    assert [len(b) for b in embedder.sparse_batches] == sizes


def test_default_batch_size_is_256(settings: Settings, client: QdrantClient, fixture_csv: Path) -> None:
    embedder = FakeEmbedder()
    ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    assert len(embedder.dense_batches[0]) == 256


def test_every_embedded_text_starts_with_the_film_header(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    embedder = FakeEmbedder()
    ingest_csv(settings, fixture_csv, client=client, embedder=embedder)
    headers = {t.split("\n", 1)[0] for t in embedder.embedded_texts}
    assert all(" (" in h and h.split(" (")[1][:4].isdigit() for h in headers)
    assert all("\n" in t for t in embedder.embedded_texts)
    assert any("The Forgetting Hour (1947) | " in h for h in headers)


def test_points_carry_both_named_vectors_and_the_documented_payload(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    records, _ = clean_csv(fixture_csv, settings.ingest.min_plot_words)
    long_film = next(r for r in records if len(r.plot.split()) > 250)
    first, second = client.retrieve(
        settings.qdrant.collection,
        ids=[point_id(long_film.movie_id, 0), point_id(long_film.movie_id, 1)],
        with_payload=True,
        with_vectors=True,
    )
    assert isinstance(first.vector, dict) and set(first.vector) == {DENSE_VECTOR, SPARSE_VECTOR}
    assert len(first.vector[DENSE_VECTOR]) == 384  # type: ignore[arg-type]
    payload = first.payload or {}
    assert payload["movie_id"] == long_film.movie_id
    assert payload["title"] == long_film.title and payload["release_year"] == long_film.release_year
    assert payload["wiki_page"] == long_film.wiki_url
    assert (payload["chunk_idx"], payload["director"], payload["genre"]) == (0, long_film.director, long_film.genre)
    assert payload["full_plot"] == long_film.plot  # get_movie reads the whole plot from chunk 0
    assert payload["text"] and payload["text"] in long_film.plot
    assert payload["n_chunks"] >= 2
    assert second.payload is not None and "full_plot" not in second.payload and second.payload["chunk_idx"] == 1


def test_film_ids_are_recoverable_from_movie_id_and_index(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    records, _ = clean_csv(fixture_csv, settings.ingest.min_plot_words)
    ids = [point_id(r.movie_id, 0) for r in records]
    found = client.retrieve(settings.qdrant.collection, ids=ids, with_payload=["movie_id", "full_plot"])
    assert len(found) == len(records)
    plots = {(p.payload or {})["movie_id"]: (p.payload or {})["full_plot"] for p in found}
    assert plots == {r.movie_id: r.plot for r in records}


def test_ingest_logs_rows_chunks_and_elapsed(
    settings: Settings, client: QdrantClient, fixture_csv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    report = ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    text = caplog.text
    assert "rows_read=300 kept=291 dropped_short_plot=9" in text
    assert "ingest done: rows_read=300 rows_dropped=9 films=291" in text
    assert f"chunks_written={report.chunks_written}" in text
    assert f"points_in_collection={report.points_in_collection}" in text
    assert "elapsed_s=" in text


def test_recreate_rebuilds_the_collection(settings: Settings, client: QdrantClient, fixture_csv: Path) -> None:
    ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    embedder = FakeEmbedder()
    report = ingest_csv(settings, fixture_csv, client=client, embedder=embedder, recreate=True)
    assert report.chunks_written == report.chunks_total and report.chunks_skipped == 0
    assert len(embedder.embedded_texts) == report.chunks_total


def test_ingest_raises_when_qdrant_holds_fewer_points_than_were_produced(
    settings: Settings, client: QdrantClient, fixture_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "count", lambda *a, **k: m.CountResult(count=1))
    with pytest.raises(IndexingError, match="holds 1 points"):
        ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())


def test_write_records_rejects_an_embedder_returning_the_wrong_number_of_vectors(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    class Short(FakeEmbedder):
        def embed_sparse(self, texts: Any) -> list[m.SparseVector]:
            return super().embed_sparse(texts)[:-1]

    records, _ = clean_csv(fixture_csv, settings.ingest.min_plot_words)
    ensure_collection(client, settings)
    with pytest.raises(IndexingError, match="sparse vectors"):
        write_records(client, Short(), settings, records[:3])


def test_ingest_of_an_empty_dataset_writes_nothing(settings: Settings, client: QdrantClient, tmp_path: Path) -> None:
    header = "Release Year,Title,Origin/Ethnicity,Director,Cast,Genre,Wiki Page,Plot\n"
    path = tmp_path / "empty.csv"
    path.write_text(header, encoding="utf-8")
    report = ingest_csv(settings, path, client=client, embedder=FakeEmbedder())
    assert (report.rows_read, report.films, report.chunks_total, report.points_in_collection) == (0, 0, 0, 0)


# --- source resolution -----------------------------------------------------------------------------------


def test_resolve_source_prefers_an_explicit_csv(make_settings: MakeSettings, tmp_path: Path) -> None:
    explicit = tmp_path / "mine.csv"
    assert resolve_source(make_settings(), csv=explicit) == explicit
    assert resolve_source(make_settings(), csv=explicit, fixture=True) == explicit


def test_resolve_source_fixture_flag_wins_over_a_downloaded_dataset(
    make_settings: MakeSettings, tmp_path: Path, fixture_csv: Path
) -> None:
    settings = make_settings(DATA__RAW_DIR=str(tmp_path))
    (tmp_path / settings.data.csv_name).write_text("x", encoding="utf-8")
    assert resolve_source(settings, fixture=True) == fixture_csv


def test_resolve_source_uses_the_downloaded_dataset_when_present(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = make_settings(DATA__RAW_DIR=str(tmp_path))
    downloaded = tmp_path / settings.data.csv_name
    downloaded.write_text("x", encoding="utf-8")
    assert resolve_source(settings) == downloaded


def test_resolve_source_falls_back_to_the_fixture_with_a_warning(
    make_settings: MakeSettings, tmp_path: Path, fixture_csv: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(DATA__RAW_DIR=str(tmp_path / "missing"))
    with caplog.at_level(logging.WARNING):
        assert resolve_source(settings) == fixture_csv
    assert "make download" in caplog.text and "fixture" in caplog.text


# --- LangSmith -------------------------------------------------------------------------------------------


def traced_calls(settings: Settings, client: QdrantClient, csv_path: Path) -> MagicMock:
    langsmith_client = MagicMock(spec=Client)
    langsmith_client.tracing_queue = None
    with tracing_context(enabled=True, client=langsmith_client):
        ingest_csv(settings, csv_path, client=client, embedder=FakeEmbedder())
    return langsmith_client


def test_one_langsmith_run_per_ingest_with_metadata_and_counts(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    langsmith_client = traced_calls(settings, client, fixture_csv)
    assert langsmith_client.create_run.call_count == 1
    created = langsmith_client.create_run.call_args.kwargs
    assert created["name"] == "ingest" and created["run_type"] == "chain"
    assert created["inputs"] == {"csv": str(fixture_csv), "collection": "movie_plots", "recreate": False}

    finished = langsmith_client.update_run.call_args.kwargs
    metadata = finished["extra"]["metadata"]
    for key in ("git_sha", "prompt_version", "chat_model", "embedding_model", "config_hash"):
        assert metadata[key]
    assert metadata["source"] == "movies_sample.csv" and metadata["collection"] == "movie_plots"
    assert finished["outputs"]["films"] == 291
    assert finished["outputs"]["points_in_collection"] == finished["outputs"]["chunks_total"]


def test_traces_never_contain_configured_secrets(
    make_settings: MakeSettings, client: QdrantClient, fixture_csv: Path
) -> None:
    settings = make_settings(LANGSMITH_API_KEY="fake-langsmith-secret-0002", NEBIUS_API_KEY="fake-nebius-secret-0001")
    recorded = repr(traced_calls(settings, client, fixture_csv).mock_calls)
    assert "fake-langsmith-secret-0002" not in recorded and "fake-nebius-secret-0001" not in recorded
    assert "movies_sample.csv" in recorded  # the run was recorded, so the absence above means something


def test_ingest_runs_identically_with_tracing_off_and_no_key(
    settings: Settings, client: QdrantClient, fixture_csv: Path
) -> None:
    report = ingest_csv(settings, fixture_csv, client=client, embedder=FakeEmbedder())
    assert report.chunks_written > 0  # same code path, nothing traced, no key needed
