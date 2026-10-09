"""Ingestion: cleaned films -> chunks -> dense + BM25 vectors -> Qdrant points.

Idempotent and resumable: point ids are deterministic, and before embedding a batch the ids that already exist in the
collection are skipped, so re-running after an interruption (or on the same data) only does the missing work and
leaves the point count unchanged. A stale point whose text changed keeps its id and is therefore *not* refreshed; use
``recreate`` when the chunking parameters or the data changed.

One LangSmith run is recorded per :func:`ingest_csv` call when tracing is on; without a key it is a no-op.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client import models as m

from movie_rag.config import Settings
from movie_rag.errors import IndexingError
from movie_rag.ingest.chunk import Chunk, chunk_record
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.ingest.download import raw_csv_path
from movie_rag.ingest.embed import Embedder
from movie_rag.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, chunk_payload, ensure_collection, point_id
from movie_rag.observability import run_metadata

logger = logging.getLogger(__name__)


class IngestReport(BaseModel):
    """Counts for one ingest run. ``points_in_collection`` is read back from Qdrant after the last upsert."""

    source: str
    collection: str
    rows_read: int
    rows_dropped: int
    films: int
    chunks_total: int
    chunks_written: int
    chunks_skipped: int
    points_in_collection: int
    elapsed_s: float


def resolve_source(settings: Settings, *, csv: Path | None = None, fixture: bool = False) -> Path:
    """The CSV to ingest: ``csv``, else the fixture if ``fixture``, else the downloaded dataset, else the fixture.

    The fallback to the fixture (with a warning) is what lets ``make ingest`` work without Kaggle credentials.
    """
    if csv is not None:
        return csv
    fixture_path = settings.data.resolve(settings.data.fixture_path)
    if fixture:
        return fixture_path
    downloaded = raw_csv_path(settings)
    if downloaded.is_file():
        return downloaded
    logger.warning("no downloaded dataset at %s (run `make download`): using the fixture %s", downloaded, fixture_path)
    return fixture_path


def _batched[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _iter_chunks(
    records: Iterable[MovieRecord], settings: Settings, embedder: Embedder
) -> Iterator[tuple[MovieRecord, Chunk]]:
    for record in records:
        for chunk in chunk_record(
            record, settings.ingest.chunk_tokens, settings.ingest.chunk_overlap, embedder.count_tokens
        ):
            yield record, chunk


def write_records(
    client: QdrantClient, embedder: Embedder, settings: Settings, records: Iterable[MovieRecord]
) -> tuple[int, int, int]:
    """Chunk, embed and upsert ``records``. Returns ``(chunks_total, chunks_written, chunks_skipped)``.

    The collection must already exist (see :func:`ensure_collection`).
    """
    collection = settings.qdrant.collection
    total = written = skipped = 0
    pairs = _iter_chunks(records, settings, embedder)
    for batch in _batched(pairs, settings.ingest.batch_size):
        ids = [point_id(chunk.movie_id, chunk.chunk_idx) for _, chunk in batch]
        existing = {str(p.id) for p in client.retrieve(collection, ids=ids, with_payload=False, with_vectors=False)}
        todo = [(pid, record, chunk) for pid, (record, chunk) in zip(ids, batch, strict=True) if pid not in existing]
        total += len(batch)
        skipped += len(batch) - len(todo)
        if todo:
            texts = [chunk.embed_text for _, _, chunk in todo]
            dense = embedder.embed_dense(texts)
            sparse = embedder.embed_sparse(texts)
            if not len(dense) == len(sparse) == len(todo):
                raise IndexingError(
                    f"embedder returned {len(dense)} dense / {len(sparse)} sparse vectors for {len(todo)} texts"
                )
            client.upsert(
                collection,
                points=[
                    m.PointStruct(
                        id=pid,
                        vector={DENSE_VECTOR: d, SPARSE_VECTOR: s},
                        payload=chunk_payload(record, chunk),
                    )
                    for (pid, record, chunk), d, s in zip(todo, dense, sparse, strict=True)
                ],
                wait=True,
            )
            written += len(todo)
        logger.info("ingest progress: %d chunks seen, %d written, %d skipped", total, written, skipped)
    return total, written, skipped


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Only plain, non-secret facts go into the trace (never the settings object, client or embedder)."""
    settings = inputs["settings"]
    return {
        "csv": str(inputs["csv_path"]),
        "collection": settings.qdrant.collection,
        "recreate": inputs.get("recreate", False),
    }


@traceable(run_type="chain", name="ingest", process_inputs=_trace_inputs)
def ingest_csv(
    settings: Settings,
    csv_path: Path,
    *,
    client: QdrantClient,
    embedder: Embedder,
    recreate: bool = False,
) -> IngestReport:
    """Clean ``csv_path`` and index every film; log rows read/dropped, chunks written and elapsed time."""
    started = time.perf_counter()
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(
            run_metadata(settings, retrieval_mode="n/a", source=csv_path.name, collection=settings.qdrant.collection)
        )
    records, stats = clean_csv(csv_path, settings.ingest.min_plot_words)
    created = ensure_collection(client, settings, recreate=recreate)
    collection = settings.qdrant.collection
    logger.info("collection %s %s", collection, "created" if created else "already exists (resuming)")
    total, written, skipped = write_records(client, embedder, settings, records)
    in_collection = client.count(collection, exact=True).count
    report = IngestReport(
        source=csv_path.name,
        collection=collection,
        rows_read=stats.rows_read,
        rows_dropped=stats.rows_dropped,
        films=len(records),
        chunks_total=total,
        chunks_written=written,
        chunks_skipped=skipped,
        points_in_collection=in_collection,
        elapsed_s=round(time.perf_counter() - started, 2),
    )
    logger.info(
        "ingest done: rows_read=%d rows_dropped=%d films=%d chunks_total=%d chunks_written=%d chunks_skipped=%d "
        "points_in_collection=%d elapsed_s=%.2f",
        report.rows_read,
        report.rows_dropped,
        report.films,
        report.chunks_total,
        report.chunks_written,
        report.chunks_skipped,
        report.points_in_collection,
        report.elapsed_s,
    )
    if in_collection < total:
        raise IndexingError(f"Qdrant holds {in_collection} points but this run produced {total} chunks")
    return report
