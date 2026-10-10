"""The fixture ingested into an in-memory Qdrant (fake embedder) plus one remake; shared by the MCP and agent tests."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from fakes import FakeEmbedder
from movie_rag.config import load_settings
from movie_rag.ingest.clean import MovieRecord, clean_csv
from movie_rag.ingest.index import ensure_collection
from movie_rag.ingest.pipeline import ingest_csv, write_records
from movie_rag.retrieval import Retriever

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "movies_sample.csv"


class Index:
    """The fixture in an in-memory collection plus one remake, so that a title can be ambiguous."""

    def __init__(self) -> None:
        self.settings = load_settings(env_file=None)
        self.client = QdrantClient(":memory:")
        ingest_csv(self.settings, FIXTURE, client=self.client, embedder=FakeEmbedder(), recreate=True)
        records, _ = clean_csv(FIXTURE, self.settings.ingest.min_plot_words)
        self.records = records
        self.original = records[0]
        self.remake = self.original.model_copy(
            update={"movie_id": f"{self.original.movie_id}-remake", "release_year": self.original.release_year + 30}
        )
        ensure_collection(self.client, self.settings)
        write_records(self.client, FakeEmbedder(), self.settings, [self.remake])

    def retriever(self) -> Retriever:
        return Retriever(self.settings, client=self.client, embedder=FakeEmbedder())


def premise(record: MovieRecord) -> str:
    return ". ".join(record.plot.split(". ")[:2])
