"""Offline stand-ins shared by the unit tests (no model downloads, no network)."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from langsmith import Client
from langsmith.run_helpers import tracing_context
from qdrant_client import models as m

SPARSE_SPACE = 1 << 20


def _bucket(word: str, modulo: int) -> int:
    return int.from_bytes(hashlib.md5(word.lower().encode()).digest()[:4], "big") % modulo


class FakeEmbedder:
    """Deterministic bag-of-words embeddings; one token is one whitespace-separated word.

    Records every batch it is asked to embed so tests can assert what was (not) re-embedded.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.dense_batches: list[list[str]] = []
        self.sparse_batches: list[list[str]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _dense(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in text.split():
            vector[_bucket(word, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def _sparse(self, text: str) -> m.SparseVector:
        counts = Counter(_bucket(word, SPARSE_SPACE) for word in text.split())
        indices = sorted(counts)
        return m.SparseVector(indices=indices, values=[float(counts[i]) for i in indices])

    def embed_dense(self, texts: Sequence[str]) -> list[list[float]]:
        self.dense_batches.append(list(texts))
        return [self._dense(t) for t in texts]

    def embed_sparse(self, texts: Sequence[str]) -> list[m.SparseVector]:
        self.sparse_batches.append(list(texts))
        return [self._sparse(t) for t in texts]

    def embed_dense_query(self, text: str) -> list[float]:
        return self._dense(text)

    def embed_sparse_query(self, text: str) -> m.SparseVector:
        return self._sparse(text)

    @property
    def embedded_texts(self) -> list[str]:
        return [t for batch in self.dense_batches for t in batch]


class RunCapture:
    """LangSmith runs recorded from the HTTP requests of a client whose session is a mock (nothing leaves the process).

    Use as ``with RunCapture() as capture:`` around code that opens spans; ``capture.runs`` then maps run name to the
    merged POST/PATCH payloads (inputs, outputs, metadata, parent).
    """

    def __init__(self) -> None:
        self.session = MagicMock()
        self.client = Client(
            api_key="test-key", api_url="http://localhost:9", session=self.session, auto_batch_tracing=False
        )
        self._context = tracing_context(enabled=True, client=self.client)

    def __enter__(self) -> RunCapture:
        self._context.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._context.__exit__(None, None, None)

    @property
    def runs(self) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for call in self.session.request.call_args_list:
            body = call.kwargs.get("data")
            if not body:
                continue
            payload = json.loads(body)
            run = merged.setdefault(payload["id"], {})
            run.update({k: v for k, v in payload.items() if v is not None})
        return {run["name"]: run for run in merged.values() if "name" in run}

    @property
    def payloads(self) -> str:
        """Everything that would have been uploaded, as one string (for secret scans)."""
        return " ".join(str(call.kwargs.get("data")) for call in self.session.request.call_args_list)
