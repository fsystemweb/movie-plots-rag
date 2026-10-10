"""Offline stand-ins shared by the unit tests (no model downloads, no network)."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langsmith import Client
from langsmith.run_helpers import tracing_context
from pydantic import Field
from qdrant_client import models as m

SPARSE_SPACE = 1 << 20
_CALL_IDS = itertools.count(1)


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


class ScriptedChatModel(BaseChatModel):
    """A tool-calling chat model that replays scripted replies (no network, no key).

    ``replies`` are returned in order; once they run out the last one repeats (a model that never stops calling tools
    is just ``[tool_call_message(...)]``). ``prompts`` records the messages each call received.
    """

    replies: list[AIMessage]
    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:  # type: ignore[override]
        self.bound_tool_names = [t.name for t in tools if isinstance(t, BaseTool)]
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.prompts.append(list(messages))
        turn = len(self.prompts) - 1
        reply = self.replies[min(turn, len(self.replies) - 1)]
        if reply.tool_calls:  # a repeated reply must not reuse tool call ids
            reply = reply.model_copy(
                update={"tool_calls": [{**call, "id": f"call_{next(_CALL_IDS)}"} for call in reply.tool_calls]}
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call_message(*calls: tuple[str, dict[str, Any]], tokens: int = 0) -> AIMessage:
    """An assistant turn that requests the given ``(tool name, args)`` calls at once."""
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{next(_CALL_IDS)}", "type": "tool_call"} for name, args in calls
        ],
    )
    if tokens:
        message.usage_metadata = {"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1}
    return message
