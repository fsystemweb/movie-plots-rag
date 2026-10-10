"""Typed results of an agent run. Everything a caller shows (UI, CLI, eval) comes from these objects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A film the answer is allowed to cite. Always built from a tool result, never from the model's own text."""

    movie_id: str
    title: str
    release_year: int | None = None
    wiki_url: str | None = None

    @property
    def label(self) -> str:
        """``Title (Year)``, the form the prompt asks the model to write."""
        return f"{self.title} ({self.release_year})" if self.release_year is not None else self.title


class RetrievedFilm(Citation):
    """A film returned by a tool during the run (the "Retrieved films" list of the UI)."""

    director: str | None = None
    genre: str | None = None
    origin: str | None = None
    score: float | None = None
    snippet: str | None = None
    tool: str | None = None


class ToolCallRecord(BaseModel):
    """One tool call the agent made (blocked calls over the cap are counted separately, not recorded here)."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_count: int = Field(description="Films the call returned (0 for list_filters and failed calls).")
    error: str | None = None


class Usage(BaseModel):
    """Token usage summed over the model calls of the run (0 when the model reports none)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class Answer(BaseModel):
    """What ``MovieAgent.ask`` returns."""

    question: str
    text: str = Field(description="The answer to show. The abstention message when no retrieved film is cited.")
    citations: list[Citation] = Field(description="Retrieved films the answer mentions, in order of mention.")
    abstained: bool = Field(description="True when the answer cites no film (nothing fitted, or nothing verifiable).")
    retrieved: list[RetrievedFilm] = Field(description="Every distinct film the tools returned, in order of retrieval.")
    tool_calls: list[ToolCallRecord]
    blocked_tool_calls: int = Field(default=0, description="Calls refused because the per-question cap was reached.")
    stopped_early: bool = Field(default=False, description="The run hit the step limit before the model answered.")
    metadata: dict[str, Any] = Field(description="observability.run_metadata(): git sha, prompt version, models, ...")
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
