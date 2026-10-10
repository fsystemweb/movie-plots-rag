"""Versioned system prompts (``prompts/<version>.md``); the version in ``config.yaml`` is recorded in every trace."""

from __future__ import annotations

import re
from pathlib import Path

from movie_rag.config import Settings
from movie_rag.errors import AgentError

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_VERSION = re.compile(r"^[a-z0-9_]+$")


def load_prompt(settings: Settings) -> str:
    """The system prompt for ``agent.prompt_version`` with the limits from the configuration filled in."""
    version = settings.agent.prompt_version
    path = PROMPT_DIR / f"{version}.md"
    if not _VERSION.fullmatch(version) or not path.is_file():
        raise AgentError(
            f"unknown prompt version {version!r}: expected a file prompts/{version}.md in the agent package"
        )
    return path.read_text(encoding="utf-8").format_map(
        {"max_tool_calls": settings.llm.max_tool_calls, "max_films": settings.agent.max_citations}
    )
