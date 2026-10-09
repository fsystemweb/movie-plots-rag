"""Tracing switch and run metadata.

LangSmith tracing is a strict no-op unless ``LANGSMITH_TRACING`` is true *and* ``LANGSMITH_API_KEY`` is set; the calling
code is identical either way. :func:`run_metadata` returns the fields every traced run carries. Nothing returned here
may contain a secret.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from functools import lru_cache
from typing import Any, cast

from langsmith import utils as ls_utils

from movie_rag.config import PROJECT_ROOT, Settings, load_settings

logger = logging.getLogger(__name__)

UNKNOWN_SHA = "unknown"
_KEY_SPLIT = re.compile(r"[^a-z0-9]+")
_SENSITIVE_WORDS = frozenset(
    {"password", "passwd", "secret", "secrets", "authorization", "credential", "credentials", "apikey", "bearer"}
)
_KEY_QUALIFIERS = frozenset(
    {"api", "kaggle", "nebius", "langsmith", "secret", "private", "access", "auth", "ssh", "signing"}
)
_TOKEN_QUALIFIERS = frozenset({"access", "auth", "bearer", "refresh", "api", "session", "id", "secret"})
_SECRET_SHAPED_VALUE = re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|lsv2_[A-Za-z0-9_]{8,})")


def is_credential_like(key: str) -> bool:
    """True for field names such as ``api_key``, ``NEBIUS_API_KEY``, ``access_token`` or ``authorization``.

    Counters such as ``total_tokens`` or ``prompt_tokens`` are fine: only a lone ``token``/``key`` or one qualified as
    a credential counts.
    """
    words = [w for w in _KEY_SPLIT.split(key.lower()) if w]
    if not words:
        return False
    if _SENSITIVE_WORDS & set(words):
        return True
    last, before = words[-1], words[-2] if len(words) > 1 else ""
    if last == "key":
        return not before or before in _KEY_QUALIFIERS
    if last == "token":
        return not before or before in _TOKEN_QUALIFIERS
    return False


@lru_cache(maxsize=4)
def git_sha(timeout_s: float) -> str:
    """Short git sha of the working tree's HEAD, or ``"unknown"`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else UNKNOWN_SHA


def tracing_enabled(settings: Settings | None = None) -> bool:
    """True only when tracing was requested and a LangSmith key exists."""
    return (settings or load_settings()).tracing_requested


def configure_tracing(settings: Settings | None = None) -> bool:
    """Align the process environment with the tracing switch and return whether tracing is on.

    Without a key, ``LANGSMITH_TRACING`` is forced to ``false`` so ``@traceable`` stays a silent no-op instead of
    failing to upload. With a key and the switch on, the key and project are exported for the LangSmith client.
    """
    settings = settings or load_settings()
    enabled = settings.tracing_requested
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled and settings.langsmith_api_key is not None:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
        os.environ["LANGSMITH_PROJECT"] = settings.observability.project
    cast(Any, ls_utils.get_env_var).cache_clear()  # the client memoises environment reads
    if settings.langsmith_tracing and not enabled:
        logger.info("LangSmith tracing requested but LANGSMITH_API_KEY is not set: tracing disabled")
    return enabled


def run_metadata(settings: Settings | None = None, **extra: Any) -> dict[str, Any]:
    """Metadata attached to every traced run.

    Always contains ``git_sha``, ``prompt_version``, ``chat_model``, ``embedding_model``, ``retrieval_mode`` and
    ``config_hash``. Keyword arguments override a default (``retrieval_mode="dense"``) or add a field. Keys that look
    like credentials, and values shaped like API keys, are rejected rather than traced.
    """
    settings = settings or load_settings()
    metadata: dict[str, Any] = {
        "git_sha": git_sha(settings.observability.git_timeout_s),
        "prompt_version": settings.agent.prompt_version,
        "chat_model": settings.llm.chat_model,
        "embedding_model": settings.embeddings.dense_model,
        "sparse_model": settings.embeddings.sparse_model,
        "retrieval_mode": settings.retrieval.default_mode,
        "config_hash": settings.config_hash(),
    }
    for key, value in extra.items():
        if is_credential_like(key):
            raise ValueError(f"refusing to put a credential-like field in trace metadata: {key!r}")
        if isinstance(value, str) and (_SECRET_SHAPED_VALUE.search(value) or value in settings.secret_values()):
            raise ValueError(f"refusing to put a secret-shaped value in trace metadata field {key!r}")
        metadata[key] = value
    return metadata
