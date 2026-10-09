"""Shared fixtures. Tests never need credentials and never touch the network.

``LANGSMITH_TRACING`` is forced off at import time (before any test module imports langsmith) and again per test.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from langsmith import utils as ls_utils

os.environ["LANGSMITH_TRACING"] = "false"

from movie_rag.config import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "movies_sample.csv"

# Everything the settings layer reads from the environment; scrubbed so a developer's shell or .env cannot leak in.
SETTINGS_ENV_VARS = (
    "NEBIUS_API_KEY",
    "NEBIUS_BASE_URL",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "QDRANT_URL",
    "MCP_URL",
    "MOVIE_RAG_CONFIG",
)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in [n for n in os.environ if "__" in n and n.split("__")[0].lower() in Settings.model_fields]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    ls_utils.get_env_var.cache_clear()  # type: ignore[attr-defined]
    yield
    ls_utils.get_env_var.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def make_settings(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    """Build settings from ``config.yaml`` plus the given environment variables, ignoring any ``.env``."""

    def _make(**env: str) -> Settings:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return load_settings(env_file=None)

    return _make
