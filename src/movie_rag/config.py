"""Typed settings: ``config.yaml`` for tunables, environment (and ``.env``) for overrides and secrets.

Precedence, highest first: init kwargs, process environment, ``.env``, ``config.yaml``. Secrets are ``SecretStr`` so
that ``repr()``, ``str()`` and logging never reveal them; use :meth:`Settings.require_nebius_api_key` and friends to
get a value at call time (they raise :class:`~movie_rag.errors.MissingCredentialError` with the documented hint).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from movie_rag.errors import MissingCredentialError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"  # resolved against the repository, never the current directory
CONFIG_ENV_VAR = "MOVIE_RAG_CONFIG"
REDACTED = "***"

RetrievalMode = Literal["dense", "sparse", "hybrid"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QdrantConfig(_Section):
    url: str
    collection: str
    timeout_s: int = Field(gt=0)


class EmbeddingsConfig(_Section):
    dense_model: str
    dense_dim: int = Field(gt=0)
    sparse_model: str


class IngestConfig(_Section):
    chunk_tokens: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    min_plot_words: int = Field(ge=0)
    random_seed: int


class RetrievalConfig(_Section):
    default_mode: RetrievalMode
    top_k: int = Field(gt=0)
    prefetch_limit: int = Field(gt=0)
    snippet_max_chars: int = Field(gt=0)
    demo_query: str = Field(min_length=1)


class LLMConfig(_Section):
    base_url: str
    chat_model: str
    judge_model: str
    temperature: float = Field(ge=0)
    max_tool_calls: int = Field(gt=0)


class McpConfig(_Section):
    url: str
    host: str
    port: int = Field(gt=0, le=65535)
    path: str = Field(pattern=r"^/")
    health_path: str = Field(pattern=r"^/")
    min_query_chars: int = Field(ge=1)
    max_query_chars: int = Field(ge=1)
    max_top_k: int = Field(gt=0)
    max_filter_values: int = Field(gt=0)
    title_lookup_limit: int = Field(gt=0)
    healthcheck_timeout_s: float = Field(gt=0)


class AgentConfig(_Section):
    prompt_version: str = Field(pattern=r"^[a-z0-9_]+$")
    max_citations: int = Field(gt=0)
    mcp_timeout_s: float = Field(gt=0)


class UiConfig(_Section):
    title: str = Field(min_length=1)
    mcp_timeout_s: float = Field(gt=0)
    example_questions: list[str] = Field(min_length=1)


class ObservabilityConfig(_Section):
    project: str
    langsmith_api_url: str
    git_timeout_s: float = Field(gt=0)


class DataConfig(_Section):
    dataset_slug: str
    kaggle_api_url: str
    kaggle_token_help_url: str
    raw_dir: Path
    csv_name: str
    fixture_path: Path
    max_download_mb: int = Field(gt=0)
    max_extracted_mb: int = Field(gt=0)
    http_timeout_s: float = Field(gt=0)

    def resolve(self, path: Path, root: Path = PROJECT_ROOT) -> Path:
        """Resolve a configured relative path against the project root."""
        return path if path.is_absolute() else root / path


class DoctorConfig(_Section):
    http_timeout_s: float = Field(gt=0)
    docker_timeout_s: float = Field(gt=0)


TUNABLE_SECTIONS = (
    "qdrant",
    "embeddings",
    "ingest",
    "retrieval",
    "llm",
    "mcp",
    "agent",
    "ui",
    "observability",
    "data",
    "doctor",
)


class Settings(BaseSettings):
    """Application settings. Build with :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=DEFAULT_CONFIG_PATH,
        yaml_file_encoding="utf-8",
    )

    # Tunables (config.yaml).
    qdrant: QdrantConfig
    embeddings: EmbeddingsConfig
    ingest: IngestConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    mcp: McpConfig
    agent: AgentConfig
    ui: UiConfig
    observability: ObservabilityConfig
    data: DataConfig
    doctor: DoctorConfig

    # Secrets (environment only; SecretStr never prints its value).
    nebius_api_key: SecretStr | None = None
    langsmith_api_key: SecretStr | None = None
    kaggle_key: SecretStr | None = None
    kaggle_username: str | None = None

    # Switches and well-known overrides (environment).
    langsmith_tracing: bool = False
    langsmith_project: str | None = None
    nebius_base_url: str | None = None
    qdrant_url: str | None = None
    mcp_url: str | None = None

    @field_validator(
        "nebius_api_key",
        "langsmith_api_key",
        "kaggle_key",
        "kaggle_username",
        "langsmith_project",
        "nebius_base_url",
        "qdrant_url",
        "mcp_url",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        """``KEY=`` in ``.env.example`` means "not set", not "set to the empty string"."""
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, SecretStr) and not value.get_secret_value().strip():
            return None
        return value

    @model_validator(mode="after")
    def _apply_overrides(self) -> Settings:
        if self.qdrant_url:
            self.qdrant.url = self.qdrant_url
        if self.nebius_base_url:
            self.llm.base_url = self.nebius_base_url
        if self.mcp_url:
            self.mcp.url = self.mcp_url
        if self.langsmith_project:
            self.observability.project = self.langsmith_project
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))

    # --- credentials ---------------------------------------------------------------------------------

    def require_nebius_api_key(self) -> SecretStr:
        """Return the Nebius key or raise ``MissingCredentialError`` (call time, never import time)."""
        if self.nebius_api_key is None:
            raise MissingCredentialError("NEBIUS_API_KEY")
        return self.nebius_api_key

    def require_kaggle_credentials(self) -> tuple[str, SecretStr]:
        """Return ``(username, key)`` or raise ``MissingCredentialError`` naming what is missing."""
        missing = [
            name
            for name, value in (("KAGGLE_USERNAME", self.kaggle_username), ("KAGGLE_KEY", self.kaggle_key))
            if value is None
        ]
        if missing or self.kaggle_username is None or self.kaggle_key is None:
            raise MissingCredentialError(*missing)
        return self.kaggle_username, self.kaggle_key

    @property
    def tracing_requested(self) -> bool:
        """True when the user asked for LangSmith tracing *and* a key exists (otherwise tracing is a no-op)."""
        return self.langsmith_tracing and self.langsmith_api_key is not None

    def secret_values(self) -> list[str]:
        """Plain-text secrets currently configured; only for redaction and tests. Never log the result."""
        secrets = [self.nebius_api_key, self.langsmith_api_key, self.kaggle_key]
        return [s.get_secret_value() for s in secrets if s is not None]

    def redact(self, text: str) -> str:
        """Replace every configured secret in ``text`` with ``***``."""
        for secret in sorted(self.secret_values(), key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        return text

    # --- hashing -------------------------------------------------------------------------------------

    def tunables(self) -> dict[str, Any]:
        """The effective tunables (no secrets, no switches) as plain data."""
        dumped = self.model_dump(mode="json", include=set(TUNABLE_SECTIONS))
        return dict(dumped)

    def config_hash(self) -> str:
        """Short, stable hash of the effective tunables: same config, same hash, across runs and machines."""
        canonical = json.dumps(self.tunables(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def default_config_path() -> Path:
    """``$MOVIE_RAG_CONFIG`` if set, else the repository's ``config.yaml``."""
    override = os.environ.get(CONFIG_ENV_VAR)
    return Path(override) if override else DEFAULT_CONFIG_PATH


class _UseDefault(Enum):
    DEFAULT = "default"


USE_DEFAULT = _UseDefault.DEFAULT


def load_settings(
    config_path: Path | str | None = None, *, env_file: Path | str | _UseDefault | None = USE_DEFAULT
) -> Settings:
    """Load settings from ``config_path`` (default: :func:`default_config_path`), the environment and ``env_file``.

    The default ``env_file`` is ``<repository>/.env`` wherever the process was started. Pass ``env_file=None`` to
    ignore ``.env`` (tests do this for isolation).
    """
    path = Path(config_path) if config_path is not None else default_config_path()
    dotenv = DEFAULT_ENV_FILE if isinstance(env_file, _UseDefault) else env_file

    class _Loaded(Settings):
        model_config = SettingsConfigDict(yaml_file=path, env_file=dotenv)

    _Loaded.__name__ = "Settings"
    # Fields are populated from the sources, not from constructor arguments, so build through a bare factory.
    factory: Callable[[], Settings] = _Loaded  # type: ignore[assignment]
    return factory()
