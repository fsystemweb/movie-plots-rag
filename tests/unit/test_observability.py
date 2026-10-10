from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from typing import Any

import pytest
from langsmith import traceable
from langsmith import utils as ls_utils

from movie_rag import observability
from movie_rag.config import Settings
from movie_rag.observability import configure_tracing, git_sha, is_credential_like, run_metadata, tracing_enabled

MakeSettings = Callable[..., Settings]
REQUIRED_KEYS = {"git_sha", "prompt_version", "chat_model", "embedding_model", "retrieval_mode", "config_hash"}


def test_run_metadata_has_every_required_field(make_settings: MakeSettings) -> None:
    s = make_settings()
    meta = run_metadata(s)
    assert meta.keys() >= REQUIRED_KEYS
    assert meta["prompt_version"] == "system_v1"
    assert meta["chat_model"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert meta["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert meta["retrieval_mode"] == "hybrid"
    assert meta["config_hash"] == s.config_hash()
    assert re.fullmatch(r"[0-9a-f]{7,40}|unknown", meta["git_sha"])


def test_run_metadata_extras_override_and_add(make_settings: MakeSettings) -> None:
    meta = run_metadata(make_settings(), retrieval_mode="dense", top_k=3)
    assert meta["retrieval_mode"] == "dense"
    assert meta["top_k"] == 3


def test_run_metadata_loads_settings_when_none_given() -> None:
    assert run_metadata().keys() >= REQUIRED_KEYS


CREDENTIAL_KEYS = [
    "api_key",
    "NEBIUS_API_KEY",
    "LANGSMITH_API_KEY",
    "KAGGLE_KEY",
    "key",
    "apikey",
    "private-key",
    "authorization",
    "Authorization",
    "access_token",
    "auth_token",
    "bearer_token",
    "token",
    "password",
    "client_secret",
    "credentials",
]
HARMLESS_KEYS = [
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "tokens",
    "max_tokens",
    "tool_calls",
    "top_k",
    "cache_key",
    "sort_key",
    "latency_ms",
    "filters",
    "retrieval_mode",
]


@pytest.mark.parametrize("key", CREDENTIAL_KEYS)
def test_run_metadata_rejects_credential_like_fields(make_settings: MakeSettings, key: str) -> None:
    assert is_credential_like(key)
    with pytest.raises(ValueError, match="credential-like"):
        run_metadata(make_settings(), **{key: "anything"})


@pytest.mark.parametrize("key", HARMLESS_KEYS)
def test_run_metadata_accepts_harmless_fields_such_as_token_counts(make_settings: MakeSettings, key: str) -> None:
    assert not is_credential_like(key)
    assert run_metadata(make_settings(), **{key: 123})[key] == 123


def test_empty_key_is_not_credential_like() -> None:
    assert is_credential_like("") is False


def test_git_timeout_comes_from_settings(make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="abc1234\n", stderr="")

    git_sha.cache_clear()
    monkeypatch.setattr(observability.subprocess, "run", fake_run)
    meta = run_metadata(make_settings(OBSERVABILITY__GIT_TIMEOUT_S="7"))
    assert meta["git_sha"] == "abc1234"
    assert seen == [7.0]
    git_sha.cache_clear()


@pytest.mark.parametrize("value", ["sk-abcdefghijklmnop", "lsv2_pt_abcdefghijkl", "fake-nebius-value"])
def test_run_metadata_rejects_secret_values(make_settings: MakeSettings, value: str) -> None:
    s = make_settings(NEBIUS_API_KEY="fake-nebius-value")
    with pytest.raises(ValueError, match="secret-shaped"):
        run_metadata(s, note=value)


def test_run_metadata_never_contains_configured_secrets(make_settings: MakeSettings) -> None:
    s = make_settings(
        NEBIUS_API_KEY="fake-nebius-value", LANGSMITH_API_KEY="fake-ls-value", KAGGLE_KEY="fake-kaggle-value"
    )
    text = repr(run_metadata(s, retrieval_mode="sparse"))
    assert not any(secret in text for secret in s.secret_values())


def test_non_string_extras_pass_through(make_settings: MakeSettings) -> None:
    assert run_metadata(make_settings(), filters={"genre": "drama"})["filters"] == {"genre": "drama"}


def test_git_sha_is_cached_and_short() -> None:
    git_sha.cache_clear()
    first = git_sha(5.0)
    assert git_sha(5.0) == first
    assert first == "unknown" or 7 <= len(first) <= 12


def test_git_sha_falls_back_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    git_sha.cache_clear()
    monkeypatch.setattr(observability.subprocess, "run", boom)
    assert git_sha(5.0) == "unknown"
    git_sha.cache_clear()


def test_git_sha_falls_back_outside_a_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    git_sha.cache_clear()
    failed = subprocess.CompletedProcess(args=[], returncode=128, stdout="", stderr="not a git repository")
    monkeypatch.setattr(observability.subprocess, "run", lambda *a, **k: failed)
    assert git_sha(5.0) == "unknown"
    git_sha.cache_clear()


def test_tracing_enabled_follows_switch_and_key(make_settings: MakeSettings) -> None:
    assert tracing_enabled(make_settings()) is False
    assert tracing_enabled(make_settings(LANGSMITH_TRACING="true")) is False
    assert tracing_enabled(make_settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="fake-ls-value")) is True


def test_tracing_enabled_loads_settings_when_none_given() -> None:
    assert tracing_enabled() is False


def test_configure_tracing_without_key_forces_tracing_off(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    s = make_settings(LANGSMITH_TRACING="true")
    with caplog.at_level("INFO", logger="movie_rag.observability"):
        assert configure_tracing(s) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert ls_utils.tracing_is_enabled() is False
    assert "LANGSMITH_API_KEY is not set" in caplog.text
    assert "LANGSMITH_API_KEY" not in os.environ


def test_configure_tracing_with_key_exports_environment(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = make_settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="fake-ls-value", LANGSMITH_PROJECT="proj")
    monkeypatch.setenv("LANGSMITH_PROJECT", "to-be-overwritten")  # registered, so monkeypatch restores it
    assert configure_tracing(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    assert ls_utils.tracing_is_enabled() is True


def test_configure_tracing_loads_settings_when_none_given() -> None:
    assert configure_tracing() is False


def test_traceable_functions_behave_identically_with_tracing_off(make_settings: MakeSettings) -> None:
    @traceable(run_type="retriever", name="search", metadata=run_metadata(make_settings(), retrieval_mode="dense"))
    def search(query: str) -> list[dict[str, Any]]:
        return [{"page_content": query.upper(), "type": "Document", "metadata": {}}]

    assert ls_utils.tracing_is_enabled() is False
    assert search("heist")[0]["page_content"] == "HEIST"


# --- scrub and span --------------------------------------------------------------------------------------------

FAKE_KEY = "sk-" + "abcdefghijklmnop1234"  # built at runtime: not a secret, and not shaped like one in the source
CONFIGURED_SECRET = "-".join(["configured", "value", "0123456789"])


def test_scrub_replaces_secret_shaped_and_configured_secrets_recursively(make_settings: MakeSettings) -> None:
    s = make_settings(NEBIUS_API_KEY="plain-secret-value")
    data = {"q": f"use {FAKE_KEY} now", "nested": ["plain-secret-value", ("lsv2_abcdefgh1234", 3)], "n": 4}
    out = observability.scrub(data, s)
    assert out == {"q": "use *** now", "nested": ["***", ["***", 3]], "n": 4}


def test_span_runs_the_block_identically_with_tracing_off(make_settings: MakeSettings) -> None:
    with observability.span("work", run_type="tool", settings=make_settings(), inputs={"a": 1}, tool="x") as run:
        run.set(result_count=2)
    assert run.outputs["result_count"] == 2
    assert run.outputs["latency_ms"] >= 0


def test_span_loads_settings_when_none_given() -> None:
    with observability.span("work") as run:
        run.set(ok=True)
    assert run.outputs["ok"] is True


def test_span_reraises_errors_unchanged(make_settings: MakeSettings) -> None:
    with pytest.raises(KeyError, match="boom"), observability.span("work", settings=make_settings()):
        raise KeyError("boom")


def test_span_uploads_metadata_inputs_outputs_and_latency_without_secrets(make_settings: MakeSettings) -> None:
    from fakes import RunCapture

    s = make_settings(LANGSMITH_API_KEY=CONFIGURED_SECRET)
    with (
        RunCapture() as capture,
        observability.span(
            "outer",
            run_type="tool",
            settings=s,
            inputs={"query": f"find {FAKE_KEY}"},
            tool="t",
            genre=FAKE_KEY,
            retrieval_mode="sparse",
        ) as outer,
    ):
        with observability.span("inner", run_type="retriever", settings=s) as inner:
            inner.set(result_count=3, note=CONFIGURED_SECRET)
        outer.set(result_count=1)
    runs = capture.runs
    assert runs["outer"]["run_type"] == "tool" and runs["inner"]["run_type"] == "retriever"
    assert runs["inner"]["parent_run_id"] == runs["outer"]["id"]
    assert runs["outer"]["inputs"] == {"query": "find ***"}
    assert runs["outer"]["outputs"]["result_count"] == 1 and "latency_ms" in runs["outer"]["outputs"]
    assert runs["inner"]["outputs"]["result_count"] == 3
    assert runs["inner"]["extra"]["metadata"].keys() >= REQUIRED_KEYS
    outer_meta = runs["outer"]["extra"]["metadata"]
    assert outer_meta.keys() >= REQUIRED_KEYS and outer_meta["tool"] == "t" and outer_meta["retrieval_mode"] == "sparse"
    assert outer_meta["genre"] == "***"
    assert FAKE_KEY not in capture.payloads and CONFIGURED_SECRET not in capture.payloads


# --- trace link (used by the Streamlit page) -------------------------------------------------------------------


class _Run:
    def __init__(self, url: str | Exception) -> None:
        self.url = url

    def get_url(self) -> str:
        if isinstance(self.url, Exception):
            raise self.url
        return self.url


def test_trace_url_is_none_when_tracing_is_off(make_settings: MakeSettings) -> None:
    handle = observability.Span(make_settings(), _Run("https://smith.example/r/1"))
    assert handle.trace_url() is None


def test_trace_url_is_none_without_a_run(make_settings: MakeSettings) -> None:
    on = make_settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="fake-ls-key-456")
    assert observability.Span(on).trace_url() is None


def test_trace_url_is_the_runs_link_when_tracing_is_on(make_settings: MakeSettings) -> None:
    on = make_settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="fake-ls-key-456")
    assert observability.Span(on, _Run("https://smith.example/r/1")).trace_url() == "https://smith.example/r/1"


def test_a_link_that_cannot_be_built_is_none_not_an_error(
    make_settings: MakeSettings, caplog: pytest.LogCaptureFixture
) -> None:
    on = make_settings(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="fake-ls-key-456")
    assert observability.Span(on, _Run(RuntimeError("tenant lookup failed"))).trace_url() is None
    assert "could not build the LangSmith trace link" in caplog.text


def test_the_span_handle_carries_the_run_so_the_caller_can_ask_for_the_link(make_settings: MakeSettings) -> None:
    with observability.span("t", settings=make_settings()) as handle:
        assert handle.trace_url() is None  # tracing is off in tests
