from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from movie_rag.config import CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH, Settings, default_config_path, load_settings
from movie_rag.errors import MissingCredentialError

MakeSettings = Callable[..., Settings]


def test_committed_config_matches_the_spec(make_settings: MakeSettings) -> None:
    s = make_settings()
    assert s.llm.base_url == "https://api.tokenfactory.nebius.com/v1/"
    assert s.llm.chat_model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert s.llm.judge_model == "openai/gpt-oss-120b"
    assert s.llm.chat_model != s.llm.judge_model
    assert (s.embeddings.dense_model, s.embeddings.dense_dim) == ("BAAI/bge-small-en-v1.5", 384)
    assert s.embeddings.sparse_model == "Qdrant/bm25"
    assert (s.ingest.chunk_tokens, s.ingest.chunk_overlap, s.ingest.batch_size) == (250, 40, 256)
    assert s.ingest.min_plot_words == 50
    assert (s.retrieval.default_mode, s.retrieval.top_k, s.retrieval.prefetch_limit) == ("hybrid", 8, 50)
    assert s.llm.max_tool_calls == 4


def test_secrets_default_to_unset(make_settings: MakeSettings) -> None:
    s = make_settings()
    assert s.nebius_api_key is None
    assert s.kaggle_key is None
    assert s.langsmith_api_key is None
    assert s.langsmith_tracing is False


def test_environment_overrides_nested_values(make_settings: MakeSettings) -> None:
    s = make_settings(LLM__CHAT_MODEL="other/model", RETRIEVAL__TOP_K="3")
    assert s.llm.chat_model == "other/model"
    assert s.retrieval.top_k == 3
    assert s.llm.judge_model == "openai/gpt-oss-120b"  # untouched siblings survive the merge


def test_well_known_variables_override_yaml(make_settings: MakeSettings) -> None:
    s = make_settings(
        QDRANT_URL="http://qdrant:6333",
        NEBIUS_BASE_URL="https://example.test/v1/",
        MCP_URL="http://mcp:8000/mcp",
        LANGSMITH_PROJECT="my-project",
    )
    assert s.qdrant.url == "http://qdrant:6333"
    assert s.llm.base_url == "https://example.test/v1/"
    assert s.mcp.url == "http://mcp:8000/mcp"
    assert s.observability.project == "my-project"


def test_blank_values_count_as_unset(make_settings: MakeSettings) -> None:
    s = make_settings(NEBIUS_API_KEY="  ", KAGGLE_USERNAME="", QDRANT_URL="")
    assert s.nebius_api_key is None
    assert s.kaggle_username is None
    assert s.qdrant.url == "http://localhost:6333"


def test_init_kwargs_with_blank_secret_object_are_unset() -> None:
    from pydantic import SecretStr

    s = load_settings(env_file=None)
    rebuilt = Settings.model_validate({**s.model_dump(), "nebius_api_key": SecretStr(" ")})
    assert rebuilt.nebius_api_key is None


def test_config_path_can_be_given_or_come_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["retrieval"]["top_k"] = 11
    custom = tmp_path / "custom.yaml"
    custom.write_text(yaml.safe_dump(raw))

    assert load_settings(custom, env_file=None).retrieval.top_k == 11
    monkeypatch.setenv(CONFIG_ENV_VAR, str(custom))
    assert default_config_path() == custom
    assert load_settings(env_file=None).retrieval.top_k == 11
    monkeypatch.delenv(CONFIG_ENV_VAR)
    assert default_config_path() == DEFAULT_CONFIG_PATH


def test_env_file_is_read_when_given(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("NEBIUS_API_KEY=from-dotenv-value\nRETRIEVAL__TOP_K=5\n")
    s = load_settings(env_file=env_file)
    assert s.require_nebius_api_key().get_secret_value() == "from-dotenv-value"
    assert s.retrieval.top_k == 5


def test_unknown_yaml_keys_are_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["retrieval"]["top_kk"] = 1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValidationError, match="top_kk"):
        load_settings(bad, env_file=None)


def test_invalid_value_is_rejected(make_settings: MakeSettings) -> None:
    with pytest.raises(ValidationError):
        make_settings(RETRIEVAL__DEFAULT_MODE="fuzzy")


def test_require_nebius_key_raises_documented_error_only_when_called(make_settings: MakeSettings) -> None:
    s = make_settings()  # building settings without a key must not raise
    with pytest.raises(MissingCredentialError) as excinfo:
        s.require_nebius_api_key()
    assert str(excinfo.value) == "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"


def test_require_nebius_key_returns_secret_when_set(make_settings: MakeSettings) -> None:
    s = make_settings(NEBIUS_API_KEY="fake-nebius-value")
    assert s.require_nebius_api_key().get_secret_value() == "fake-nebius-value"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "set KAGGLE_USERNAME and KAGGLE_KEY — see docs/CREDENTIALS.md"),
        ({"KAGGLE_USERNAME": "someone"}, "set KAGGLE_KEY — see docs/CREDENTIALS.md"),
        ({"KAGGLE_KEY": "fake-kaggle-value"}, "set KAGGLE_USERNAME — see docs/CREDENTIALS.md"),
    ],
)
def test_require_kaggle_names_what_is_missing(make_settings: MakeSettings, env: dict[str, str], expected: str) -> None:
    with pytest.raises(MissingCredentialError) as excinfo:
        make_settings(**env).require_kaggle_credentials()
    assert str(excinfo.value) == expected


def test_require_kaggle_returns_both_parts(make_settings: MakeSettings) -> None:
    username, key = make_settings(
        KAGGLE_USERNAME="someone", KAGGLE_KEY="fake-kaggle-value"
    ).require_kaggle_credentials()
    assert username == "someone"
    assert key.get_secret_value() == "fake-kaggle-value"


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, False),
        ({"LANGSMITH_TRACING": "true"}, False),  # requested but no key: no-op
        ({"LANGSMITH_API_KEY": "fake-ls-value"}, False),  # key but not requested
        ({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "fake-ls-value"}, True),
    ],
)
def test_tracing_requires_switch_and_key(make_settings: MakeSettings, env: dict[str, str], expected: bool) -> None:
    assert make_settings(**env).tracing_requested is expected


def test_config_hash_is_stable_across_loads(make_settings: MakeSettings) -> None:
    first, second = make_settings().config_hash(), load_settings(env_file=None).config_hash()
    assert first == second
    assert len(first) == 12
    int(first, 16)  # hex


def test_config_hash_changes_with_tunables(monkeypatch: pytest.MonkeyPatch) -> None:
    base = load_settings(env_file=None).config_hash()
    monkeypatch.setenv("RETRIEVAL__TOP_K", "9")
    assert load_settings(env_file=None).config_hash() != base


def test_config_hash_ignores_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    base = load_settings(env_file=None).config_hash()
    monkeypatch.setenv("NEBIUS_API_KEY", "fake-nebius-value")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert load_settings(env_file=None).config_hash() == base


def test_tunables_contain_no_secret_fields(make_settings: MakeSettings) -> None:
    tunables = make_settings(NEBIUS_API_KEY="fake-nebius-value").tunables()
    assert "fake-nebius-value" not in str(tunables)
    assert not {"nebius_api_key", "kaggle_key", "langsmith_api_key"} & tunables.keys()


def test_redact_replaces_longest_secret_first(make_settings: MakeSettings) -> None:
    s = make_settings(NEBIUS_API_KEY="fake-nebius-value", KAGGLE_KEY="fake-nebius")
    redacted = s.redact("token fake-nebius-value and fake-nebius")
    assert "fake-nebius" not in redacted
    assert redacted == "token *** and ***"


def test_secrets_are_masked_in_repr_and_dumps(make_settings: MakeSettings) -> None:
    s = make_settings(
        NEBIUS_API_KEY="fake-nebius-value", LANGSMITH_API_KEY="fake-ls-value", KAGGLE_KEY="fake-kaggle-value"
    )
    for text in (repr(s), str(s), s.model_dump_json(), str(s.model_dump())):
        for secret in s.secret_values():
            assert secret not in text
