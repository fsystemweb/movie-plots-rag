"""Cross-cutting guarantee: configured secrets never reach logs, output, reprs or trace metadata."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from movie_rag import doctor
from movie_rag.config import Settings
from movie_rag.ingest import download
from movie_rag.observability import configure_tracing, run_metadata

SECRETS = {
    "NEBIUS_API_KEY": "fake-nebius-secret-0001",
    "LANGSMITH_API_KEY": "fake-langsmith-secret-0002",
    "KAGGLE_KEY": "fake-kaggle-secret-0003",
}
ENV = {**SECRETS, "KAGGLE_USERNAME": "someone", "LANGSMITH_TRACING": "true"}


def leaky(request: httpx.Request) -> httpx.Response:
    """A failing service whose error text echoes every secret, the worst case for accidental logging."""
    raise httpx.ConnectError("connection failed: " + " ".join(SECRETS.values()), request=request)


@pytest.fixture
def settings(make_settings: Callable[..., Settings], tmp_path: Path) -> Settings:
    return make_settings(DATA__RAW_DIR=str(tmp_path / "raw"), **ENV)


def assert_no_secret(*texts: str) -> None:
    for text in texts:
        for secret in SECRETS.values():
            assert secret not in text


def test_secrets_are_really_configured(settings: Settings) -> None:
    assert sorted(settings.secret_values()) == sorted(SECRETS.values())  # guards against a vacuous test


def test_settings_never_expose_secrets_in_text_forms(settings: Settings) -> None:
    assert_no_secret(
        repr(settings), str(settings), settings.model_dump_json(), str(settings.model_dump()), str(settings.tunables())
    )
    assert_no_secret(repr(settings.nebius_api_key), str(settings.kaggle_key))


def test_run_metadata_has_no_secrets(settings: Settings) -> None:
    assert_no_secret(repr(run_metadata(settings, retrieval_mode="hybrid")))


def test_configure_tracing_logs_no_secrets(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        configure_tracing(settings)
        configure_tracing(settings.model_copy(update={"langsmith_api_key": None}))
    assert_no_secret(caplog.text)


def test_doctor_report_and_logs_contain_no_secrets(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(doctor, "load_settings", lambda: settings)
    client = httpx.Client(transport=httpx.MockTransport(leaky))
    with caplog.at_level(logging.DEBUG):
        assert doctor.main(client=client) == 0
    captured = capsys.readouterr()
    assert "FAIL" in captured.out  # the failures were reported...
    assert_no_secret(captured.out, captured.err, caplog.text)  # ...without echoing a secret


def test_download_failure_output_and_logs_contain_no_secrets(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(download, "load_settings", lambda: settings)
    client = httpx.Client(transport=httpx.MockTransport(leaky))
    with caplog.at_level(logging.DEBUG):
        assert download.main([], client=client) == download.EXIT_FAILURE
    captured = capsys.readouterr()
    assert "could not reach Kaggle" in captured.err
    assert_no_secret(captured.out, captured.err, caplog.text)
