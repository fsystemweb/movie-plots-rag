from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from movie_rag import doctor
from movie_rag.config import Settings
from movie_rag.doctor import CheckResult, Status, render_table, run_checks

MakeSettings = Callable[..., Settings]
CHAT = "Qwen/Qwen3-30B-A3B-Instruct-2507"
JUDGE = "openai/gpt-oss-120b"
ALL_CREDS = {
    "NEBIUS_API_KEY": "fake-nebius-value",
    "LANGSMITH_API_KEY": "fake-ls-value",
    "KAGGLE_USERNAME": "someone",
    "KAGGLE_KEY": "fake-kaggle-value",
}


class FakeNetwork:
    """Routes mocked HTTP calls by host; records every request. Nothing leaves the process."""

    def __init__(self, **overrides: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.handlers: dict[str, Callable[[httpx.Request], httpx.Response]] = {
            "localhost": lambda r: httpx.Response(200, text="all shards are ready"),
            "api.tokenfactory.nebius.com": lambda r: httpx.Response(
                200, json={"data": [{"id": CHAT}, {"id": JUDGE}, {"id": "other/model"}]}
            ),
            "api.smith.langchain.com": lambda r: httpx.Response(200, json=[]),
            "www.kaggle.com": lambda r: httpx.Response(200, json={"ref": "x"}),
        }
        self.handlers.update(overrides)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handlers[request.url.host](request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


@pytest.fixture(autouse=True)
def docker_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0, stdout="29.0.1\n", stderr=""),
    )


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


@pytest.fixture
def with_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("DATA__RAW_DIR", str(tmp_path / "raw"))
    return tmp_path / "raw"


# --- no credentials: the common case -----------------------------------------------------------------


def test_without_credentials_every_missing_item_gets_a_next_step_and_nothing_authenticated_is_called(
    make_settings: MakeSettings, with_data: Path
) -> None:
    net = FakeNetwork()
    results = by_name(run_checks(make_settings(), net.client()))
    for name in ("Nebius API key", "LangSmith", "Kaggle"):
        assert results[name].status is Status.MISSING
        assert "docs/CREDENTIALS.md" in results[name].next_step
    assert results["Nebius API key"].next_step == "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"
    assert results["Model: chat"].status is Status.SKIPPED
    assert results["Model: judge"].status is Status.SKIPPED
    assert [r.url.host for r in net.requests] == ["localhost"]  # only the Qdrant readiness probe


def test_dataset_row_points_to_the_fixture_when_nothing_is_downloaded(
    make_settings: MakeSettings, with_data: Path
) -> None:
    result = by_name(run_checks(make_settings(), FakeNetwork().client()))["Dataset"]
    assert result.status is Status.WARN
    assert "synthetic fixture" in result.detail
    assert "make download" in result.next_step


def test_dataset_row_ok_when_downloaded(make_settings: MakeSettings, with_data: Path) -> None:
    with_data.mkdir()
    (with_data / "wiki_movie_plots_deduped.csv").write_text("x")
    assert by_name(run_checks(make_settings(), FakeNetwork().client()))["Dataset"].status is Status.OK


def test_dataset_row_fails_when_fixture_is_gone_too(
    make_settings: MakeSettings, with_data: Path, tmp_path: Path
) -> None:
    s = make_settings(DATA__FIXTURE_PATH=str(tmp_path / "nope.csv"))
    assert by_name(run_checks(s, FakeNetwork().client()))["Dataset"].status is Status.FAIL


# --- all credentials present and valid ---------------------------------------------------------------


def test_everything_ok_with_valid_credentials(make_settings: MakeSettings, with_data: Path) -> None:
    net = FakeNetwork()
    results = by_name(run_checks(make_settings(LANGSMITH_TRACING="true", **ALL_CREDS), net.client()))
    for name in ("Docker", "Qdrant", "Nebius API key", "Model: chat", "Model: judge", "LangSmith", "Kaggle"):
        assert results[name].status is Status.OK, name
    assert results["Docker"].detail == "daemon 29.0.1"
    hosts = {r.url.host for r in net.requests}
    assert hosts == {"localhost", "api.tokenfactory.nebius.com", "api.smith.langchain.com", "www.kaggle.com"}


def test_authenticated_calls_use_the_documented_endpoints_and_headers(
    make_settings: MakeSettings, with_data: Path
) -> None:
    net = FakeNetwork()
    run_checks(make_settings(**ALL_CREDS), net.client())
    requests = {r.url.host: r for r in net.requests}
    assert requests["localhost"].url.path == "/readyz"
    nebius = requests["api.tokenfactory.nebius.com"]
    assert nebius.url.path == "/v1/models"
    assert nebius.headers["Authorization"] == "Bearer fake-nebius-value"
    assert requests["api.smith.langchain.com"].headers["x-api-key"] == "fake-ls-value"
    assert requests["www.kaggle.com"].url.path == "/api/v1/datasets/view/jrobischon/wikipedia-movie-plots"
    assert requests["www.kaggle.com"].headers["Authorization"].startswith("Basic ")


def test_langsmith_ok_but_tracing_switch_off_suggests_enabling_it(make_settings: MakeSettings, with_data: Path) -> None:
    result = by_name(run_checks(make_settings(LANGSMITH_API_KEY="fake-ls-value"), FakeNetwork().client()))["LangSmith"]
    assert result.status is Status.OK
    assert "LANGSMITH_TRACING=true" in result.next_step


# --- failures ------------------------------------------------------------------------------------------


def test_unreachable_qdrant_says_how_to_start_it(make_settings: MakeSettings, with_data: Path) -> None:
    result = by_name(run_checks(make_settings(), FakeNetwork(localhost=refuse).client()))["Qdrant"]
    assert result.status is Status.FAIL
    assert "make up" in result.next_step


def test_qdrant_not_ready(make_settings: MakeSettings, with_data: Path) -> None:
    net = FakeNetwork(localhost=lambda r: httpx.Response(503))
    result = by_name(run_checks(make_settings(), net.client()))["Qdrant"]
    assert result.status is Status.FAIL
    assert "503" in result.detail


def test_model_ids_are_checked_against_v1_models(make_settings: MakeSettings, with_data: Path) -> None:
    s = make_settings(NEBIUS_API_KEY="fake-nebius-value", LLM__CHAT_MODEL="not/a-real-model")
    results = by_name(run_checks(s, FakeNetwork().client()))
    assert results["Nebius API key"].status is Status.OK
    assert results["Model: chat"].status is Status.FAIL
    assert "not/a-real-model" in results["Model: chat"].detail
    assert "llm.chat_model" in results["Model: chat"].next_step
    assert results["Model: judge"].status is Status.OK


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_nebius_key(make_settings: MakeSettings, with_data: Path, status: int) -> None:
    net = FakeNetwork(**{"api.tokenfactory.nebius.com": lambda r: httpx.Response(status)})
    results = by_name(run_checks(make_settings(NEBIUS_API_KEY="fake-nebius-value"), net.client()))
    assert results["Nebius API key"].status is Status.FAIL
    assert results["Model: chat"].status is Status.SKIPPED


def test_nebius_server_error(make_settings: MakeSettings, with_data: Path) -> None:
    net = FakeNetwork(**{"api.tokenfactory.nebius.com": lambda r: httpx.Response(500)})
    results = run_checks(make_settings(NEBIUS_API_KEY="fake-nebius-value"), net.client())
    assert by_name(results)["Nebius API key"].status is Status.FAIL


def test_nebius_unreachable(make_settings: MakeSettings, with_data: Path) -> None:
    net = FakeNetwork(**{"api.tokenfactory.nebius.com": refuse})
    results = by_name(run_checks(make_settings(NEBIUS_API_KEY="fake-nebius-value"), net.client()))
    assert results["Nebius API key"].status is Status.FAIL
    assert results["Model: judge"].status is Status.SKIPPED


@pytest.mark.parametrize(("host", "name"), [("api.smith.langchain.com", "LangSmith"), ("www.kaggle.com", "Kaggle")])
@pytest.mark.parametrize("status", [401, 500])
def test_rejected_or_failing_service(
    make_settings: MakeSettings, with_data: Path, host: str, name: str, status: int
) -> None:
    net = FakeNetwork(**{host: lambda r: httpx.Response(status)})
    result = by_name(run_checks(make_settings(**ALL_CREDS), net.client()))[name]
    assert result.status is Status.FAIL
    assert str(status) in result.detail


@pytest.mark.parametrize(("host", "name"), [("api.smith.langchain.com", "LangSmith"), ("www.kaggle.com", "Kaggle")])
def test_unreachable_service(make_settings: MakeSettings, with_data: Path, host: str, name: str) -> None:
    net = FakeNetwork(**{host: refuse})
    result = by_name(run_checks(make_settings(**ALL_CREDS), net.client()))[name]
    assert result.status is Status.FAIL
    assert "ConnectError" in result.detail


def test_kaggle_with_only_one_half_names_the_missing_variable(make_settings: MakeSettings, with_data: Path) -> None:
    result = by_name(run_checks(make_settings(KAGGLE_USERNAME="someone"), FakeNetwork().client()))["Kaggle"]
    assert result.status is Status.MISSING
    assert result.detail == "KAGGLE_KEY not set"


# --- docker --------------------------------------------------------------------------------------------


def test_docker_missing(make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    result = doctor.check_docker(make_settings())
    assert result.status is Status.FAIL
    assert "QDRANT_URL" in result.next_step


def test_docker_daemon_down(make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
    )
    result = doctor.check_docker(make_settings())
    assert result.status is Status.FAIL
    assert "make up" in result.next_step


def test_docker_timeout(make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=1)

    monkeypatch.setattr(doctor.subprocess, "run", hang)
    assert doctor.check_docker(make_settings()).status is Status.FAIL


# --- robustness and rendering --------------------------------------------------------------------------


def test_a_crashing_check_becomes_a_fail_row_and_never_leaks_the_secret(
    make_settings: MakeSettings, with_data: Path
) -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom with fake-nebius-value inside")

    net = FakeNetwork(**{"api.tokenfactory.nebius.com": explode})
    results = run_checks(make_settings(NEBIUS_API_KEY="fake-nebius-value"), net.client())
    row = by_name(results)["Nebius API key"]
    assert row.status is Status.FAIL
    assert "check crashed" in row.detail
    assert "fake-nebius-value" not in row.model_dump_json()
    assert by_name(results)["Kaggle"].status is Status.MISSING  # later checks still ran


def test_run_checks_creates_and_closes_its_own_client(
    make_settings: MakeSettings, with_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    net = FakeNetwork()
    real_client = httpx.Client
    created: list[httpx.Client] = []

    def factory(**kwargs: Any) -> httpx.Client:
        assert kwargs["timeout"] == 5
        client = real_client(transport=httpx.MockTransport(net))
        created.append(client)
        return client

    monkeypatch.setattr(doctor.httpx, "Client", factory)
    run_checks(make_settings())
    assert created[0].is_closed


def test_render_table_aligns_columns_and_summarises() -> None:
    table = render_table(
        [
            CheckResult(name="Docker", status=Status.OK, detail="daemon 1"),
            CheckResult(name="Kaggle", status=Status.MISSING, detail="KAGGLE_KEY not set", next_step="set it"),
        ]
    )
    lines = table.splitlines()
    assert lines[0].split() == ["CHECK", "STATUS", "DETAIL", "NEXT", "STEP"]
    assert set(lines[1].replace(" ", "")) == {"-"}
    assert lines[3].startswith("Kaggle")
    assert "MISSING" in lines[3]
    assert lines[-1].startswith("1 ok, 1 need attention")


def test_main_always_returns_zero_and_prints_the_table(
    make_settings: MakeSettings, with_data: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(doctor, "load_settings", lambda: make_settings())
    assert doctor.main(client=FakeNetwork(localhost=refuse).client()) == 0
    out = capsys.readouterr().out
    assert "CHECK" in out
    assert "MISSING" in out
    assert "Traceback" not in out


def test_main_returns_zero_even_when_configuration_is_broken(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> Settings:
        raise ValueError("bad yaml")

    monkeypatch.setattr(doctor, "load_settings", broken)
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert "could not load the configuration" in out
    assert "Traceback" not in out


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_module_entry_point_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import runpy

    real_client = httpx.Client
    net = FakeNetwork(localhost=refuse)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(net)))
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("movie_rag.doctor", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
    assert "CHECK" in capsys.readouterr().out
