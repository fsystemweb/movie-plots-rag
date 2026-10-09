from __future__ import annotations

import base64
import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from movie_rag.config import Settings
from movie_rag.errors import DownloadError, MissingCredentialError
from movie_rag.ingest import download
from movie_rag.ingest.download import download_dataset, main, raw_csv_path

MakeSettings = Callable[..., Settings]
CSV_NAME = "wiki_movie_plots_deduped.csv"
CSV_BYTES = b"Release Year,Title\n1999,Example\n"
KAGGLE_KEY = "fake-kaggle-value"
CREDS = {"KAGGLE_USERNAME": "someone", "KAGGLE_KEY": KAGGLE_KEY}


def make_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class Recorder:
    """A mock Kaggle: records requests and answers with the configured response."""

    def __init__(self, response: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._response(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def ok_archive(members: dict[str, bytes] | None = None) -> Recorder:
    payload = make_zip(members or {CSV_NAME: CSV_BYTES})
    return Recorder(lambda request: httpx.Response(200, content=payload))


@pytest.fixture
def settings(make_settings: MakeSettings, tmp_path: Path) -> Settings:
    return make_settings(DATA__RAW_DIR=str(tmp_path / "raw"), **CREDS)


def test_without_credentials_raises_before_any_request(make_settings: MakeSettings, tmp_path: Path) -> None:
    recorder = ok_archive()
    s = make_settings(DATA__RAW_DIR=str(tmp_path / "raw"))
    with pytest.raises(MissingCredentialError) as excinfo:
        download_dataset(s, client=recorder.client())
    assert str(excinfo.value) == "set KAGGLE_USERNAME and KAGGLE_KEY — see docs/CREDENTIALS.md"
    assert recorder.requests == []
    assert not (tmp_path / "raw").exists()


def test_download_writes_the_csv_and_sends_basic_auth(settings: Settings, tmp_path: Path) -> None:
    recorder = ok_archive({f"nested/{CSV_NAME}": CSV_BYTES, "other.txt": b"ignored"})
    path = download_dataset(settings, client=recorder.client())
    assert path == tmp_path / "raw" / CSV_NAME == raw_csv_path(settings)
    assert path.read_bytes() == CSV_BYTES
    assert not list((tmp_path / "raw").glob("*.part"))
    (request,) = recorder.requests
    assert str(request.url) == "https://www.kaggle.com/api/v1/datasets/download/jrobischon/wikipedia-movie-plots"
    expected = "Basic " + base64.b64encode(f"someone:{KAGGLE_KEY}".encode()).decode()
    assert request.headers["Authorization"] == expected


def test_download_is_idempotent_unless_forced(settings: Settings) -> None:
    recorder = ok_archive()
    first = download_dataset(settings, client=recorder.client())
    again = download_dataset(settings, client=recorder.client())
    assert first == again
    assert len(recorder.requests) == 1
    download_dataset(settings, client=recorder.client(), force=True)
    assert len(recorder.requests) == 2


def test_default_client_is_created_and_closed_when_none_is_given(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = ok_archive()
    real_client = httpx.Client
    created: list[httpx.Client] = []

    def factory(**kwargs: object) -> httpx.Client:
        client = real_client(transport=httpx.MockTransport(recorder))
        created.append(client)
        assert kwargs["timeout"] == settings.data.http_timeout_s
        return client

    monkeypatch.setattr(download.httpx, "Client", factory)
    download_dataset(settings)
    assert len(recorder.requests) == 1
    assert created[0].is_closed


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credentials_explain_themselves_without_leaking(settings: Settings, status: int) -> None:
    recorder = Recorder(lambda request: httpx.Response(status))
    with pytest.raises(DownloadError) as excinfo:
        download_dataset(settings, client=recorder.client())
    assert f"HTTP {status}" in str(excinfo.value)
    assert "KAGGLE_USERNAME" in str(excinfo.value)
    assert KAGGLE_KEY not in str(excinfo.value)


def test_other_http_errors_are_reported(settings: Settings) -> None:
    recorder = Recorder(lambda request: httpx.Response(404))
    with pytest.raises(DownloadError, match="HTTP 404"):
        download_dataset(settings, client=recorder.client())


def test_network_failures_become_download_errors_and_are_redacted(settings: Settings) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot connect, token was {KAGGLE_KEY}")

    with pytest.raises(DownloadError) as excinfo:
        download_dataset(settings, client=Recorder(boom).client())
    assert "could not reach Kaggle" in str(excinfo.value)
    assert KAGGLE_KEY not in str(excinfo.value)


def test_oversized_download_is_aborted(make_settings: MakeSettings, tmp_path: Path) -> None:
    s = make_settings(DATA__RAW_DIR=str(tmp_path / "raw"), DATA__MAX_DOWNLOAD_MB="1", **CREDS)
    recorder = Recorder(lambda request: httpx.Response(200, content=b"x" * (2 * 1024 * 1024)))
    with pytest.raises(DownloadError, match="exceeds the configured limit"):
        download_dataset(s, client=recorder.client())


def test_not_a_zip_is_reported(settings: Settings) -> None:
    recorder = Recorder(lambda request: httpx.Response(200, content=b"<html>login</html>"))
    with pytest.raises(DownloadError, match="not a zip"):
        download_dataset(settings, client=recorder.client())


def test_archive_without_the_expected_csv_is_reported(settings: Settings) -> None:
    with pytest.raises(DownloadError, match="not found in the downloaded archive"):
        download_dataset(settings, client=ok_archive({"readme.txt": b"hi"}).client())


def test_hostile_member_paths_cannot_escape_the_target_directory(settings: Settings, tmp_path: Path) -> None:
    recorder = ok_archive({f"../../escape/{CSV_NAME}": CSV_BYTES})
    path = download_dataset(settings, client=recorder.client())
    assert path.parent == tmp_path / "raw"
    assert not (tmp_path.parent / "escape").exists()


def test_decompression_bombs_are_stopped_by_counting_real_bytes(make_settings: MakeSettings, tmp_path: Path) -> None:
    s = make_settings(DATA__RAW_DIR=str(tmp_path / "raw"), DATA__MAX_DOWNLOAD_MB="1", **CREDS)
    bomb = ok_archive({CSV_NAME: b"0" * (6 * 1024 * 1024)})  # small zip, 6 MB inflated, limit is 5 MB
    with pytest.raises(DownloadError, match="larger than the configured limit"):
        download_dataset(s, client=bomb.client())
    assert not list((tmp_path / "raw").glob("*"))  # neither the CSV nor a .part file is left behind


# --- CLI -----------------------------------------------------------------------------------------------


def test_cli_without_credentials_prints_the_hint_and_exits_2_without_a_traceback(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(download, "load_settings", lambda: make_settings())
    assert main([]) == download.EXIT_MISSING_CREDENTIALS == 2
    captured = capsys.readouterr()
    assert "set KAGGLE_USERNAME and KAGGLE_KEY — see docs/CREDENTIALS.md" in captured.err
    assert ".env" in captured.err
    assert "tests/fixtures/movies_sample.csv" in captured.err
    assert "Traceback" not in captured.err + captured.out


def test_cli_success(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(download, "load_settings", lambda: settings)
    assert main([], client=ok_archive().client()) == 0
    assert "dataset ready" in capsys.readouterr().err
    assert (tmp_path / "raw" / CSV_NAME).read_bytes() == CSV_BYTES


def test_cli_force_flag_is_passed_through(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(download, "load_settings", lambda: settings)
    recorder = ok_archive()
    assert main([], client=recorder.client()) == 0
    assert main([], client=recorder.client()) == 0
    assert len(recorder.requests) == 1
    assert main(["--force"], client=recorder.client()) == 0
    assert len(recorder.requests) == 2


def test_cli_download_failure_exits_1_with_a_message(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(download, "load_settings", lambda: settings)
    recorder = Recorder(lambda request: httpx.Response(500))
    assert main([], client=recorder.client()) == download.EXIT_FAILURE
    captured = capsys.readouterr()
    assert "download failed" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_module_entry_point_exits_2_without_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import runpy

    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setattr("sys.argv", ["download"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("movie_rag.ingest.download", run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 2
    assert "docs/CREDENTIALS.md" in capsys.readouterr().err
