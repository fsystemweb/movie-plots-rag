"""``make doctor``: a report on what works and what to do next. It is never a gate: the exit code is always 0.

Checks: Docker, Qdrant, the dataset file, and each credential (is it present? and, if so, one cheap authenticated
call), plus the configured model ids against Nebius ``/v1/models`` when a Nebius key exists. A failing or missing item
is a row in the table with a concrete next step; it never aborts the report. Secrets are never printed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum

import httpx
from pydantic import BaseModel

from movie_rag.config import Settings, load_settings
from movie_rag.errors import CREDENTIALS_DOC
from movie_rag.ingest.download import raw_csv_path

logger = logging.getLogger(__name__)

HEADERS = ("CHECK", "STATUS", "DETAIL", "NEXT STEP")
NO_ACTION = "-"


class Status(StrEnum):
    OK = "OK"
    MISSING = "MISSING"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIPPED = "SKIPPED"


class CheckResult(BaseModel):
    name: str
    status: Status
    detail: str
    next_step: str = NO_ACTION


def _result(name: str, status: Status, detail: str, next_step: str = NO_ACTION) -> CheckResult:
    return CheckResult(name=name, status=status, detail=detail, next_step=next_step)


def _join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


# --- individual checks ---------------------------------------------------------------------------------


def check_docker(settings: Settings) -> CheckResult:
    name = "Docker"
    if shutil.which("docker") is None:
        return _result(
            name, Status.FAIL, "docker not found on PATH", "install Docker, or run Qdrant elsewhere and set QDRANT_URL"
        )
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=settings.doctor.docker_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _result(
            name, Status.FAIL, "docker daemon did not answer in time", "start Docker and re-run `make doctor`"
        )
    if completed.returncode != 0:
        return _result(name, Status.FAIL, "docker daemon is not running", "start Docker, then `make up`")
    return _result(name, Status.OK, f"daemon {completed.stdout.strip()}")


def check_qdrant(settings: Settings, client: httpx.Client) -> CheckResult:
    name = "Qdrant"
    url = _join(settings.qdrant.url, "readyz")
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return _result(name, Status.FAIL, f"not reachable at {settings.qdrant.url}", "run `make up` (docker compose)")
    if response.status_code != 200:
        return _result(
            name, Status.FAIL, f"{url} answered HTTP {response.status_code}", "check `docker compose logs qdrant`"
        )
    return _result(name, Status.OK, f"ready at {settings.qdrant.url}")


def check_dataset(settings: Settings) -> CheckResult:
    name = "Dataset"
    path = raw_csv_path(settings)
    if path.is_file():
        return _result(name, Status.OK, f"{path}")
    fixture = settings.data.resolve(settings.data.fixture_path)
    detail = f"{settings.data.raw_dir / settings.data.csv_name} not downloaded"
    if fixture.is_file():
        return _result(
            name,
            Status.WARN,
            f"{detail}; the synthetic fixture is available",
            "after setting Kaggle keys: `make download`",
        )
    return _result(
        name,
        Status.FAIL,
        f"{detail} and the fixture {settings.data.fixture_path} is missing",
        "restore the fixture from git",
    )


def check_nebius(settings: Settings, client: httpx.Client) -> list[CheckResult]:
    chat, judge = settings.llm.chat_model, settings.llm.judge_model
    if settings.nebius_api_key is None:
        hint = f"set NEBIUS_API_KEY — see {CREDENTIALS_DOC}"
        return [
            _result("Nebius API key", Status.MISSING, "NEBIUS_API_KEY is not set (agent and RAGAS need it)", hint),
            _result("Model: chat", Status.SKIPPED, f"{chat} (no key to list models)"),
            _result("Model: judge", Status.SKIPPED, f"{judge} (no key to list models)"),
        ]
    url = _join(settings.llm.base_url, "models")
    headers = {"Authorization": f"Bearer {settings.nebius_api_key.get_secret_value()}"}
    try:
        response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        reason = type(exc).__name__
        skipped = f"({reason}: models not checked)"
        return [
            _result(
                "Nebius API key",
                Status.FAIL,
                f"could not reach {settings.llm.base_url}: {reason}",
                "check network and NEBIUS_BASE_URL",
            ),
            _result("Model: chat", Status.SKIPPED, f"{chat} {skipped}"),
            _result("Model: judge", Status.SKIPPED, f"{judge} {skipped}"),
        ]
    if response.status_code in (401, 403):
        return [
            _result(
                "Nebius API key",
                Status.FAIL,
                f"rejected (HTTP {response.status_code})",
                f"regenerate the key — see {CREDENTIALS_DOC}",
            ),
            _result("Model: chat", Status.SKIPPED, f"{chat} (key rejected)"),
            _result("Model: judge", Status.SKIPPED, f"{judge} (key rejected)"),
        ]
    if response.status_code != 200:
        return [
            _result(
                "Nebius API key",
                Status.FAIL,
                f"/models answered HTTP {response.status_code}",
                "retry later or check NEBIUS_BASE_URL",
            )
        ]
    available = {str(item.get("id")) for item in response.json().get("data", []) if isinstance(item, dict)}
    results = [_result("Nebius API key", Status.OK, f"valid; {len(available)} models listed")]
    for label, model, key in (("chat", chat, "llm.chat_model"), ("judge", judge, "llm.judge_model")):
        if model in available:
            results.append(_result(f"Model: {label}", Status.OK, model))
        else:
            results.append(
                _result(
                    f"Model: {label}",
                    Status.FAIL,
                    f"{model} is not listed by Nebius",
                    f"pick an id from /v1/models in config.yaml ({key})",
                )
            )
    return results


def check_langsmith(settings: Settings, client: httpx.Client) -> CheckResult:
    name = "LangSmith"
    switch = "on" if settings.langsmith_tracing else "off"
    if settings.langsmith_api_key is None:
        return _result(
            name,
            Status.MISSING,
            f"LANGSMITH_API_KEY is not set; tracing is a no-op (switch {switch})",
            f"optional: set LANGSMITH_API_KEY and LANGSMITH_TRACING=true — see {CREDENTIALS_DOC}",
        )
    url = _join(settings.observability.langsmith_api_url, "api/v1/sessions")
    try:
        response = client.get(
            url, params={"limit": 1}, headers={"x-api-key": settings.langsmith_api_key.get_secret_value()}
        )
    except httpx.HTTPError as exc:
        return _result(name, Status.FAIL, f"could not reach LangSmith: {type(exc).__name__}", "check network")
    if response.status_code in (401, 403):
        return _result(
            name,
            Status.FAIL,
            f"key rejected (HTTP {response.status_code})",
            f"regenerate the key — see {CREDENTIALS_DOC}",
        )
    if response.status_code >= 400:
        return _result(name, Status.FAIL, f"answered HTTP {response.status_code}", "retry later")
    tracing_hint = NO_ACTION if settings.langsmith_tracing else "set LANGSMITH_TRACING=true to send traces"
    return _result(name, Status.OK, f"key valid; tracing switch {switch}", tracing_hint)


def check_kaggle(settings: Settings, client: httpx.Client) -> CheckResult:
    name = "Kaggle"
    if settings.kaggle_username is None or settings.kaggle_key is None:
        absent = [
            n
            for n, v in (("KAGGLE_USERNAME", settings.kaggle_username), ("KAGGLE_KEY", settings.kaggle_key))
            if v is None
        ]
        return _result(
            name,
            Status.MISSING,
            f"{' and '.join(absent)} not set",
            f"set them — see {CREDENTIALS_DOC} (the fixture works without)",
        )
    url = _join(settings.data.kaggle_api_url, f"datasets/view/{settings.data.dataset_slug}")
    try:
        response = client.get(url, auth=(settings.kaggle_username, settings.kaggle_key.get_secret_value()))
    except httpx.HTTPError as exc:
        return _result(name, Status.FAIL, f"could not reach Kaggle: {type(exc).__name__}", "check network")
    if response.status_code in (401, 403):
        return _result(
            name,
            Status.FAIL,
            f"credentials rejected (HTTP {response.status_code})",
            f"regenerate the token — see {CREDENTIALS_DOC}",
        )
    if response.status_code >= 400:
        return _result(name, Status.FAIL, f"answered HTTP {response.status_code}", "retry later")
    return _result(name, Status.OK, "credentials valid", "`make download`")


# --- orchestration and rendering -----------------------------------------------------------------------


def _guarded(settings: Settings, name: str, check: Callable[[], list[CheckResult]]) -> list[CheckResult]:
    """Run a check; any unexpected error becomes a FAIL row so the report always completes."""
    try:
        return check()
    except Exception as exc:
        message = settings.redact(f"{type(exc).__name__}: {exc}")
        return [_result(name, Status.FAIL, f"check crashed: {message}", "re-run with logging or report a bug")]


def run_checks(settings: Settings, client: httpx.Client | None = None) -> list[CheckResult]:
    """Run every check. ``client`` is injectable so tests never touch the network."""
    owns_client = client is None
    http = client or httpx.Client(timeout=settings.doctor.http_timeout_s)
    try:
        results: list[CheckResult] = []
        results += _guarded(settings, "Docker", lambda: [check_docker(settings)])
        results += _guarded(settings, "Qdrant", lambda: [check_qdrant(settings, http)])
        results += _guarded(settings, "Dataset", lambda: [check_dataset(settings)])
        results += _guarded(settings, "Nebius API key", lambda: check_nebius(settings, http))
        results += _guarded(settings, "LangSmith", lambda: [check_langsmith(settings, http)])
        results += _guarded(settings, "Kaggle", lambda: [check_kaggle(settings, http)])
        return results
    finally:
        if owns_client:
            http.close()


def render_table(results: Sequence[CheckResult]) -> str:
    """Plain-text table plus a one-line summary."""
    rows = [HEADERS, *[(r.name, r.status.value, r.detail, r.next_step) for r in results]]
    widths = [max(len(row[i]) for row in rows) for i in range(len(HEADERS))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    needs_attention = sum(r.status in (Status.MISSING, Status.FAIL) for r in results)
    ok = sum(r.status is Status.OK for r in results)
    lines.append("")
    lines.append(
        f"{ok} ok, {needs_attention} need attention. Missing credentials are fine for development: use the fixture."
    )
    return "\n".join(lines)


def emit(text: str) -> None:
    """User-facing CLI output."""
    sys.stdout.write(text.rstrip("\n") + "\n")


def main(*, client: httpx.Client | None = None) -> int:
    """Entry point for ``python -m movie_rag.doctor``. Always returns 0: it is a report, not a gate."""
    try:
        settings = load_settings()
    except Exception as exc:
        emit(f"doctor could not load the configuration: {type(exc).__name__}: fix config.yaml / .env and re-run")
        logger.debug("configuration error", exc_info=exc)
        return 0
    emit(render_table(run_checks(settings, client)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
