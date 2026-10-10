"""Smoke tests for the package scaffold."""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
import yaml

import movie_rag

ROOT = Path(__file__).resolve().parents[2]


def test_version_is_exposed() -> None:
    assert movie_rag.__version__ == movie_rag.get_version()
    assert movie_rag.__version__ != ""


def test_get_version_falls_back_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(movie_rag, "version", boom)
    assert movie_rag.get_version() == "unknown"


def test_config_skeleton_has_required_sections() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert {"qdrant", "embeddings", "ingest", "retrieval", "llm", "mcp"} <= cfg.keys()
    assert cfg["embeddings"]["dense_dim"] == 384


def test_qdrant_tag_matches_between_compose_and_ci() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    image = compose["services"]["qdrant"]["image"]
    assert image in (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert not image.endswith(":latest")


def _dry_run(*args: str) -> str:
    return subprocess.run(["make", "-n", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def test_make_eval_forwards_the_mode_and_the_llm_switch() -> None:
    assert "python -m movie_rag.eval run --mode hybrid" in _dry_run("eval", "MODE=hybrid")
    plain = _dry_run("eval")
    assert "python -m movie_rag.eval run" in plain and "--mode" not in plain and "--no-llm" not in plain
    assert "--no-llm" in _dry_run("eval", "LLM=0")


def test_make_eval_smoke_has_exactly_the_two_forms_ci_calls() -> None:
    assert _dry_run("eval-smoke").strip().endswith("python -m movie_rag.eval smoke")
    assert "python -m movie_rag.eval smoke --llm" in _dry_run("eval-smoke", "LLM=1")


def test_make_report_renders_the_results_page() -> None:
    assert "python -m movie_rag.eval report" in _dry_run("report")


def test_make_does_not_export_llm_to_the_environment() -> None:
    """``LLM=1`` in the environment would be parsed as the whole ``llm:`` settings section."""
    out = subprocess.run(
        ["make", "-s", "-f", "-", "show", "LLM=1"],
        input=f'include {ROOT / "Makefile"}\nshow:\n\t@echo "[$$LLM]"\n',
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out.strip() == "[]"


def test_make_ui_runs_streamlit_on_the_page_and_forwards_its_options() -> None:
    dry_run = subprocess.run(
        ["make", "-n", "ui", "PORT=8502", "HEADLESS=1"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert "streamlit run src/movie_rag/ui/app.py" in dry_run
    assert "--server.port 8502" in dry_run and "--server.headless true" in dry_run
    plain = subprocess.run(["make", "-n", "ui"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "--server.port" not in plain and "--server.headless" not in plain


def test_make_ingest_runs_the_ingest_module_and_forwards_its_options() -> None:
    dry_run = subprocess.run(
        ["make", "-n", "ingest", "FIXTURE=1", "RECREATE=1", "CSV=data/x.csv"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "python -m movie_rag.ingest" in dry_run
    for flag in ("--fixture", "--recreate", "--csv data/x.csv"):
        assert flag in dry_run
    plain = subprocess.run(["make", "-n", "ingest"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert "--fixture" not in plain and "--recreate" not in plain


def test_make_demo_brings_up_qdrant_ingests_the_fixture_and_queries_in_all_modes() -> None:
    dry_run = subprocess.run(["make", "-n", "demo"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    steps = [line for line in dry_run.splitlines() if line.strip()]
    positions = [
        next(i for i, line in enumerate(steps) if needle in line)
        for needle in ("docker compose up", "movie_rag.ingest --fixture", "movie_rag.retrieval --mode all")
    ]
    assert positions == sorted(positions)
    with_query = subprocess.run(
        ["make", "-n", "demo", "Q=a heist"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert '"a heist"' in with_query


def test_make_serve_runs_the_mcp_server_over_http_stdio_or_compose() -> None:
    def dry_run(*args: str) -> str:
        return subprocess.run(
            ["make", "-n", "serve", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout

    http = dry_run()
    assert "python -m movie_rag.mcp_server" in http and "--transport stdio" not in http
    assert "--transport stdio" in dry_run("STDIO=1")
    assert "docker compose up -d --build --wait mcp-server" in dry_run("DOCKER=1")


def test_the_mcp_server_compose_service_is_healthy_gated_and_consistent_with_the_configuration() -> None:
    from urllib.parse import urlparse

    from movie_rag.config import load_settings

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = compose["services"]["mcp-server"]
    mcp = load_settings(env_file=None).mcp
    assert service["depends_on"]["qdrant"]["condition"] == "service_healthy"
    assert service["environment"]["QDRANT_URL"] == "http://qdrant:6333"
    assert service["environment"]["MCP__HOST"] == "0.0.0.0"  # reachable from outside the container
    assert f"{mcp.port}:{mcp.port}" in service["ports"]
    assert service["healthcheck"]["test"][-1] == "--healthcheck"
    assert urlparse(mcp.url).path == mcp.path and urlparse(mcp.url).port == mcp.port
    assert (ROOT / "Dockerfile").is_file() and "movie_rag.mcp_server" in (ROOT / "Dockerfile").read_text()
