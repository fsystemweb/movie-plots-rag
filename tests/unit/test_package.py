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


@pytest.mark.parametrize(
    ("target", "pr"),
    [("eval-smoke", "PR-09")],
)
def test_unimplemented_make_targets_are_stubs(target: str, pr: str) -> None:
    out = subprocess.run(["make", "-s", target], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert out.strip() == f"not implemented yet ({pr})"


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
