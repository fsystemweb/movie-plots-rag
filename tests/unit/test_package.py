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
    [("demo", "PR-04"), ("serve", "PR-05"), ("eval-smoke", "PR-09")],
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
