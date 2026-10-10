"""The handoff documents exist, agree with the code they describe, and only mention real commands."""

from __future__ import annotations

import re

from movie_rag.config import PROJECT_ROOT, Settings, load_settings
from movie_rag.errors import CREDENTIALS_DOC, MissingCredentialError

DOCS = PROJECT_ROOT / "docs"
CREDENTIALS = PROJECT_ROOT / CREDENTIALS_DOC
DEMO = DOCS / "DEMO.md"
FULL_RESULTS_SEQUENCE = (
    "make download && make ingest && make eval MODE=dense && make eval MODE=sparse && make eval MODE=hybrid "
    "&& make report"
)
SECRET_FIELDS = ("nebius_api_key", "langsmith_api_key", "kaggle_key", "kaggle_username")
SWITCH_FIELDS = ("langsmith_tracing", "langsmith_project", "nebius_base_url")


def _make_targets() -> set[str]:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    phony = re.search(r"^\.PHONY:(.*?)\n\n", makefile, flags=re.MULTILINE | re.DOTALL)
    assert phony is not None
    return set(phony.group(1).replace("\\", " ").split())


def _referenced_targets(text: str) -> set[str]:
    return set(re.findall(r"\bmake ([a-z][a-z-]*)\b", text))


def test_the_file_named_in_every_credential_hint_exists() -> None:
    assert CREDENTIALS_DOC == "docs/CREDENTIALS.md" and CREDENTIALS.is_file()
    assert CREDENTIALS_DOC in str(MissingCredentialError("NEBIUS_API_KEY"))
    referenced = {
        name
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        for name in re.findall(r"docs/([A-Z_]+\.md)", path.read_text(encoding="utf-8"))
    }
    assert "CREDENTIALS.md" in referenced
    assert referenced <= {p.name for p in DOCS.glob("*.md")}


def test_the_credentials_guide_names_every_variable_the_settings_read() -> None:
    text = CREDENTIALS.read_text(encoding="utf-8")
    for field in (*SECRET_FIELDS, *SWITCH_FIELDS):
        assert field in Settings.model_fields
        assert field.upper() in text, field.upper()
    for secret in ("NEBIUS_API_KEY", "LANGSMITH_API_KEY", "DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"):
        assert secret in text


def test_the_credentials_guide_gives_the_exact_full_results_sequence() -> None:
    text = CREDENTIALS.read_text(encoding="utf-8")
    assert FULL_RESULTS_SEQUENCE in text
    for step in ("cp .env.example .env", "make doctor", "uv sync --extra eval"):
        assert step in text
    assert "PR-06 QA m2" in text  # the priority item before the first live run


def test_the_credentials_guide_quotes_the_configured_models_and_endpoint() -> None:
    settings = load_settings(env_file=None)
    text = CREDENTIALS.read_text(encoding="utf-8")
    for configured in (
        settings.llm.chat_model,
        settings.llm.judge_model,
        settings.llm.base_url,
        settings.eval.collection,
    ):
        assert configured in text
    assert settings.eval.collection != settings.qdrant.collection


def test_every_make_target_in_the_handoff_docs_exists() -> None:
    targets = _make_targets()
    for path in (CREDENTIALS, DEMO):
        mentioned = _referenced_targets(path.read_text(encoding="utf-8"))
        assert mentioned, path
        assert mentioned <= targets, f"{path.name}: {sorted(mentioned - targets)}"


def test_the_demo_uses_the_pages_example_question_and_real_controls() -> None:
    settings = load_settings(env_file=None)
    text = DEMO.read_text(encoding="utf-8")
    assert settings.ui.example_questions[0].replace("**", "") in text.replace("**", "")
    app = (PROJECT_ROOT / "src" / "movie_rag" / "ui" / "app.py").read_text(encoding="utf-8")
    for control in ("Compare modes", "Retrieved films", "Agent steps"):
        assert control in app and control in text
    assert "Retrieval only" in text and "Retrieval only" in app


def test_the_readme_links_the_live_guides() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("CREDENTIALS.md", "SOLUTION_OVERVIEW.md", "DEMO.md"):
        assert f"docs/{name}" in readme
        assert (DOCS / name).is_file()
    assert "coming in PR-11" not in readme
