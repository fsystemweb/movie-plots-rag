from __future__ import annotations

import pytest

from movie_rag.errors import CREDENTIALS_DOC, DataError, DownloadError, MissingCredentialError, MovieRagError


def test_missing_credential_message_is_the_documented_hint() -> None:
    error = MissingCredentialError("NEBIUS_API_KEY")
    assert str(error) == "set NEBIUS_API_KEY — see docs/CREDENTIALS.md"
    assert error.env_vars == ("NEBIUS_API_KEY",)
    assert CREDENTIALS_DOC == "docs/CREDENTIALS.md"


def test_missing_credential_names_every_variable() -> None:
    assert str(MissingCredentialError("KAGGLE_USERNAME", "KAGGLE_KEY")) == (
        "set KAGGLE_USERNAME and KAGGLE_KEY — see docs/CREDENTIALS.md"
    )


def test_missing_credential_requires_a_variable_name() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MissingCredentialError()


@pytest.mark.parametrize("error", [MissingCredentialError("X"), DataError("x"), DownloadError("x")])
def test_all_expected_failures_share_a_base_class(error: Exception) -> None:
    assert isinstance(error, MovieRagError)
