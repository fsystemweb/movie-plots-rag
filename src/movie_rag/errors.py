"""Project exceptions.

Anything a human can fix by changing their setup (a missing key, a missing file) gets a dedicated error whose message
says what to do next, so CLIs can print it without a stack trace.
"""

from __future__ import annotations

CREDENTIALS_DOC = "docs/CREDENTIALS.md"


class MovieRagError(Exception):
    """Base class for expected, user-facing failures."""


class MissingCredentialError(MovieRagError):
    """A credential needed by the requested operation is not configured.

    Raised at call time, never at import. ``str(error)`` is the documented hint, for example
    ``set NEBIUS_API_KEY — see docs/CREDENTIALS.md``.
    """

    def __init__(self, *env_vars: str) -> None:
        if not env_vars:
            raise ValueError("MissingCredentialError needs at least one environment variable name")
        self.env_vars: tuple[str, ...] = env_vars
        super().__init__(f"set {' and '.join(env_vars)} — see {CREDENTIALS_DOC}")


class DataError(MovieRagError):
    """The dataset on disk or in a download is missing, malformed or unsafe to read."""


class DownloadError(MovieRagError):
    """The dataset download failed for a reason other than missing credentials."""


class IndexingError(MovieRagError):
    """The vector index (Qdrant) is incompatible with the configuration or disagrees with what was written."""


class RetrievalError(MovieRagError):
    """A search request is invalid (empty query, bad ``top_k``, unknown mode)."""


class MovieNotFoundError(MovieRagError):
    """No film matches the given id or title."""


class AgentError(MovieRagError):
    """The agent could not run for a reason the user can act on (bad prompt version, unusable model output)."""


class McpUnavailableError(AgentError):
    """The MCP server the agent gets its tools from cannot be reached (not started, wrong ``mcp.url``)."""

    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__(
            f"The movie MCP server at {url} is unreachable: start it with `make serve` "
            "(after `make up` and `make ingest`) and retry."
        )


class EvalSetError(MovieRagError):
    """The evaluation question file is malformed or disagrees with the dataset it is checked against."""
