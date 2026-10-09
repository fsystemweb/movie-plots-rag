"""Movie Plots RAG: a movie-discovery agent that answers fuzzy plot questions with cited results."""

from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Return the installed package version, or ``"unknown"`` when not installed."""
    try:
        return version("movie-rag")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

__all__ = ["__version__", "get_version"]
