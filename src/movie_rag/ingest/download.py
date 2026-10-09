"""Download the Kaggle "Wikipedia Movie Plots" CSV (``make download``).

Needs ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``. Without them the command prints what to set and exits 2 with no stack
trace; development and CI use ``tests/fixtures/movies_sample.csv`` instead (see ``docs/DATASET.md``).
No public no-auth mirror with a verified matching licence and schema was found, so none is used.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import httpx

from movie_rag.config import Settings, load_settings
from movie_rag.errors import DownloadError, MissingCredentialError, MovieRagError

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_MISSING_CREDENTIALS = 2
_CHUNK = 1 << 16
MISSING_CREDENTIALS_HELP = (
    "Create a Kaggle API token at https://www.kaggle.com/settings (API section), put KAGGLE_USERNAME and KAGGLE_KEY\n"
    "in .env (names are listed in .env.example), then run `make download` again.\n"
    "No account needed to develop: tests/fixtures/movies_sample.csv is a synthetic stand-in (see docs/DATASET.md)."
)


def say(message: str) -> None:
    """User-facing CLI output (the one place this module writes to the terminal)."""
    sys.stderr.write(message.rstrip("\n") + "\n")


def raw_csv_path(settings: Settings) -> Path:
    """Where the downloaded CSV lives (``data/`` is gitignored)."""
    return settings.data.resolve(settings.data.raw_dir) / settings.data.csv_name


def _fetch_zip(settings: Settings, client: httpx.Client) -> bytes:
    username, key = settings.require_kaggle_credentials()
    url = f"{settings.data.kaggle_api_url.rstrip('/')}/datasets/download/{settings.data.dataset_slug}"
    limit = settings.data.max_download_mb * 1024 * 1024
    buffer = io.BytesIO()
    try:
        with client.stream("GET", url, auth=(username, key.get_secret_value()), follow_redirects=True) as response:
            if response.status_code in (401, 403):
                raise DownloadError(
                    f"Kaggle refused the request (HTTP {response.status_code}). Check KAGGLE_USERNAME and KAGGLE_KEY "
                    "and that the dataset terms were accepted once in a browser — see docs/CREDENTIALS.md"
                )
            if response.status_code >= 400:
                raise DownloadError(
                    f"Kaggle download failed with HTTP {response.status_code} for {settings.data.dataset_slug}"
                )
            for chunk in response.iter_bytes(_CHUNK):
                buffer.write(chunk)
                if buffer.tell() > limit:
                    raise DownloadError(
                        f"download exceeds the configured limit of {settings.data.max_download_mb} MB; aborting"
                    )
    except httpx.HTTPError as exc:
        raise DownloadError(settings.redact(f"could not reach Kaggle: {type(exc).__name__}: {exc}")) from exc
    return buffer.getvalue()


def _extract_csv(payload: bytes, member_name: str, destination: Path, limit_bytes: int) -> None:
    """Extract exactly one named member (by basename, so a hostile archive cannot write elsewhere)."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise DownloadError("Kaggle returned something that is not a zip archive") from exc
    with archive:
        matches = [i for i in archive.infolist() if Path(i.filename).name == member_name and not i.is_dir()]
        if not matches:
            raise DownloadError(f"{member_name} not found in the downloaded archive")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        written = 0
        try:
            with archive.open(matches[0]) as source, temporary.open("wb") as target:
                while chunk := source.read(_CHUNK):
                    written += len(chunk)
                    if written > limit_bytes:  # the declared size can lie, so count what is actually written
                        raise DownloadError(f"{member_name} is larger than the configured limit; aborting")
                    target.write(chunk)
        except DownloadError:
            temporary.unlink(missing_ok=True)
            raise
        temporary.replace(destination)


def download_dataset(settings: Settings, *, client: httpx.Client | None = None, force: bool = False) -> Path:
    """Download and unpack the dataset CSV; return its path. Idempotent unless ``force``.

    Raises :class:`MissingCredentialError` without Kaggle credentials and :class:`DownloadError` on any other failure.
    """
    settings.require_kaggle_credentials()  # fail before touching the network or the disk
    destination = raw_csv_path(settings)
    if destination.is_file() and not force:
        logger.info("dataset already present at %s (use --force to download again)", destination)
        return destination
    owns_client = client is None
    http = client or httpx.Client(timeout=settings.data.http_timeout_s)
    try:
        payload = _fetch_zip(settings, http)
    finally:
        if owns_client:
            http.close()
    _extract_csv(payload, settings.data.csv_name, destination, settings.data.max_download_mb * 1024 * 1024 * 5)
    logger.info("dataset written to %s", destination)
    return destination


def main(argv: Sequence[str] | None = None, *, client: httpx.Client | None = None) -> int:
    """CLI entry point (``python -m movie_rag.ingest.download``). Returns the process exit code."""
    parser = argparse.ArgumentParser(description="Download the Kaggle Wikipedia Movie Plots dataset.")
    parser.add_argument("--force", action="store_true", help="download again even if the CSV already exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        path = download_dataset(load_settings(), client=client, force=args.force)
    except MissingCredentialError as exc:
        say(f"Kaggle credentials are not configured.\n  {exc}\n{MISSING_CREDENTIALS_HELP}")
        return EXIT_MISSING_CREDENTIALS
    except MovieRagError as exc:
        say(f"download failed: {exc}")
        return EXIT_FAILURE
    say(f"dataset ready: {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
