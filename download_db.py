"""Download the Chinook SQLite database used by the CLI."""

from __future__ import annotations

import os
import ssl
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import certifi

DATABASE_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "Chinook.db"


def _make_read_only(path: Path) -> None:
    """Remove write bits from the database file for its current permission set."""
    path.chmod(path.stat().st_mode & ~0o222)


def download_db(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Download the database if it is missing and return its path.

    The download is written to a temporary file and atomically moved into
    place, so an interrupted request cannot leave a partial database behind.
    """
    destination = Path(db_path)
    if destination.is_file():
        _make_read_only(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".download",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            tls_context = ssl.create_default_context(cafile=certifi.where())
            with urlopen(DATABASE_URL, timeout=60, context=tls_context) as response:
                shutil.copyfileobj(response, temporary_file)

        if temporary_path.stat().st_size == 0:
            raise RuntimeError("The database download was empty.")
        os.replace(temporary_path, destination)
        _make_read_only(destination)
        return destination
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
