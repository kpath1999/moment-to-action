"""Web utility functions."""

from __future__ import annotations

import socket
from typing import IO, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pathlib import Path
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def pick_free_port() -> int:
    """Return an unused TCP port on localhost.

    Binds a socket to port 0 (OS assigns a free port), reads the assigned
    port number, and closes the socket before returning.

    Returns:
        An available TCP port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stream_with_progress(
    res: httpx.Response,
    dest: IO[bytes],
    progress: Progress,
    description: str,
    *,
    total: int | None = None,
) -> int:
    """Stream an HTTP response to a destination file-like object, with progress tracking.

    Args:
        res: The HTTP response to stream. Must be a streaming response.
        dest: The file-like object to write the response content to.
        progress: A Rich Progress instance to track download progress.
        description: Description for the progress task.
        total: Total size of the content in bytes, if known. If None, will attempt to
            infer from the Content-Length header or proceed without a total.

    Returns:
        The total number of bytes written to the destination.
    """
    if total is None:
        total = int(res.headers.get("Content-Length", 0))

    task_id = progress.add_task(description, total=total)
    total_written = 0

    for chunk in res.iter_bytes():
        dest.write(chunk)
        progress.update(task_id, advance=len(chunk))

        total_written += len(chunk)

    return total_written


def download_file(
    url: str,
    dest_path: Path,
    *,
    show_progress: bool = True,
    total: int | None = None,
    headers: dict[str, str] | None = None,
) -> int:
    """Download a file from a URL to a local path, with optional progress display.

    Args:
        url: The URL to download the file from.
        dest_path: The local path to save the downloaded file to.
        show_progress: Whether to display download progress using Rich.
        total: Total size of the content in bytes, if known. If None, will attempt to
            infer from the Content-Length header or proceed without a total.
        headers: Optional HTTP headers to include in the request (e.g. Authorization).

    Returns:
        The total number of bytes downloaded.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        OSError: If writing to the destination path fails.
    """
    with (
        httpx.stream("GET", url, headers=headers or {}, follow_redirects=True) as res,
        dest_path.open("wb") as f,
    ):
        res.raise_for_status()

        # Fast path: just download directly
        if not show_progress:
            total_written = 0

            for chunk in res.iter_bytes():
                f.write(chunk)
                total_written += len(chunk)

            return total_written

        # Rich progress tracking path
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            return stream_with_progress(
                res, f, progress, f"Downloading {dest_path.name}", total=total
            )
