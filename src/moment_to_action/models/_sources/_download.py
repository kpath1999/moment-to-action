from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from moment_to_action.utils.web import download_file

if TYPE_CHECKING:
    from pathlib import Path


@attrs.frozen
class DownloadSource:
    """Model files downloaded from a URL.

    Currently, only supports single-file downloads.
    """

    url: str
    """URL to download the model file from."""

    filename: str
    """Filename to save the downloaded model as."""


def resolve_download_source(
    source: DownloadSource, variant_dir: Path, *, download: bool = False, progress: bool = True
) -> Path | None:
    """Resolve the path of a download source to an absolute path on disk.

    If `download` is True, will attempt to download the file if it does not already exist.
    Otherwise, it will return None.

    Args:
        source: The DownloadSource to resolve.
        variant_dir: The directory where the model variant files are stored.
        download: Whether to attempt downloading the file if it does not exist.
        progress: Whether to display download progress.

    Returns:
        The path to the model file if it exists or was downloaded successfully,
        or None if it does not exist and download is False.
    """
    # Check if we need to download the file
    target_path = variant_dir / source.filename
    if not target_path.exists():
        # Not allowed to download
        if not download:
            return None

        # Run the download
        download_file(source.url, target_path, show_progress=progress)

    # At this point, the file should exist (either it already existed or we just downloaded it)
    if not target_path.exists():
        msg = f"Failed to download {source.filename}"
        raise RuntimeError(msg)

    return target_path
