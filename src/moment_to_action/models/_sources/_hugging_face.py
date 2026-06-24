from __future__ import annotations

import os
from typing import TYPE_CHECKING

import attrs
from huggingface_hub import get_hf_file_metadata, hf_hub_url

from moment_to_action.utils.web import download_file

if TYPE_CHECKING:
    from pathlib import Path


@attrs.frozen
class HuggingFaceSource:
    """Model files sourced from HuggingFace Hub."""

    hf_repo_id: str
    """HuggingFace Hub repository identifier (e.g. 'user/repo')."""

    files: list[str]
    """File paths relative to `hf_subdir` (or repo root if `hf_subdir` is None) to download.

    The same relative paths are used for local storage under the variant directory,
    preserving directory structure (e.g. ``reference_outputs/inputs.npy``).
    """

    revision: str
    """Revision of the HuggingFace repo to download from (i.e., commit hash).

    Must be specified for repoducible downloads.
    """

    hf_subdir: str | None = None
    """Optional subdirectory prefix within the HuggingFace repo.

    When set, each file in `files` is fetched from ``{hf_subdir}/{file}`` in the repo
    but stored at ``{variant_dir}/{file}`` locally, preserving structure relative to the
    subdirectory root.
    """


def resolve_hugging_face_source(
    source: HuggingFaceSource, variant_dir: Path, *, download: bool = False, progress: bool = True
) -> Path | None:
    """Resolve the path of a HuggingFace source to an absolute path on disk.

    If `download` is True, will attempt to download the files if they do not already exist.
    Otherwise, it will return None if any files are missing.

    Each file in ``source.files`` is a path relative to ``source.hf_subdir`` (or the repo
    root). Files are stored locally under ``variant_dir`` with the same relative structure.

    Reads ``HF_TOKEN`` from the environment for authenticating against private repositories.

    Args:
        source: The HuggingFaceSource to resolve.
        variant_dir: The directory where the model variant files are stored.
        download: Whether to attempt downloading the files if they do not exist.
        progress: Whether to display download progress.

    Returns:
        The path to the directory containing the model files if all files exist or were downloaded
        successfully, or None if any files are missing and download is False.
    """
    # Check if there is anything missing
    missing_files = {f for f in source.files if not (variant_dir / f).exists()}

    # Do the download if needed
    if missing_files:
        # Not allowed to download
        if not download:
            return None

        token = os.environ.get("HF_TOKEN")
        auth_headers = {"Authorization": f"Bearer {token}"} if token else {}

        # Run downloads
        # TODO(#102): This could easily be parallelized if there are multiple files
        for filename in missing_files:
            # Build the HF repo path: prepend subdir if configured
            hf_path = f"{source.hf_subdir}/{filename}" if source.hf_subdir else filename

            # Get the URL for the file in the HuggingFace repo
            url = hf_hub_url(
                repo_id=source.hf_repo_id,
                filename=hf_path,
                revision=source.revision,
            )

            # Get the expected file size for progress tracking
            metadata = get_hf_file_metadata(url, token=token)
            file_size = metadata.size

            # Store at the relative path under variant_dir, creating parent dirs as needed
            target_path = variant_dir / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            download_file(
                url, target_path, show_progress=progress, total=file_size, headers=auth_headers
            )

    # All files should now exist (either they already existed or we just downloaded them)
    for filename in source.files:
        if not (variant_dir / filename).exists():
            msg = f"Failed to download {filename} from HuggingFace repo {source.hf_repo_id}"
            raise RuntimeError(msg)

    return variant_dir
