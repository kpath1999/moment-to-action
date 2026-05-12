from pathlib import Path

import attrs
from huggingface_hub import get_hf_file_metadata, hf_hub_url

from moment_to_action.models._formats import ModelFormat
from moment_to_action.utils.web import download_file


@attrs.frozen
class HuggingFaceSource:
    """Model files sourced from HuggingFace Hub."""

    format: ModelFormat
    """Format of the model file (e.g. ONNX, TFLite)."""

    hf_repo_id: str
    """HuggingFace Hub repository identifier (e.g. 'user/repo')."""

    files: list[str]
    """Filenames within the HuggingFace repo to download."""

    revision: str
    """Revision of the HuggingFace repo to download from (i.e., commit hash).

    Must be specified for repoducible downloads.
    """


def resolve_hugging_face_source(
    source: HuggingFaceSource, variant_dir: Path, *, download: bool = False, progress: bool = True
) -> Path | None:
    """Resolve the path of a HuggingFace source to an absolute path on disk.

    If `download` is True, will attempt to download the files if they do not already exist.
    Otherwise, it will return None if any files are missing.

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

        # Run downloads
        # TODO(nikola): This could easily be parallelized if there are multiple files
        for filename in missing_files:
            # Get the URL for the file in the HuggingFace repo
            url = hf_hub_url(
                repo_id=source.hf_repo_id,
                filename=filename,
                revision=source.revision,
            )

            # Get the expected file size for progress tracking
            metadata = get_hf_file_metadata(url)
            file_size = metadata.size

            # Download the file
            target_path = variant_dir / filename
            download_file(url, target_path, show_progress=progress, total=file_size)

    # All files should now exist (either they already existed or we just downloaded them)
    for filename in source.files:
        if not (variant_dir / filename).exists():
            msg = f"Failed to download {filename} from HuggingFace repo {source.hf_repo_id}"
            raise RuntimeError(msg)

    return variant_dir
