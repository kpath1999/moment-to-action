from pathlib import Path
from typing import TypeAlias

from ._download import DownloadSource, resolve_download_source
from ._hugging_face import HuggingFaceSource, resolve_hugging_face_source
from ._ultralytics import UltralyticsSource, resolve_ultralytics_source
from ._vendored import VendoredSource, resolve_vendored_source

# Type alias for all possible model sources
ModelSource: TypeAlias = DownloadSource | HuggingFaceSource | UltralyticsSource | VendoredSource


# Function to resolve any model source to a local file path
def resolve_model_source(
    source: ModelSource, variant_dir: Path, *, download: bool = False, progress: bool = True
) -> Path | None:
    """Resolve a ModelSource to a local file path.

    If download is True, if possible, the source will be downloaded if it is not already
    available locally. Otherwise, None will be returned.

    Args:
        source: The ModelSource to resolve.
        variant_dir: The directory where the resolved model file should be located.
        download: Whether to attempt downloading the source if it is not available locally.
        progress: Whether to show a progress bar during download (if applicable).

    Returns:
        A Path to the resolved model file, or None if it cannot be resolved.
    """
    match source:
        case DownloadSource():
            return resolve_download_source(
                source, variant_dir, download=download, progress=progress
            )
        case HuggingFaceSource():
            return resolve_hugging_face_source(
                source, variant_dir, download=download, progress=progress
            )
        case UltralyticsSource():
            return resolve_ultralytics_source(
                source, variant_dir, download=download, progress=progress
            )
        case VendoredSource():
            return resolve_vendored_source(source)
        case _:
            msg = f"Unsupported ModelSource type: {type(source)}"
            raise ValueError(msg)


__all__ = [
    "DownloadSource",
    "HuggingFaceSource",
    "ModelSource",
    "UltralyticsSource",
    "VendoredSource",
    "resolve_download_source",
    "resolve_hugging_face_source",
    "resolve_model_source",
    "resolve_ultralytics_source",
    "resolve_vendored_source",
]
