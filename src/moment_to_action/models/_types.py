from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from pathlib import Path


__all__ = [
    "DownloadSource",
    "ModelID",
    "ModelInfo",
    "ModelSource",
    "ModelStatus",
    "VendoredSource",
]


class ModelID(Enum):
    """Unique identifier for each model in the registry."""

    YOLO_V8 = "yolo_v8"
    MOBILECLIP_S2 = "mobileclip_s2"


@attrs.frozen
class VendoredSource:
    """Model files included directly in the repository.

    Attributes:
        subdir: Subdirectory under _vendored/ where model files are stored.
    """

    subdir: str


@attrs.frozen
class DownloadSource:
    """Model files sourced from HuggingFace Hub.

    Attributes:
        hf_repo_id: HuggingFace Hub repository identifier (e.g. 'user/repo').
        hf_filename: Filename within the HuggingFace repo to download.
    """

    hf_repo_id: str
    hf_filename: str


type ModelSource = VendoredSource | DownloadSource


@attrs.frozen
class ModelInfo:
    """Static metadata describing a model in the registry.

    Attributes:
        id: Unique model identifier.
        filename: Expected filename after acquisition (ONNX, TFLite, etc.).
        source: Where to load the model from (vendored or HuggingFace).
    """

    id: ModelID
    filename: str
    source: ModelSource


@attrs.frozen
class ModelStatus:
    """Runtime status of a loaded or loadable model.

    Attributes:
        info: Static metadata for this model.
        available: Whether the model file is present and accessible.
        path: Full path to the model file if available; None otherwise.
        size_bytes: Size of the model file in bytes if available; None
            otherwise.
    """

    info: ModelInfo
    available: bool
    path: Path | None
    size_bytes: int | None
