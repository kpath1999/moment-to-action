"""Model management — discovery, caching, and resolution.

The ModelManager provides a unified interface for working with ML models,
supporting both vendored models (shipped with the package) and downloaded
models cached in the user's cache directory via HuggingFace Hub.

Example::

    from moment_to_action.models import ModelManager, ModelID

    manager = ModelManager()
    yolo_path = manager.get_path(ModelID.YOLO_V8)

    # Check availability
    if manager.is_available(ModelID.MOBILECLIP_S2):
        clip_path = manager.get_path(ModelID.MOBILECLIP_S2)

    # List all models
    for status in manager.list_models():
        print(f"{status.info.id}: {status.available}")

    # Clear cache
    bytes_freed, removed = manager.clear_cache()
"""

from __future__ import annotations

from ._manager import ModelManager
from ._types import (
    DownloadSource,
    ModelID,
    ModelInfo,
    ModelSource,
    ModelStatus,
    VendoredSource,
)

__all__ = [
    "DownloadSource",
    "ModelID",
    "ModelInfo",
    "ModelManager",
    "ModelSource",
    "ModelStatus",
    "VendoredSource",
]
