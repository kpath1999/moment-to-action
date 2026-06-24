"""Ultralytics (YOLOv8, etc.) on-demand ONNX export source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from pathlib import Path


@attrs.frozen
class UltralyticsSource:
    """Model files exported from Ultralytics on demand.

    The model is downloaded and exported to ONNX the first time it is resolved
    with ``download=True``.  Subsequent calls return the cached ONNX without
    re-downloading.

    Requires ``ultralytics`` to be installed (``uv sync --extra yolo-export``).
    """

    name: str
    """Ultralytics model name, e.g. ``"yolov8n"``."""

    filename: str = "model.onnx"
    """Local filename stored under ``variant_dir``."""


def resolve_ultralytics_source(
    source: UltralyticsSource,
    variant_dir: Path,
    *,
    download: bool = False,
    progress: bool = True,  # noqa: ARG001 — kept for interface uniformity
) -> Path | None:
    """Resolve an UltralyticsSource to a local ONNX file.

    Returns the cached path if the file already exists.  When ``download`` is
    ``True`` and the file is missing, exports ``source.name`` to ONNX via
    ``ultralytics.YOLO`` and caches the result.

    Args:
        source: The UltralyticsSource to resolve.
        variant_dir: Directory where the exported ONNX should be cached.
        download: Whether to run the export if the file is missing.
        progress: Unused; present for interface uniformity with other resolvers.

    Returns:
        Path to the cached ONNX file, or ``None`` if ``download`` is ``False``
        and the file does not yet exist.

    Raises:
        ImportError: If ``ultralytics`` is not installed and ``download`` is ``True``.
    """
    from pathlib import Path as _Path  # noqa: PLC0415

    target = variant_dir / source.filename
    if target.exists():
        return target
    if not download:
        return None

    import shutil  # noqa: PLC0415

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        msg = "'ultralytics' is not installed. Install with: uv sync --extra yolo-export"
        raise ImportError(msg) from exc

    yolo = YOLO(f"{source.name}.pt")
    exported_path = _Path(yolo.export(format="onnx", dynamic=False))
    target.parent.mkdir(parents=True, exist_ok=True)
    # shutil.move handles cross-device moves (Path.replace does not).
    shutil.move(str(exported_path), target)
    _Path(f"{source.name}.pt").unlink(missing_ok=True)
    return target
