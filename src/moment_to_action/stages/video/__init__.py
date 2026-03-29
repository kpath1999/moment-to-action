"""Video pipeline stages — preprocessing, object detection, and clip buffering."""

from __future__ import annotations

__all__ = [
    "ClipBufferStage",
    "ImagePreprocessConfig",
    "ImagePreprocessor",
    "PreprocessorStage",
    "ProcessedFrame",
    "YOLOStage",
]


def __getattr__(name: str) -> object:
    """Load video stages lazily."""
    if name == "ClipBufferStage":
        from ._clip_buffer import ClipBufferStage

        return ClipBufferStage

    if name in {"ImagePreprocessConfig", "ImagePreprocessor", "PreprocessorStage", "ProcessedFrame"}:
        from ._preprocess import (
            ImagePreprocessConfig,
            ImagePreprocessor,
            PreprocessorStage,
            ProcessedFrame,
        )

        return {
            "ImagePreprocessConfig": ImagePreprocessConfig,
            "ImagePreprocessor": ImagePreprocessor,
            "PreprocessorStage": PreprocessorStage,
            "ProcessedFrame": ProcessedFrame,
        }[name]

    if name == "YOLOStage":
        from ._yolo import YOLOStage

        return YOLOStage

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
