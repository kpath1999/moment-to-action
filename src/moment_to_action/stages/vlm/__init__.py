"""VLM pipeline stages — vision-language model classification and video description."""

from __future__ import annotations

from ._smolvlm2 import SmolVLM2Stage

__all__ = ["MobileCLIPStage", "SmolVLM2Stage"]


def __getattr__(name: str) -> object:
    """Load optional VLM stages lazily."""
    if name != "MobileCLIPStage":
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    try:
        from ._mobileclip import MobileCLIPStage
    except ModuleNotFoundError as exc:
        if exc.name != "open_clip":
            raise
        msg = "MobileCLIPStage requires the optional dependency open-clip-torch."
        raise ModuleNotFoundError(msg) from exc

    return MobileCLIPStage
