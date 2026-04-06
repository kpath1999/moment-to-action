"""VLM pipeline stages — vision-language model classification and video description."""

from __future__ import annotations

from ._mobileclip import MobileCLIPStage
from ._smolvlm2 import SmolVLM2Stage

__all__ = ["MobileCLIPStage", "SmolVLM2Stage"]
