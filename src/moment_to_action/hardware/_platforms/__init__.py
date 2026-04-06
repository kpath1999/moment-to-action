"""Platform backends package — ABCs, platform detection, and concrete implementations."""

from __future__ import annotations

from ._base import InferenceBackend, ModelInput, ResourceMonitor
from ._detection import Platform, detect_platform

__all__ = ["InferenceBackend", "ModelInput", "Platform", "ResourceMonitor", "detect_platform"]
