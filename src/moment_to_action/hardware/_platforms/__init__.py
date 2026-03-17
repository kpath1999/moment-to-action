"""Platform backends package — ABCs, platform detection, and concrete implementations."""

from __future__ import annotations

from ._base import InferenceBackend, ModelInput, PowerMonitor
from ._detection import Platform, detect_platform

__all__ = ["InferenceBackend", "ModelInput", "Platform", "PowerMonitor", "detect_platform"]
