"""LiteRT backend for the x86_64 platform.

Currently identical to the shared base implementation (CPU-only). Provided as a
separate subclass so platform-specific configuration can be added in the future
(e.g. CUDA/ROCm support via custom delegates) without modifying shared code.
"""

from __future__ import annotations

from moment_to_action.hardware._platforms._runtimes import LiteRTBackend


class X86_64LiteRTBackend(LiteRTBackend):  # noqa: N801
    """x86_64-specific LiteRT backend (CPU-only, identical to base)."""
