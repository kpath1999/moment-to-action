"""Shared runtime backends — platform-agnostic inference wrappers."""

from __future__ import annotations

from ._litert import LiteRTBackend
from ._onnx import ONNXBackend

__all__ = ["LiteRTBackend", "ONNXBackend"]
