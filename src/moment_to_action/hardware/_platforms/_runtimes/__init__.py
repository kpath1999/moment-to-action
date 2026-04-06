"""Shared runtime backends — platform-agnostic inference wrappers."""

from __future__ import annotations

from ._litert import LiteRTBackend
from ._onnx import ONNXBackend
from ._torch_policy import resolve_torch_execution_policy

__all__ = ["LiteRTBackend", "ONNXBackend", "resolve_torch_execution_policy"]
