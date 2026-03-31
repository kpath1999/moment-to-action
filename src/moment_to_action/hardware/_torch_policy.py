"""Shared torch device/dtype policy helpers for hardware backends."""

from __future__ import annotations

from moment_to_action.hardware._types import TorchExecutionPolicy


def resolve_torch_execution_policy(requested: str = "auto") -> TorchExecutionPolicy:
    """Resolve torch device and dtype for a requested execution target.

    Args:
        requested: ``"auto"`` or any string accepted by ``torch.device``.

    Returns:
        A resolved :class:`TorchExecutionPolicy`.
    """
    import torch

    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        dtype = "bfloat16"
    elif device.type == "mps":
        dtype = "float16"
    else:
        dtype = "float32"

    return TorchExecutionPolicy(device=str(device), dtype=dtype)
