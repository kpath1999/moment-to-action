"""QAIRT backend string mapping."""

from __future__ import annotations

from moment_to_action.hardware._types import ComputeUnit

# Maps ComputeUnit values to the string QAIRT's initialize() expects.
# DSP shares the HTP path on Qualcomm silicon.
_QAIRT_BACKEND_MAP: dict[ComputeUnit, str] = {
    ComputeUnit.CPU: "CPU",
    ComputeUnit.GPU: "GPU",
    ComputeUnit.NPU: "HTP",
    ComputeUnit.DSP: "HTP",
}


def qairt_backend_for(unit: ComputeUnit) -> str:
    """Return the QAIRT ``initialize()`` backend string for a compute unit.

    Args:
        unit: The desired compute unit.

    Returns:
        A string accepted by ``qairt_model.initialize(backend=...)``.
        Unknown units fall back to ``"CPU"``.
    """
    return _QAIRT_BACKEND_MAP.get(unit, "CPU")
