"""Per-backend artifact resolver for DLC variants.

Resolves the best inference artifact for a given compute unit: a per-backend
context binary (``model.<unit>.bin``) when present, falling back to the
portable ``model.dlc``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from moment_to_action.hardware._types import ComputeUnit

if TYPE_CHECKING:
    from pathlib import Path

# Map each compute unit to its preferred context-binary filename.
_BIN_BY_UNIT: dict[ComputeUnit, str] = {
    ComputeUnit.CPU: "model.cpu.bin",
    ComputeUnit.GPU: "model.gpu.bin",
    ComputeUnit.NPU: "model.npu.bin",
    ComputeUnit.DSP: "model.npu.bin",
}


def resolve_backend_artifact(variant_dir: Path, unit: ComputeUnit) -> Path:
    """Return the best artifact path for the given compute unit.

    Checks for a per-backend context binary first (e.g. ``model.npu.bin``);
    falls back to the portable ``model.dlc`` if the binary is absent.

    Context binaries load in milliseconds (AOT compiled); the ``model.dlc``
    fallback is always correct but HTP graph-prepare can take seconds per load.

    Args:
        variant_dir: Directory containing the model variant files.
        unit: Preferred compute unit.

    Returns:
        Path to the artifact file (either a context binary or ``model.dlc``).

    Raises:
        FileNotFoundError: If neither the context binary nor ``model.dlc`` exists.
    """
    cand = variant_dir / _BIN_BY_UNIT[unit]
    if cand.exists():
        return cand
    dlc = variant_dir / "model.dlc"
    if dlc.exists():
        return dlc
    msg = f"No artifact found in {variant_dir}: tried {cand.name} and model.dlc"
    raise FileNotFoundError(msg)
