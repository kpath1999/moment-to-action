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

# Context binaries are HTP-compiled device artifacts; only NPU/DSP can load them.
# CPU and GPU inference uses the portable DLC (no entry → DLC fallback).
_BIN_UNITS: frozenset[ComputeUnit] = frozenset({ComputeUnit.NPU, ComputeUnit.DSP})


def resolve_backend_artifact(variant_dir: Path, unit: ComputeUnit, stem: str = "model") -> Path:
    """Return the best artifact path for the given compute unit.

    Checks for a per-backend context binary first (e.g. ``model.npu.bin``);
    falls back to the portable ``model.dlc`` if the binary is absent.

    Context binaries load in milliseconds (AOT compiled); the DLC fallback is
    always correct but HTP graph-prepare can take seconds per load.

    The ``stem`` parameter selects which artifact family to resolve, so a
    multi-component model can hold several graphs side by side in one variant
    directory.  With the default ``"model"`` stem this resolves
    ``model.npu.bin`` / ``model.dlc`` exactly as before.  The Detectron2
    detector resolves ``model.proposal_generator.*`` and ``model.roi_head.*``.

    Args:
        variant_dir: Directory containing the model variant files.
        unit: Preferred compute unit.
        stem: Artifact filename stem (without the ``.npu.bin`` / ``.dlc``
            suffix).  Defaults to ``"model"``.

    Returns:
        Path to the artifact file (either a context binary or a DLC).

    Raises:
        FileNotFoundError: If neither the context binary nor the DLC exists.
    """
    if unit in _BIN_UNITS:
        cand = variant_dir / f"{stem}.npu.bin"
        if cand.exists():
            return cand
    dlc = variant_dir / f"{stem}.dlc"
    if dlc.exists():
        return dlc
    tried = f"{stem}.npu.bin, " if unit in _BIN_UNITS else ""
    msg = f"No artifact found in {variant_dir}: tried {tried}{stem}.dlc"
    raise FileNotFoundError(msg)
