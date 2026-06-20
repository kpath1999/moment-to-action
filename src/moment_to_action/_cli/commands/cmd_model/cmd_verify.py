"""Verify model output correctness command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.models import ModelID, ModelManager
from moment_to_action.models.image.detection._base import ImageDetectionModel
from moment_to_action.utils.cli import GlobalData, pass_global_data

if TYPE_CHECKING:
    from pathlib import Path

_BACKEND_UNITS: dict[str, ComputeUnit] = {
    "cpu": ComputeUnit.CPU,
    "gpu": ComputeUnit.GPU,
    "npu": ComputeUnit.NPU,
}


def _load_reference(ref_dir: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load reference inputs and outputs from the reference_outputs directory.

    Args:
        ref_dir: Directory written by ``m2a model convert``.

    Returns:
        Tuple of (inputs array of shape (N, C, H, W), list of output arrays).

    Raises:
        click.ClickException: If the directory or required files are missing.
    """
    if not ref_dir.exists():
        msg = f"Reference outputs not found at {ref_dir}. Run 'm2a model convert' first."
        raise click.ClickException(msg)
    inputs = np.load(str(ref_dir / "inputs.npy"))
    ref_outputs: list[np.ndarray] = []
    k = 0
    while (ref_dir / f"outputs_{k}.npy").exists():
        ref_outputs.append(np.load(str(ref_dir / f"outputs_{k}.npy")))
        k += 1
    return inputs, ref_outputs


def _resolve_model(
    mgr: ModelManager,
    mid: ModelID,
    variant: str,
) -> tuple[object, str] | tuple[None, str]:
    """Return (model, "") on success or (None, error_reason) on failure.

    Args:
        mgr: ModelManager instance.
        mid: Model identifier.
        variant: Variant key to load.

    Returns:
        ``(model, "")`` if the model was resolved, ``(None, reason)`` otherwise.
    """
    if not mgr.is_available(mid, variant):
        return None, f"Variant '{variant}' not cached"
    return mgr.get_model(mid, variant=variant), ""


@click.command()
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
@click.argument("variant")
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu", "npu"], case_sensitive=False),
    default=None,
    help="Backend to verify. Omit to verify all three.",
)
@click.option(
    "--tol",
    default=0.01,
    show_default=True,
    type=float,
    help="Max absolute element-wise error for CPU/GPU raw comparison.",
)
@pass_global_data
def verify(
    data: GlobalData,
    model_id: str,
    variant: str,
    backend: str | None,
    tol: float,
) -> None:
    r"""Verify model output correctness against reference outputs.

    Loads reference inputs and outputs captured during ``m2a model convert``
    (or ``m2a model convert-aihub``), then re-runs inference on each backend
    and compares:

    - CPU/GPU: decoded detections (label match) AND raw element-wise diff ≤ tol.
    - NPU: decoded detections only (INT8 quantization noise dominates raw diff).

    Exits non-zero if any backend fails.

    \b
    Examples:
      m2a model verify yolo_v8 default
      m2a model verify yolo_v8 qcs6490 --backend npu
      m2a model verify yolo_v8 default --backend cpu --tol 0.005
    """
    mid = ModelID(model_id)
    mgr = ModelManager(data.path_manager)

    ref_dir = data.path_manager.cache.models.get_variant_dir(mid.value, variant)
    ref_dir = ref_dir / "reference_outputs"

    inputs, ref_outputs = _load_reference(ref_dir)

    backends_to_test = [backend.lower()] if backend else ["cpu", "gpu", "npu"]

    results: list[tuple[str, bool, str]] = []

    for backend_name in backends_to_test:
        unit = _BACKEND_UNITS[backend_name]
        is_npu = backend_name == "npu"

        model, err = _resolve_model(mgr, mid, variant)
        if model is None:
            results.append((backend_name, False, err))
            continue

        if not isinstance(model, ImageDetectionModel):
            results.append((backend_name, False, "model does not support verify"))
            continue

        platform = Platform()
        model.load(platform, unit)
        try:
            pass_all, fail_reason = model.verify_outputs(
                inputs, ref_outputs, tol=tol, is_npu=is_npu
            )
        finally:
            model.unload()

        results.append((backend_name, pass_all, fail_reason))

    console = Console()
    table = Table(title=f"Verify: {model_id}")
    table.add_column("Backend")
    table.add_column("Result")
    table.add_column("Detail")

    any_fail = False
    for bname, passed, reason in results:
        if passed:
            result_str = "[green]PASS[/green]"
        else:
            result_str = "[red]FAIL[/red]"
            any_fail = True
        table.add_row(bname.upper(), result_str, reason or "")

    console.print(table)

    if any_fail:
        raise SystemExit(1)
