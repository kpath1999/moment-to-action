"""Verify model output correctness command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rich_click as click
from rich.console import Console
from rich.table import Table

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import DEFAULT_VARIANT_KEY, MODEL_REGISTRY, ModelID, ModelManager
from moment_to_action.models._formats import ModelFormat
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
    backend_name: str,
    variant: str | None,
) -> tuple[object, str] | tuple[None, str]:
    """Return (model, "") on success or (None, error_reason) on failure.

    Args:
        mgr: ModelManager instance.
        mid: Model identifier.
        backend_name: Backend name ("cpu", "gpu", "npu").
        variant: Explicit variant override, or None for auto-selection.

    Returns:
        ``(model, "")`` if the model was resolved, ``(None, reason)`` otherwise.
    """
    is_npu = backend_name == "npu"
    if variant is not None:
        if not mgr.is_available(mid, variant):
            return None, f"Variant '{variant}' not cached"
        return mgr.get_model(mid, variant=variant), ""
    if is_npu:
        dlc_variant = _find_dlc_variant(mid)
        if dlc_variant is None:
            return None, "No DLC variant registered"
        if not mgr.is_available(mid, dlc_variant):
            return None, f"DLC variant '{dlc_variant}' not cached"
        return mgr.get_model(mid, variant=dlc_variant), ""
    return mgr.get_model(mid, variant=DEFAULT_VARIANT_KEY), ""


def _find_dlc_variant(model_id: ModelID) -> str | None:
    """Return the first DLC variant key for a model, or None if none exist.

    Args:
        model_id: Model to search.

    Returns:
        Variant key string, or None.
    """
    for vkey, source in MODEL_REGISTRY[model_id].variants.items():
        if source.source.format == ModelFormat.DLC:
            return vkey
    return None


@click.command()
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
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
@click.option(
    "--variant",
    default=None,
    help=(
        "Explicit variant to use for all backends. When set, reference outputs are "
        "loaded from that variant's directory instead of the default. "
        "Overrides the automatic DLC-variant selection for NPU."
    ),
)
@pass_global_data
def verify(
    data: GlobalData,
    model_id: str,
    backend: str | None,
    tol: float,
    variant: str | None,
) -> None:
    r"""Verify model output correctness against reference outputs.

    Loads reference inputs and outputs captured during ``m2a model convert``
    (or ``m2a model convert-aihub``), then re-runs inference on each backend
    and compares:

    - CPU/GPU: decoded detections (label match) AND raw element-wise diff ≤ tol.
    - NPU: decoded detections only (INT8 quantization noise dominates raw diff).

    Use ``--variant`` to test a specific variant (e.g. an AI Hub DLC) against
    that variant's own reference outputs.

    Exits non-zero if any backend fails.

    \b
    Examples:
      m2a model verify yolo_v8
      m2a model verify yolo_v8 --backend npu
      m2a model verify yolo_v8 --variant qcs6490 --backend npu
      m2a model verify yolo_v8 --backend cpu --tol 0.005
    """
    mid = ModelID(model_id)
    mgr = ModelManager(data.path_manager)

    # Determine the variant from which to load reference outputs.
    ref_variant = variant if variant is not None else DEFAULT_VARIANT_KEY
    ref_dir = data.path_manager.cache.models.get_variant_dir(mid.value, ref_variant)
    ref_dir = ref_dir / "reference_outputs"

    inputs, ref_outputs = _load_reference(ref_dir)

    backends_to_test = [backend.lower()] if backend else ["cpu", "gpu", "npu"]

    results: list[tuple[str, bool, str]] = []

    for backend_name in backends_to_test:
        unit = _BACKEND_UNITS[backend_name]
        is_npu = backend_name == "npu"

        model, err = _resolve_model(mgr, mid, backend_name, variant)
        if model is None:
            results.append((backend_name, False, err))
            continue

        if not isinstance(model, ImageDetectionModel):
            results.append((backend_name, False, "model does not support verify"))
            continue

        be = ComputeBackend(unit)
        model.load(be)
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
