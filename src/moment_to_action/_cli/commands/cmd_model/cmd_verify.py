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


def _find_dlc_variant(model_id: ModelID) -> str | None:
    """Return the first DLC variant key for a model, or None if none exist.

    Args:
        model_id: Model to search.

    Returns:
        Variant key string, or None.
    """
    for vkey, source in MODEL_REGISTRY[model_id].variants.items():
        if source.format == ModelFormat.DLC:
            return vkey
    return None


def _compare_decoded(
    ref_detections: list[object],
    act_detections: list[object],
) -> bool:
    """Check that actual decoded detections match reference by label set.

    Args:
        ref_detections: Reference Detection list.
        act_detections: Actual Detection list.

    Returns:
        True if the sorted label lists match.
    """
    ref_labels = sorted(d.label for d in ref_detections)  # type: ignore[attr-defined]
    act_labels = sorted(d.label for d in act_detections)  # type: ignore[attr-defined]
    return ref_labels == act_labels


def _check_images(
    model: object,
    inputs: np.ndarray,
    ref_outputs: list[np.ndarray],
    *,
    tol: float,
    is_npu: bool,
) -> tuple[bool, str]:
    """Run per-image comparison between model outputs and reference outputs.

    Args:
        model: Loaded model with ``run`` and ``post_proc`` methods.
        inputs: Input array of shape (N, C, H, W).
        ref_outputs: List of reference output arrays, each of shape (N, ...).
        tol: Max absolute element-wise error allowed for CPU/GPU raw comparison.
        is_npu: When True, skip raw diff check and compare decoded detections only.

    Returns:
        Tuple of (passed, fail_reason). ``passed`` is True when all images pass;
        ``fail_reason`` is an empty string on success, or a description of the
        first failure encountered.
    """
    for i in range(len(inputs)):
        inp = inputs[i : i + 1]
        act_raw = model.run(inp)  # type: ignore[attr-defined]

        if not is_npu:
            for k, (act_t, ref_t) in enumerate(zip(act_raw, ref_outputs, strict=False)):
                ref_row = ref_t[i : i + 1]
                max_err = float(
                    np.max(np.abs(act_t.astype(np.float32) - ref_row.astype(np.float32)))
                )
                if max_err > tol:
                    return False, f"output_{k}[{i}] max_err={max_err:.4f} > tol={tol}"

        ref_raw = [ref_outputs[k][i : i + 1] for k in range(len(ref_outputs))]
        ref_dets = model.post_proc(ref_raw)  # type: ignore[attr-defined]
        act_dets = model.post_proc(act_raw)  # type: ignore[attr-defined]
        if not _compare_decoded(ref_dets, act_dets):
            return False, f"decoded mismatch at image {i}"

    return True, ""


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
@pass_global_data
def verify(
    data: GlobalData,
    model_id: str,
    backend: str | None,
    tol: float,
) -> None:
    r"""Verify model output correctness against reference outputs.

    Loads reference inputs and outputs captured during ``m2a model convert``,
    then re-runs inference on each backend and compares:

    - CPU/GPU: decoded detections (label match) AND raw element-wise diff ≤ tol.
    - NPU: decoded detections only (INT8 quantization noise dominates raw diff).

    Exits non-zero if any backend fails.

    \b
    Examples:
      m2a model verify yolo_v8
      m2a model verify yolo_v8 --backend npu
      m2a model verify yolo_v8 --backend cpu --tol 0.005
    """
    mid = ModelID(model_id)
    mgr = ModelManager(data.path_manager)
    ref_dir = data.path_manager.cache.models.get_variant_dir(mid.value, DEFAULT_VARIANT_KEY)
    ref_dir = ref_dir / "reference_outputs"

    inputs, ref_outputs = _load_reference(ref_dir)

    backends_to_test = [backend.lower()] if backend else ["cpu", "gpu", "npu"]

    results: list[tuple[str, bool, str]] = []

    for backend_name in backends_to_test:
        unit = _BACKEND_UNITS[backend_name]
        is_npu = backend_name == "npu"

        if is_npu:
            dlc_variant = _find_dlc_variant(mid)
            if dlc_variant is None:
                results.append((backend_name, False, "No DLC variant registered"))
                continue
            if not mgr.is_available(mid, dlc_variant):
                results.append((backend_name, False, f"DLC variant '{dlc_variant}' not cached"))
                continue
            model = mgr.get_model(mid, variant=dlc_variant)
        else:
            model = mgr.get_model(mid, variant=DEFAULT_VARIANT_KEY)

        be = ComputeBackend(unit)
        model.load(be)
        try:
            pass_all, fail_reason = _check_images(
                model, inputs, ref_outputs, tol=tol, is_npu=is_npu
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
