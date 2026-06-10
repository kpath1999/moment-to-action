"""Convert a model to a quantized DLC via Qualcomm AI Hub."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import rich_click as click

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import ModelID
from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image._base import ImageModel
from moment_to_action.models.image.detection.yolo._model import YOLOModel
from moment_to_action.utils.cli import GlobalData, pass_global_data

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Map ModelID → (qai_hub_models module id, pip extra for the model)
_AIHUB_MODEL_MAP: dict[ModelID, tuple[str, str]] = {
    ModelID.YOLO_V8: ("yolov8_det", "yolov8-det"),
}


def _build_dlc_model(model_id: ModelID, variant_dir: Path) -> ImageModel:
    """Instantiate the correct model class pointing at the DLC in ``variant_dir``.

    Creates a model object configured for the AI Hub ``qcs6490`` DLC
    (NHWC input layout, DLC format, ``qcs6490`` variant key) without going
    through :class:`~moment_to_action.models.ModelManager` — the artifacts
    live in ``variant_dir``, not in the model cache.

    Args:
        model_id: Which model to instantiate.
        variant_dir: Directory containing the freshly-produced ``model.dlc``.

    Returns:
        An unloaded :class:`~moment_to_action.models.image.ImageModel` instance.

    Raises:
        click.ClickException: If ``model_id`` has no registered factory.
    """
    if model_id is ModelID.YOLO_V8:
        return YOLOModel(variant="qcs6490", path=variant_dir, model_format=ModelFormat.DLC)
    msg = f"No DLC model factory for '{model_id.value}'."
    raise click.ClickException(msg)


def _capture_reference_outputs(
    model_id: ModelID,
    calibration_dir: Path,
    output_dir: Path,
) -> None:
    """Run the AI Hub DLC on calibration images and save reference outputs.

    Loads the DLC from ``output_dir/model.dlc`` (using the qcs6490 NHWC
    variant configuration), runs inference on CPU, and writes ``inputs.npy``
    and ``outputs_k.npy`` into ``<output_dir>/reference_outputs/``.

    The output tensors produced by the AI Hub DLC are
    ``[boxes (1,8400,4), scores (1,8400), class_idx (1,8400)]``, matching
    the three-output contract expected by ``YOLOModel.verify_outputs``.

    Args:
        model_id: Model to use for reference capture.
        calibration_dir: Directory of calibration images.
        output_dir: Variant output directory; ``reference_outputs/`` written here.
            Must already contain ``model.dlc``.

    Raises:
        click.ClickException: If no images are found or model_id has no factory.
    """
    images = sorted(p for p in calibration_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        msg = f"No images found in {calibration_dir} for reference output capture."
        raise click.ClickException(msg)

    model = _build_dlc_model(model_id, output_dir)
    if not isinstance(model, ImageModel):
        msg = f"'{model_id.value}' is not an image model."
        raise click.ClickException(msg)

    raw_imgs = [cv2.imread(str(p)) for p in images]
    prepared = [model.prepare(img) for img in raw_imgs]
    calib = np.vstack(prepared).astype(np.float32)

    backend = ComputeBackend(ComputeUnit.CPU)
    model.load(backend)
    all_raw: list[list[np.ndarray]] = [model.run(calib[i : i + 1]) for i in range(len(calib))]
    model.unload()

    ref_dir = output_dir / "reference_outputs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(ref_dir / "inputs.npy"), calib)
    for k in range(len(all_raw[0])):
        stacked = np.stack([all_raw[i][k] for i in range(len(all_raw))])
        np.save(str(ref_dir / f"outputs_{k}.npy"), stacked)
    click.echo(f"Reference outputs written to {ref_dir}")


def _check_token() -> str:
    """Return the AI Hub token or raise a clear error if missing.

    Returns:
        The token string.

    Raises:
        click.ClickException: If ``QAI_HUB_API_TOKEN`` is not set.
    """
    token = os.environ.get("QAI_HUB_API_TOKEN") or os.environ.get("QAI_HUB_API_KEY")
    if not token:
        msg = (
            "QAI_HUB_API_TOKEN is not set. "
            "Set it in your .env file or environment. "
            "Sign up at https://aihub.qualcomm.com to obtain a token."
        )
        raise click.ClickException(msg)
    return token


def _run_aihub_export(
    model_id: str,
    precision: str,
    runtime: str,
    chipset: str,
    output_dir: Path,
    token: str,
) -> Path:
    """Run the qai_hub_models export and return the path to the produced artifact.

    Drives ``qai_hub_models.models.<model_id>.export`` via its Python API.
    For ``qnn_dlc`` runtime, returns the ``.dlc`` file.  For
    ``qnn_context_binary`` runtime, returns the ``.bin`` context binary.

    Args:
        model_id: qai_hub_models model identifier (e.g. ``"yolov8_det"``).
        precision: Quantization precision (e.g. ``"w8a8"``).
        runtime: Target runtime (``"qnn_dlc"`` or ``"qnn_context_binary"``).
        chipset: Target chipset slug (e.g. ``"qualcomm-qcs6490"``).
        output_dir: Directory to write artifacts into.
        token: AI Hub API token.

    Returns:
        Path to the produced ``.dlc`` or ``.bin`` file.

    Raises:
        click.ClickException: If qai_hub_models is not installed, the export
            fails, or no artifact is found in the output.
    """
    try:
        import qai_hub  # noqa: PLC0415
    except ImportError as exc:
        msg = "qai-hub-models is not installed. Install with: uv sync --extra host"
        raise click.ClickException(msg) from exc

    hub_client = qai_hub.Client(config=qai_hub.ClientConfig(token))

    devices = hub_client.get_devices(attributes=f"chipset:{chipset}")
    if not devices:
        msg = f"No AI Hub device found for chipset '{chipset}'."
        raise click.ClickException(msg)
    hub_device = devices[-1]

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import importlib  # noqa: PLC0415

        export_mod = importlib.import_module(f"qai_hub_models.models.{model_id}.export")
    except ImportError as exc:
        extra = model_id.replace("_", "-")
        msg = (
            f"qai_hub_models model '{model_id}' is not available. "
            f"Install its extra: uv run --with 'qai-hub-models[{extra}]' ..."
        )
        raise click.ClickException(msg) from exc

    precision_obj = getattr(export_mod.Precision, precision)
    runtime_obj = export_mod.TargetRuntime(runtime)

    click.echo(
        f"Submitting AI Hub export job for {model_id} ({precision}, {runtime}, {chipset}) ..."
    )

    # Call the export function. Note: do NOT pass num_calibration_samples — their
    # parser leaves it as a str, causing TypeError inside get_calibration_data.
    export_mod.export_model(
        device=hub_device,
        skip_profiling=True,
        skip_inferencing=True,
        skip_summary=True,
        output_dir=str(output_dir),
        precision=precision_obj,
        target_runtime=runtime_obj,
    )

    # Determine the expected file extension for this runtime
    ext = ".bin" if runtime == "qnn_context_binary" else ".dlc"
    artifact_files = list(output_dir.rglob(f"*{ext}"))
    if not artifact_files:
        msg = (
            f"No {ext} file found under {output_dir} after export. "
            "The AI Hub job may have exited early (COCO download quirk). Re-run the command."
        )
        raise click.ClickException(msg)

    return artifact_files[0]


@click.command("convert-aihub")
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
@click.option(
    "--precision",
    default="w8a8",
    show_default=True,
    type=click.Choice(["w8a8", "w8a16", "float"], case_sensitive=False),
    help="Quantization precision.",
)
@click.option(
    "--chipset",
    default="qualcomm-qcs6490",
    show_default=True,
    help="Target chipset slug.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory. DLC written to <dir>/model.dlc, context binaries alongside.",
)
@click.option(
    "--calibration-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of images for DLC reference output capture.",
)
@pass_global_data
def convert_aihub(
    data: GlobalData,  # noqa: ARG001
    model_id: str,
    precision: str,
    chipset: str,
    output_dir: Path,
    calibration_dir: Path,
) -> None:
    r"""Convert a model using Qualcomm AI Hub cloud quantizer.

    Produces a quantized DLC and three per-backend context binaries via AI
    Hub's production cloud quantizer — bypassing the local QAIRT INT8
    quantizer which mis-handles some models (e.g. YOLOv8's detection head).
    Also captures reference outputs from the DLC for ``m2a model verify``.

    Output layout::

        <output_dir>/
          model.dlc          # portable master DLC
          model.cpu.bin      # AOT QNN context binary, CPU backend
          model.gpu.bin      # AOT QNN context binary, GPU backend
          model.npu.bin      # AOT QNN context binary, HTP/NPU backend
          reference_outputs/ # inputs + 3 output arrays for verify

    Requires ``QAI_HUB_API_TOKEN`` in the environment (or ``.env``).
    Requires the ``[host]`` extra: ``uv sync --extra host``.

    \b
    Examples:
      m2a model convert-aihub yolo_v8 -o ./out/ --calibration-dir ./calib/
      m2a model convert-aihub yolo_v8 --precision w8a8 --chipset qualcomm-qcs6490 \
          -o ./out/ --calibration-dir ./calib/
    """
    mid = ModelID(model_id)
    if mid not in _AIHUB_MODEL_MAP:
        supported = ", ".join(m.value for m in _AIHUB_MODEL_MAP)
        msg = f"'{model_id}' is not supported by convert-aihub. Supported: {supported}"
        raise click.ClickException(msg)

    token = _check_token()
    aihub_model_id, _ = _AIHUB_MODEL_MAP[mid]
    build_dir = output_dir / "_aihub_build"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Portable DLC
    dlc_path = _run_aihub_export(
        model_id=aihub_model_id,
        precision=precision,
        runtime="qnn_dlc",
        chipset=chipset,
        output_dir=build_dir / "dlc",
        token=token,
    )
    dest_dlc = output_dir / "model.dlc"
    shutil.copy2(dlc_path, dest_dlc)

    # Copy sidecar files (metadata.json, labels.txt) if present
    for sidecar in ("metadata.json", "labels.txt"):
        src = dlc_path.parent / sidecar
        if src.exists():
            shutil.copy2(src, output_dir / sidecar)

    click.echo(f"DLC: {dest_dlc}")

    # Step 2: Reference outputs from the portable DLC.
    # Must run before context binaries are copied into output_dir — context binaries
    # are compiled for the qcs6490 device (aarch64/HTP) and cannot load on x86.
    # resolve_backend_artifact falls back to model.dlc when no .bin files are present.
    _capture_reference_outputs(mid, calibration_dir, output_dir)

    # Step 3: NPU context binary (HTP AOT-compiled; CPU/GPU fall back to model.dlc)
    npu_bin_path = _run_aihub_export(
        model_id=aihub_model_id,
        precision=precision,
        runtime="qnn_context_binary",
        chipset=chipset,
        output_dir=build_dir / "npu",
        token=token,
    )
    dest_npu = output_dir / "model.npu.bin"
    shutil.copy2(npu_bin_path, dest_npu)
    click.echo(f"Context binary: {dest_npu}")
