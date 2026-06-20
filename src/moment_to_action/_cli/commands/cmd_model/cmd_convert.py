"""Convert a model to DLC command."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rich_click as click

from moment_to_action.hardware import ComputeUnit, Platform
from moment_to_action.models import DEFAULT_VARIANT_KEY, ModelID, ModelManager
from moment_to_action.models.image._base import ImageModel
from moment_to_action.qairt import QairtSDKManager
from moment_to_action.utils.cli import GlobalData, pass_global_data

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_calibration_images(calibration_dir: Path) -> list[np.ndarray]:
    """Load all supported images from a directory.

    Args:
        calibration_dir: Directory containing calibration images.

    Returns:
        List of BGR image arrays (uint8).

    Raises:
        click.ClickException: If the directory contains no supported images.
    """
    images = []
    for f in sorted(calibration_dir.iterdir()):
        if f.suffix.lower() in _IMAGE_EXTS:
            frame = cv2.imread(str(f))
            if frame is not None:
                images.append(frame)
    if not images:
        msg = (
            f"No supported images found in {calibration_dir}. "
            f"Supported extensions: {', '.join(sorted(_IMAGE_EXTS))}"
        )
        raise click.ClickException(msg)
    return images


@click.command()
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output variant directory. DLC written to <dir>/model.dlc, "
    "reference outputs to <dir>/reference_outputs/.",
)
@click.option(
    "--variant",
    default=DEFAULT_VARIANT_KEY,
    show_default=True,
    help="Source variant to convert.",
)
@click.option(
    "--calibration-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory of images for INT8 quantization calibration.",
)
@pass_global_data
def convert(
    data: GlobalData,
    model_id: str,
    output_dir: Path,
    variant: str,
    calibration_dir: Path,
) -> None:
    r"""Convert an ONNX model to quantized DLC for Qualcomm targets.

    Downloads the source model if not already cached, preprocesses calibration
    images, captures ONNX reference outputs, then converts via the QAIRT SDK.

    The output variant directory receives:
      - ``model.dlc`` — the quantized DLC file
      - ``reference_outputs/`` — ONNX reference inputs and outputs used by
        ``m2a model verify``

    \b
    Examples:
      m2a model convert yolo_v8 -o ./out/ --calibration-dir ./calib/
      m2a model convert yolo_v8 --variant default -o ./out/ --calibration-dir ./calib/
    """
    click.echo(
        click.style(
            "warning: local convert uses the QAIRT INT8 quantizer, which mis-handles some "
            "models (e.g. the YOLOv8 detection head collapses to ~0 scores) and emits only a "
            "portable .dlc — no per-backend context binaries. For AI Hub-supported models "
            "prefer 'm2a model convert-aihub'.",
            fg="yellow",
        )
    )

    qairt_mgr = QairtSDKManager.from_app_config(data.config, data.path_manager)
    if not qairt_mgr.is_available:
        msg = "QAIRT SDK not installed. Run 'm2a qairt install' first."
        raise click.ClickException(msg)

    model_mgr = ModelManager(data.path_manager)
    model = model_mgr.get_model(ModelID(model_id), variant=variant)
    if not isinstance(model, ImageModel):
        msg = f"'{model_id}' is not an image model."
        raise click.ClickException(msg)

    # Preprocess calibration images (pure numpy — no backend needed)
    raw_images = _load_calibration_images(calibration_dir)
    prepared_list = [model.prepare(img) for img in raw_images]
    calibration_data = np.vstack(prepared_list).astype(np.float32)

    # Run ONNX model on each calibration image to capture reference outputs
    platform = Platform()
    model.load(platform, ComputeUnit.CPU)
    all_raw: list[list[np.ndarray]] = []
    for i in range(len(calibration_data)):
        raw = model.run(calibration_data[i : i + 1])
        all_raw.append(raw)
    model.unload()

    ref_dir = output_dir / "reference_outputs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(ref_dir / "inputs.npy"), calibration_data)
    n_outputs = len(all_raw[0])
    for k in range(n_outputs):
        stacked = np.stack([all_raw[i][k] for i in range(len(all_raw))])
        np.save(str(ref_dir / f"outputs_{k}.npy"), stacked)

    # Resolve the actual ONNX file path (model.path may be a directory for HF sources)
    onnx_path = model.path / "model.onnx" if model.path.is_dir() else model.path
    # Apply model-specific ONNX surgery (e.g. split mixed-range outputs for YOLO)
    conversion_onnx = model.prepare_for_conversion(onnx_path)
    try:
        dlc_path = output_dir / "model.dlc"
        qairt_mgr.convert(conversion_onnx, dlc_path, calibration_data)
    finally:
        if conversion_onnx != model.path:
            conversion_onnx.unlink(missing_ok=True)

    click.echo(f"Converted: {output_dir}")
