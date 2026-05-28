"""Convert a model to DLC command."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rich_click as click

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import DEFAULT_VARIANT_KEY, ModelID, ModelManager
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
    qairt_mgr = QairtSDKManager.from_app_config(data.config, data.path_manager)
    if not qairt_mgr.is_available:
        msg = "QAIRT SDK not installed. Run 'm2a qairt install' first."
        raise click.ClickException(msg)

    model_mgr = ModelManager(data.path_manager)
    input_path = model_mgr.get_path(ModelID(model_id), variant)
    model = model_mgr.get_model(ModelID(model_id), variant=variant)

    # Preprocess calibration images (pure numpy — no backend needed)
    raw_images = _load_calibration_images(calibration_dir)
    prepared_list = [model.prepare(img) for img in raw_images]  # type: ignore[attr-defined]
    calibration_data = np.vstack(prepared_list).astype(np.float32)

    # Run ONNX model on each calibration image to capture reference outputs
    backend = ComputeBackend(ComputeUnit.CPU)
    model.load(backend)
    all_raw: list[list[np.ndarray]] = []
    for i in range(len(calibration_data)):
        raw = model.run(calibration_data[i : i + 1])  # type: ignore[attr-defined]
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

    # Convert to DLC
    dlc_path = output_dir / "model.dlc"
    qairt_mgr.convert(input_path, dlc_path, calibration_data)

    click.echo(f"Converted: {output_dir}")
