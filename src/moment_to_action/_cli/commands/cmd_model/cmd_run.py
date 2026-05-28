"""Run a model on an input image command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import cv2
import rich_click as click

from moment_to_action.hardware import ComputeBackend
from moment_to_action.models import DEFAULT_VARIANT_KEY, ModelID, ModelManager
from moment_to_action.utils.cli import GlobalData, pass_global_data

if TYPE_CHECKING:
    import numpy as np

    from moment_to_action.models.image.detection._types import Detection


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels onto a BGR frame.

    Args:
        frame: BGR uint8 image array (H, W, 3).
        detections: Detections to annotate.

    Returns:
        Annotated BGR image array.
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


@click.command()
@click.argument("model_id", type=click.Choice([m.value for m in ModelID], case_sensitive=False))
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--variant",
    default=DEFAULT_VARIANT_KEY,
    show_default=True,
    help="Variant key to run.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "image"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for image format. Defaults to <stem>_detections<ext> next to input.",
)
@pass_global_data
def run(
    data: GlobalData,
    model_id: str,
    input_path: Path,
    variant: str,
    output_format: str,
    output_path: Path | None,
) -> None:
    r"""Run a model on a single input image end-to-end.

    Loads the model, runs inference on the image, and prints detections as
    JSON or renders bounding boxes onto the image.

    \b
    Examples:
      m2a model run yolo_v8 image.jpg
      m2a model run yolo_v8 image.jpg --format image --output out.jpg
      m2a model run yolo_v8 image.jpg --variant qcs6490 --format json
    """
    frame = cv2.imread(str(input_path))
    if frame is None:
        msg = f"Could not read image: {input_path}"
        raise click.ClickException(msg)

    mid = ModelID(model_id)
    model = ModelManager(data.path_manager).get_model(mid, variant=variant)
    backend = ComputeBackend()
    model.load(backend)
    try:
        prepared = model.prepare(frame)  # type: ignore[attr-defined]
        raw = model.run(prepared)  # type: ignore[attr-defined]
        detections = model.post_proc(raw)  # type: ignore[attr-defined]
    finally:
        model.unload()

    if output_format == "json":
        click.echo(json.dumps([attrs.asdict(d) for d in detections], indent=2))
    else:
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_detections")
        annotated = _draw_detections(frame, detections)
        cv2.imwrite(str(output_path), annotated)
        click.echo(f"Saved: {output_path}")
