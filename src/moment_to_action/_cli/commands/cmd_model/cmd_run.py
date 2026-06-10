"""Run a model on an input image command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import cv2
import rich_click as click

from moment_to_action.hardware import ComputeBackend, ComputeUnit
from moment_to_action.models import DEFAULT_VARIANT_KEY, ModelID, ModelManager
from moment_to_action.models.image._base import ImageModel
from moment_to_action.models.image.classification._base import ImageClassificationModel
from moment_to_action.utils.cli import GlobalData, pass_global_data

if TYPE_CHECKING:
    import numpy as np

    from moment_to_action.models.image.classification._types import Classification
    from moment_to_action.models.image.detection._types import Detection


def _overlay_classifications(
    frame: np.ndarray, classifications: list[Classification]
) -> np.ndarray:
    """Overlay top-k classification labels onto a BGR frame.

    Args:
        frame: BGR uint8 image array (H, W, 3).
        classifications: Classification results ordered by descending confidence.

    Returns:
        Annotated BGR image array.
    """
    annotated = frame.copy()
    for rank, cls in enumerate(classifications):
        label = f"#{rank + 1} {cls.label} {cls.confidence:.2f}"
        y = 30 + rank * 30
        cv2.putText(
            annotated,
            label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


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
@click.option(
    "--backend",
    "backend_unit",
    type=click.Choice([u.value for u in ComputeUnit], case_sensitive=False),
    default=ComputeUnit.NPU.value,
    show_default=True,
    help="Preferred compute unit.",
)
@click.option(
    "--threshold",
    "confidence_threshold",
    type=float,
    default=None,
    help="Override model confidence threshold (0.0-1.0). Default uses model's built-in threshold.",
)
@pass_global_data
def run(
    data: GlobalData,
    model_id: str,
    input_path: Path,
    variant: str,
    output_format: str,
    output_path: Path | None,
    backend_unit: str,
    confidence_threshold: float | None,
) -> None:
    r"""Run a model on a single input image end-to-end.

    Loads the model, runs inference on the image, and prints detections as
    JSON or renders bounding boxes onto the image.

    Currently only image models are supported.

    \b
    Examples:
      m2a model run yolo_v8 image.jpg
      m2a model run yolo_v8 image.jpg --format image --output out.jpg
      m2a model run yolo_v8 image.jpg --variant qcs6490 --format json
      m2a model run yolo_v8 image.jpg --variant qcs6490 --backend CPU
    """
    frame = cv2.imread(str(input_path))
    if frame is None:
        msg = f"Could not read image: {input_path}"
        raise click.ClickException(msg)

    mid = ModelID(model_id)
    kwargs: dict[str, object] = {}
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold
    model = ModelManager(data.path_manager).get_model(mid, variant=variant, **kwargs)
    if not isinstance(model, ImageModel):
        msg = f"'{model_id}' is not an image model; run only supports image models currently."
        raise click.ClickException(msg)

    backend = ComputeBackend(preferred_unit=ComputeUnit(backend_unit))
    model.load(backend)
    try:
        prepared = model.prepare(frame)
        raw = model.run(prepared)
        detections = model.post_proc(raw)
    finally:
        model.unload()

    if output_format == "json":
        click.echo(json.dumps([attrs.asdict(d) for d in detections], indent=2))
    elif isinstance(model, ImageClassificationModel):
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_classifications")
        annotated = _overlay_classifications(frame, detections)  # type: ignore[arg-type]
        cv2.imwrite(str(output_path), annotated)
        click.echo(f"Saved: {output_path}")
    else:
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_detections")
        annotated = _draw_detections(frame, detections)  # type: ignore[arg-type]
        cv2.imwrite(str(output_path), annotated)
        click.echo(f"Saved: {output_path}")
