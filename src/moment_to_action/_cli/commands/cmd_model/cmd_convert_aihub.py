"""Convert a model to a quantized DLC via Qualcomm AI Hub."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import rich_click as click

from moment_to_action.models import ModelID
from moment_to_action.utils.cli import GlobalData, pass_global_data

# Map ModelID → (qai_hub_models module id, pip extra for the model)
_AIHUB_MODEL_MAP: dict[ModelID, tuple[str, str]] = {
    ModelID.YOLO_V8: ("yolov8_det", "yolov8-det"),
}

# Artifacts produced by the AI Hub export under <out>/<model_id>-<runtime>-<precision>/
_DLC_SUBDIR_PATTERN = "{model_id}-{runtime}-{precision}"


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
    """Run the qai_hub_models export and return the path to the produced DLC.

    Drives ``qai_hub_models.models.<model_id>.export`` via its Python API,
    waiting explicitly for compile/quantize jobs so the CLI exit-after-COCO-
    download quirk is avoided.

    Args:
        model_id: qai_hub_models model identifier (e.g. ``"yolov8_det"``).
        precision: Quantization precision (e.g. ``"w8a8"``).
        runtime: Target runtime (e.g. ``"qnn_dlc"``).
        chipset: Target chipset slug (e.g. ``"qualcomm-qcs6490"``).
        output_dir: Directory to write artifacts into.
        token: AI Hub API token.

    Returns:
        Path to the produced ``.dlc`` file.

    Raises:
        click.ClickException: If qai_hub_models is not installed, or the
            export fails, or no DLC file is found in the output.
    """
    try:
        import qai_hub  # noqa: PLC0415
    except ImportError as exc:
        msg = "qai-hub-models is not installed. Install with: uv sync --extra host"
        raise click.ClickException(msg) from exc

    qai_hub.set_session_token(token)

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

    click.echo(f"Submitting AI Hub export job for {model_id} ({precision}, {chipset}) ...")

    # Call the export function. Note: do NOT pass num_calibration_samples — their
    # parser leaves it as a str, causing TypeError inside get_calibration_data.
    export_mod.export_model(
        device=chipset,
        skip_profiling=True,
        skip_inferencing=True,
        skip_summary=True,
        output_dir=str(output_dir),
        precision=precision,
        target_runtime=runtime,
    )

    # Find the produced DLC — it lands in a subdirectory named like
    # yolov8_det-qnn_dlc-w8a8/yolov8_det.dlc
    dlc_files = list(output_dir.rglob("*.dlc"))
    if not dlc_files:
        msg = (
            f"No .dlc file found under {output_dir} after export. "
            "The AI Hub job may have exited early (COCO download quirk). Re-run the command."
        )
        raise click.ClickException(msg)

    return dlc_files[0]


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
    "--runtime",
    default="qnn_dlc",
    show_default=True,
    help="Target runtime (qnn_dlc or qnn_context_binary).",
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
    help="Output directory. DLC written to <dir>/model.dlc, metadata kept alongside.",
)
@pass_global_data
def convert_aihub(
    _data: GlobalData,
    model_id: str,
    precision: str,
    runtime: str,
    chipset: str,
    output_dir: Path,
) -> None:
    r"""Convert a model using Qualcomm AI Hub cloud quantizer.

    Produces a quantized DLC (and optionally per-backend context binaries) via
    AI Hub's production cloud quantizer — bypassing the local QAIRT INT8
    quantizer which mis-handles some models (e.g. YOLOv8's detection head).

    Requires ``QAI_HUB_API_TOKEN`` in the environment (or ``.env``).
    Requires the ``[host]`` extra: ``uv sync --extra host``.

    \b
    Examples:
      m2a model convert-aihub yolo_v8 -o ./out/
      m2a model convert-aihub yolo_v8 --precision w8a8 --chipset qualcomm-qcs6490 -o ./out/
    """
    mid = ModelID(model_id)
    if mid not in _AIHUB_MODEL_MAP:
        supported = ", ".join(m.value for m in _AIHUB_MODEL_MAP)
        msg = f"'{model_id}' is not supported by convert-aihub. Supported: {supported}"
        raise click.ClickException(msg)

    token = _check_token()
    aihub_model_id, _ = _AIHUB_MODEL_MAP[mid]

    build_dir = output_dir / "_aihub_build"
    dlc_path = _run_aihub_export(
        model_id=aihub_model_id,
        precision=precision,
        runtime=runtime,
        chipset=chipset,
        output_dir=build_dir,
        token=token,
    )

    # Relocate artifacts to the canonical layout
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_dlc = output_dir / "model.dlc"
    shutil.copy2(dlc_path, dest_dlc)

    # Copy sidecar files (metadata.json, labels.txt) if present
    for sidecar in ("metadata.json", "labels.txt"):
        src = dlc_path.parent / sidecar
        if src.exists():
            shutil.copy2(src, output_dir / sidecar)

    click.echo(f"Converted: {dest_dlc}")
