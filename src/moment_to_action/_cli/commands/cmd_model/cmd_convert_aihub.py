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
from moment_to_action.models.image.detection.detectron2._model import Detectron2Model
from moment_to_action.models.image.detection.rf_detr._model import RFDETRModel
from moment_to_action.models.image.detection.rtmdet._model import RTMDetModel
from moment_to_action.models.image.detection.yolo._model import YOLOModel
from moment_to_action.utils.cli import GlobalData, pass_global_data

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Map ModelID → (qai_hub_models module id, pip extra for the model)
_AIHUB_MODEL_MAP: dict[ModelID, tuple[str, str]] = {
    ModelID.YOLO_V8: ("yolov8_det", "yolov8-det"),
    ModelID.RF_DETR: ("rf_detr", "rf-detr"),
    ModelID.RTM_DET: ("rtmdet", "rtmdet"),
    ModelID.DETECTRON2: ("detectron2_detection", "detectron2-detection"),
}

# Multi-component (CollectionModel) detectors.  Each export produces one artifact
# per component, named with the component string; we place them as
# ``model.<component>.dlc`` / ``model.<component>.npu.bin`` so
# resolve_backend_artifact(stem=...) can find each graph.  Single-graph models are
# absent here and use the plain ``model.dlc`` / ``model.npu.bin`` names.
_COMPONENT_STEMS: dict[ModelID, tuple[str, ...]] = {
    ModelID.DETECTRON2: ("proposal_generator", "roi_head"),
}

# Per-model NPU precision override for the context binary step.
# None → skip the context binary entirely (resolve_backend_artifact falls back to model.dlc).
#
# The Hexagon v68 AOT context-binary linker (qcs6490) rejects *any* floating-point graph
# I/O — both float32 and fp16 (an fp16 `image` input is rejected just like a float32
# output).  A model can only produce a v68 context binary if every I/O tensor is integer-
# quantised.  Both detection models below fail that requirement and fall back to model.dlc:
#
# RF-DETR: qai_hub_models exposes only `float` precision — there is no quantised export at
#   all, so the I/O is unavoidably float32.
# RTMDet: the exportable head runs the box decode in-graph — anchor arithmetic
#   (`block ± box` over linspace constants) plus an `argmax → float32` class cast.  These
#   ops are not integer-quantisable, so the `boxes` output stays floating-point under every
#   precision.  Empirically all of w8a16_mixed_fp16, full-fp16 (`--quantize_full_type
#   float16`), and plain w8a16 fail the v68 link with "Tensor '…' has a floating-point type
#   which is not supported by the targeted device", and `--quantize_io_type` (which could
#   force integer I/O) is TFLite-only.  A working v68 binary would require exporting the raw
#   pre-decode conv heads and moving the decode to CPU post-processing — out of scope here.
#
# Detectron2 is the counter-example and is intentionally absent here: it exposes full-integer
# `w8a8` / `w8a16` precisions, so both component graphs quantise end-to-end to integer I/O and
# link cleanly on v68.  With no override entry it follows the CLI `--precision`, letting us
# build either an int8 or int16 context-binary variant.
_NPU_PRECISION_OVERRIDE: dict[ModelID, str | None] = {
    ModelID.RF_DETR: None,
    ModelID.RTM_DET: None,
}

# Models that must NOT ship the portable float DLC.  The two-component float DLC's
# proposal-generator -> ROI-head feature handoff does not reproduce on the QAIRT
# CPU/GPU reference backends, so those units use the single-graph ONNX `default`
# variant instead; only the NPU `.npu.bin` is shipped.  The component DLCs are
# still produced transiently to capture reference outputs, then deleted.
_NO_SHIP_DLC: frozenset[ModelID] = frozenset({ModelID.DETECTRON2})


def _npu_compile_link_options(model_id: ModelID) -> tuple[str, str]:
    """Return ``(compile_options, link_options)`` for the NPU context-binary step.

    YOLO needs ``default_graph_htp_precision=FLOAT16`` so any float32 tensors that
    survive quantisation are handled as fp16 (valid on v68).  The full-integer
    Detectron2 graphs need no such flag — the spike linked cleanly without it.

    Args:
        model_id: Model being converted.

    Returns:
        ``(compile_options, link_options)`` strings for :func:`_run_aihub_export`.
    """
    if model_id in _COMPONENT_STEMS:
        return "", ""
    fp16 = "--qnn_options default_graph_htp_precision=FLOAT16"
    return fp16, fp16


def _place_components(
    src_dir: Path, glob_ext: str, dest_dir: Path, components: tuple[str, ...], dest_suffix: str
) -> None:
    """Copy each component artifact into ``dest_dir`` under its stem name.

    Globs ``src_dir`` for ``*{glob_ext}`` files and matches each ``component`` by
    substring, writing it to ``dest_dir/model.<component>{dest_suffix}``.

    Args:
        src_dir: Build directory the export wrote artifacts into.
        glob_ext: Artifact extension to match (``".dlc"`` or ``".bin"``).
        dest_dir: Destination variant directory.
        components: Component stems to place (e.g. ``("proposal_generator", "roi_head")``).
        dest_suffix: Suffix for the destination filename (``".dlc"`` or ``".npu.bin"``).

    Raises:
        click.ClickException: If a component's artifact is not found.
    """
    for comp in components:
        matches = sorted(p for p in src_dir.rglob(f"*{glob_ext}") if comp in p.name)
        if not matches:
            msg = f"No {comp}{glob_ext} artifact found under {src_dir} after export."
            raise click.ClickException(msg)
        dest = dest_dir / f"model.{comp}{dest_suffix}"
        shutil.copy2(matches[0], dest)
        click.echo(f"Component artifact: {dest}")


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
    _cpu_single: dict[ComputeUnit, dict[str, str]] = {ComputeUnit.CPU: {"model": "model.dlc"}}
    _cpu_det2: dict[ComputeUnit, dict[str, str]] = {
        ComputeUnit.CPU: {
            "proposal_generator": "model.proposal_generator.dlc",
            "roi_head": "model.roi_head.dlc",
        },
    }
    if model_id is ModelID.YOLO_V8:
        return YOLOModel(
            variant="qcs6490",
            path=variant_dir,
            model_format=ModelFormat.DLC,
            backends=_cpu_single,
            input_layout="NHWC",
        )
    if model_id is ModelID.RF_DETR:
        return RFDETRModel(
            variant="qcs6490",
            path=variant_dir,
            model_format=ModelFormat.DLC,
            backends=_cpu_single,
            input_layout="NHWC",
        )
    if model_id is ModelID.RTM_DET:
        return RTMDetModel(
            variant="qcs6490",
            path=variant_dir,
            model_format=ModelFormat.DLC,
            backends=_cpu_single,
            input_layout="NHWC",
        )
    if model_id is ModelID.DETECTRON2:
        return Detectron2Model(
            variant="qcs6490",
            path=variant_dir,
            model_format=ModelFormat.DLC,
            backends=_cpu_det2,
            input_layout="NHWC",
        )
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
    compile_options: str = "",
    link_options: str = "",
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
        compile_options: Extra QNN compiler flags for the compile step.
        link_options: Extra QNN compiler flags for the link step.  The
            ``qai_hub_models`` ``export_model`` API does not forward
            ``compile_options`` to the link job, so any options needed at link
            time must be supplied separately here via monkey-patching.

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

    # export_model does not forward compile_options to the link job.  Patch
    # link_model in the export module to inject link_options for this call only.
    _orig_link = getattr(export_mod, "link_model", None)
    if link_options and _orig_link is not None:
        _injected = link_options

        def _patched_link(
            compiled_model: object,
            device: object,
            model_name: str,
            model: object,
            target_runtime: object,
            extra_options: str = "",
        ) -> object:
            combined = (_injected + " " + extra_options).strip()
            return _orig_link(  # type: ignore[misc]
                compiled_model,
                device,
                model_name,
                model,
                target_runtime,
                extra_options=combined,
            )

        export_mod.link_model = _patched_link  # type: ignore[attr-defined]

    try:
        # Note: do NOT pass num_calibration_samples — their parser leaves it as a
        # str, causing TypeError inside get_calibration_data.
        export_mod.export_model(
            device=hub_device,
            skip_profiling=True,
            skip_inferencing=True,
            skip_summary=True,
            output_dir=str(output_dir),
            precision=precision_obj,
            target_runtime=runtime_obj,
            compile_options=compile_options,
        )
    finally:
        if link_options and _orig_link is not None:
            export_mod.link_model = _orig_link  # type: ignore[attr-defined]

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
    components = _COMPONENT_STEMS.get(mid)
    build_dir = output_dir / "_aihub_build"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Portable DLC.  Multi-component (CollectionModel) detectors yield one DLC per
    # component, placed as model.<component>.dlc; single-graph models use model.dlc.
    dlc_path = _run_aihub_export(
        model_id=aihub_model_id,
        precision="float",
        runtime="qnn_dlc",
        chipset=chipset,
        output_dir=build_dir / "dlc",
        token=token,
    )
    if components is None:
        dest_dlc = output_dir / "model.dlc"
        shutil.copy2(dlc_path, dest_dlc)
        click.echo(f"DLC: {dest_dlc}")
    else:
        _place_components(build_dir / "dlc", ".dlc", output_dir, components, ".dlc")

    # Copy sidecar files (metadata.json, labels.txt) if present
    for sidecar in ("metadata.json", "labels.txt"):
        src = dlc_path.parent / sidecar
        if src.exists():
            shutil.copy2(src, output_dir / sidecar)

    # Step 2: Reference outputs from the portable DLC.
    # Must run before context binaries are copied into output_dir — context binaries
    # are compiled for the qcs6490 device (aarch64/HTP) and cannot load on x86.
    # resolve_backend_artifact falls back to model.dlc when no .bin files are present.
    _capture_reference_outputs(mid, calibration_dir, output_dir)

    # Drop the float component DLCs we just used for reference capture: they are
    # dead weight at runtime for NPU-only-shipping models (CPU/GPU use the
    # single-graph ONNX `default` variant; NPU loads the .npu.bin below).
    if mid in _NO_SHIP_DLC and components is not None:
        for comp in components:
            (output_dir / f"model.{comp}.dlc").unlink(missing_ok=True)
            click.echo(f"Removed dead float DLC: model.{comp}.dlc")

    # Step 3: NPU context binary (HTP AOT-compiled; CPU/GPU fall back to the DLC).
    # Per-model precision overrides live in _NPU_PRECISION_OVERRIDE.  When the override
    # is None the context binary step is skipped entirely; resolve_backend_artifact will
    # fall back to the DLC at runtime.  Compile/link options come from
    # _npu_compile_link_options (FLOAT16 for YOLO, none for full-int Detectron2).
    npu_precision = _NPU_PRECISION_OVERRIDE.get(mid, precision)
    if npu_precision is None:
        click.echo(
            f"Skipping NPU context binary for {model_id}: model has floating-point I/O "
            "that the Hexagon v68 linker rejects (see _NPU_PRECISION_OVERRIDE). "
            "resolve_backend_artifact will fall back to model.dlc."
        )
    else:
        compile_opts, link_opts = _npu_compile_link_options(mid)
        npu_bin_path = _run_aihub_export(
            model_id=aihub_model_id,
            precision=npu_precision,
            runtime="qnn_context_binary",
            chipset=chipset,
            output_dir=build_dir / "npu",
            token=token,
            compile_options=compile_opts,
            link_options=link_opts,
        )
        if components is None:
            dest_npu = output_dir / "model.npu.bin"
            shutil.copy2(npu_bin_path, dest_npu)
            click.echo(f"Context binary: {dest_npu}")
        else:
            _place_components(build_dir / "npu", ".bin", output_dir, components, ".npu.bin")
