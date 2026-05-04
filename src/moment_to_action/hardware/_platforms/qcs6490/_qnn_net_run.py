"""QNN net-run subprocess backend for QCS6490.

Executes compiled QNN model libraries (.so) via the qnn-net-run CLI and
converts raw output files back into NumPy arrays.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import attrs
import numpy as np

from moment_to_action.hardware._platforms._base import InferenceBackend, ModelInput
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

_ENV_QNN_NET_RUN = "MOMENT_TO_ACTION_QNN_NET_RUN"
_ENV_QNN_BIN_DIR = "MOMENT_TO_ACTION_QNN_BIN_DIR"
_ENV_QNN_LIB_DIR = "MOMENT_TO_ACTION_QNN_LIB_DIR"
_ENV_QNN_BACKEND_LIB = "MOMENT_TO_ACTION_QNN_BACKEND_LIB"
_ENV_QNN_INPUT_NAME = "MOMENT_TO_ACTION_QNN_INPUT_NAME"
_ENV_QNN_INPUT_SHAPE = "MOMENT_TO_ACTION_QNN_INPUT_SHAPE"
_ENV_QNN_INPUT_DTYPE = "MOMENT_TO_ACTION_QNN_INPUT_DTYPE"
_ENV_QNN_OUTPUT_DTYPE = "MOMENT_TO_ACTION_QNN_OUTPUT_DTYPE"
_ENV_QNN_INPUT_LIST_TEMPLATE = "MOMENT_TO_ACTION_QNN_INPUT_LIST_TEMPLATE"
_ENV_QNN_OUTPUT_MODE = "MOMENT_TO_ACTION_QNN_OUTPUT_MODE"
_ENV_QNN_TIMEOUT_S = "MOMENT_TO_ACTION_QNN_TIMEOUT_S"

_DEFAULT_INPUT_NAME = "input_0"
_DEFAULT_INPUT_SHAPE = (1, 640, 640, 3)
_DEFAULT_OUTPUT_MODE = "yolo"
_YOLO_OUTPUT_TENSORS = 3
_YOLO_SCORE_TENSORS = 2
_YOLO_BOX_COMPONENTS = 4


@attrs.define(slots=True)
class _QnnHandle:
    model_path: str
    backend_lib: str
    input_name: str
    input_shape: tuple[int, ...]
    input_dtype: np.dtype
    output_dtype: np.dtype
    output_mode: str


class QCS6490QNNNetRunBackend(InferenceBackend):
    """Run compiled QNN model libraries via qnn-net-run."""

    def __init__(self, compute_unit: ComputeUnit) -> None:
        self._unit = compute_unit
        self._handles: dict[str, _QnnHandle] = {}

    def load_model(self, path: str | os.PathLike[str]) -> object:
        model_path = os.fspath(path)
        if model_path in self._handles:
            return self._handles[model_path]

        backend_lib = os.environ.get(_ENV_QNN_BACKEND_LIB) or self._backend_lib_for_unit()
        input_name = os.environ.get(_ENV_QNN_INPUT_NAME, _DEFAULT_INPUT_NAME)
        input_shape = _parse_shape(os.environ.get(_ENV_QNN_INPUT_SHAPE))
        if input_shape is None:
            input_shape = _DEFAULT_INPUT_SHAPE

        input_dtype = _parse_dtype(os.environ.get(_ENV_QNN_INPUT_DTYPE), np.dtype(np.float32))
        output_dtype = _parse_dtype(os.environ.get(_ENV_QNN_OUTPUT_DTYPE), np.dtype(np.float32))
        output_mode = os.environ.get(_ENV_QNN_OUTPUT_MODE, _DEFAULT_OUTPUT_MODE)

        handle = _QnnHandle(
            model_path=model_path,
            backend_lib=backend_lib,
            input_name=input_name,
            input_shape=input_shape,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            output_mode=output_mode,
        )
        self._handles[model_path] = handle
        logger.info("Loaded QNN model %s (backend=%s)", model_path, backend_lib)
        return handle

    def run(self, handle: object, inputs: ModelInput) -> list[np.ndarray]:
        h = cast("_QnnHandle", handle)
        if not isinstance(inputs, np.ndarray):
            msg = "QNN net-run backend expects a single ndarray input"
            raise TypeError(msg)

        tensor = inputs
        if tensor.dtype != h.input_dtype:
            tensor = tensor.astype(h.input_dtype, copy=False)

        qnn_net_run = os.environ.get(_ENV_QNN_NET_RUN, "qnn-net-run")
        timeout_s = _parse_timeout(os.environ.get(_ENV_QNN_TIMEOUT_S))

        with tempfile.TemporaryDirectory(prefix="qnn-net-run-") as tmpdir:
            tmp_path = Path(tmpdir)
            input_raw = tmp_path / "input.raw"
            input_list = tmp_path / "input_list.txt"
            output_dir = tmp_path / "output"

            tensor.tofile(input_raw)

            template = os.environ.get(_ENV_QNN_INPUT_LIST_TEMPLATE, "{input_name}:={path}")
            line = _format_input_list_line(template, h.input_name, input_raw)
            input_list.write_text(line + "\n", encoding="utf-8")

            cmd = [
                qnn_net_run,
                "--model",
                h.model_path,
                "--backend",
                h.backend_lib,
                "--input_list",
                str(input_list),
                "--output_dir",
                str(output_dir),
            ]

            env = _build_qnn_env(self._unit)
            logger.debug("Running qnn-net-run: %s", " ".join(cmd))
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
                check=False,
            )
            if result.returncode != 0:
                msg = (
                    "qnn-net-run failed with exit code "
                    f"{result.returncode}: {result.stderr.strip()}"
                )
                raise RuntimeError(msg)

            try:
                outputs = _load_raw_outputs(output_dir, h.output_dtype)
            except RuntimeError as exc:
                msg = (
                    f"{exc} | qnn-net-run stdout={_trim_cli_output(result.stdout)!r} "
                    f"stderr={_trim_cli_output(result.stderr)!r}"
                )
                raise RuntimeError(msg) from exc
            if h.output_mode == "raw":
                return outputs
            return _reshape_outputs(outputs)

    def get_input_details(self, handle: object) -> list[dict]:
        h = cast("_QnnHandle", handle)
        return [
            {
                "name": h.input_name,
                "shape": h.input_shape,
                "dtype": h.input_dtype,
            }
        ]

    def get_output_details(self, handle: object) -> list[dict]:
        h = cast("_QnnHandle", handle)
        return [
            {
                "name": "output_0",
                "shape": None,
                "dtype": h.output_dtype,
            }
        ]

    def get_supported_unit(self) -> ComputeUnit:
        return self._unit

    def _backend_lib_for_unit(self) -> str:
        if self._unit == ComputeUnit.GPU:
            return "libQnnGpu.so"
        if self._unit == ComputeUnit.NPU:
            return "libQnnHtp.so"
        return "libQnnCpu.so"


def _parse_shape(raw: str | None) -> tuple[int, ...] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return tuple(int(p) for p in parts)


def _parse_dtype(raw: str | None, default: np.dtype) -> np.dtype:
    if not raw:
        return default
    return np.dtype(raw)


def _parse_timeout(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _format_input_list_line(template: str, input_name: str, path: Path) -> str:
    try:
        return template.format(input_name=input_name, path=str(path))
    except KeyError as exc:
        msg = f"Invalid input list template: {template!r}"
        raise ValueError(msg) from exc


def _build_qnn_env(unit: ComputeUnit) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = os.environ.get(_ENV_QNN_BIN_DIR)
    if bin_dir:
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"

    lib_dir = os.environ.get(_ENV_QNN_LIB_DIR)
    if lib_dir:
        env["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"

    if unit == ComputeUnit.NPU:
        adsp = env.get("ADSP_LIBRARY_PATH")
        if not adsp:
            env["ADSP_LIBRARY_PATH"] = "/opt/;/usr/lib/rfsa/adsp;/dsp"

    return env


def _load_raw_outputs(output_root: Path, dtype: np.dtype) -> list[np.ndarray]:
    if not output_root.exists():
        msg = f"qnn-net-run output directory not found: {output_root}"
        raise RuntimeError(msg)

    raw_files = sorted(output_root.glob("*.raw"))
    if not raw_files:
        result_dirs = sorted(
            p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("Result_")
        )
        for result_dir in result_dirs:
            result_raws = sorted(result_dir.glob("*.raw"))
            if result_raws:
                raw_files = result_raws
                break

    # Some QNN versions/layouts nest results deeper under --output_dir.
    if not raw_files:
        raw_files = sorted(output_root.rglob("*.raw"))

    if not raw_files:
        entries = sorted(p.name for p in output_root.iterdir())
        msg = f"No .raw outputs found in {output_root} (entries={entries})"
        raise RuntimeError(msg)

    return [np.fromfile(raw, dtype=dtype) for raw in raw_files]


def _trim_cli_output(raw: str, limit: int = 400) -> str:
    text = raw.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _reshape_outputs(outputs: list[np.ndarray]) -> list[np.ndarray]:
    if not outputs:
        return outputs

    if len(outputs) == 1:
        arr = outputs[0]
        size = arr.size
        if size % 84 == 0:
            n = size // 84
            return [arr.reshape((1, 84, n))]
        return outputs

    if len(outputs) == _YOLO_OUTPUT_TENSORS:
        sizes = [arr.size for arr in outputs]
        for idx, size in enumerate(sizes):
            if size % _YOLO_BOX_COMPONENTS != 0:
                continue
            n = size // _YOLO_BOX_COMPONENTS
            if sizes.count(n) != _YOLO_SCORE_TENSORS:
                continue
            box = outputs[idx].reshape((1, n, _YOLO_BOX_COMPONENTS))
            remaining = [i for i in range(_YOLO_OUTPUT_TENSORS) if i != idx]
            score = outputs[remaining[0]].reshape((1, n))
            cls = outputs[remaining[1]].reshape((1, n))
            return [box, score, cls]

    return outputs
