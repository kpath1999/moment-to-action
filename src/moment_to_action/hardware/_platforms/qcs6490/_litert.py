"""LiteRT backend for the QCS6490 platform with QNN accelerator support.

Extends the shared LiteRTBackend to add QNN delegate support for NPU/GPU
inference on the Qualcomm QCS6490.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from moment_to_action.hardware._platforms._runtimes import LiteRTBackend
from moment_to_action.hardware._types import ComputeUnit

logger = logging.getLogger(__name__)

# Path to the Qualcomm QNN TFLite delegate shared library on-device.
_QNN_DELEGATE_PATH = "/usr/lib/libQnnTFLiteDelegate.so"
_QNN_BACKEND_KEY = "backend_type"
# On Rubik Pi/QCS6490 the delegate accepts backend_type symbols. Using
# backend_path triggers a native crash in delegate init.
_QNN_NPU_BACKEND = "htp"
_QNN_GPU_BACKEND = "gpu"

# Timeout for the subprocess delegate probe.
_DELEGATE_PROBE_TIMEOUT_S = 15


def _probe_delegate_load(delegate_lib: str, options: dict[str, str]) -> str | None:
    """Test-load *delegate_lib* in a throwaway subprocess.

    QNN delegate loading can emit an unrecoverable native signal (SIGSEGV /
    SIGBUS) that kills the process before Python can raise an exception.
    Running the probe in a child process isolates that crash: if the child
    dies unexpectedly we surface the failure as a ``RuntimeError`` in the
    parent instead of taking down the whole benchmark.

    ``faulthandler`` is enabled inside the probe so that any native crash
    prints a Python + C stack trace to the child's stderr, which is captured
    and included in the returned error string for maximum diagnostics.

    Returns:
        ``None`` on success (child exited 0), or a human-readable error string
        describing the failure.
    """
    opts_repr = repr(options)
    script = "\n".join([
        "import sys, faulthandler",
        "faulthandler.enable()",
        "try:",
        "    from ai_edge_litert.interpreter import load_delegate",
        "except ImportError:",
        "    from tensorflow.lite.python.interpreter import load_delegate",
        f"load_delegate({delegate_lib!r}, {opts_repr})",
        "sys.exit(0)",
    ])
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=_DELEGATE_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"probe timed out after {_DELEGATE_PROBE_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001
        return f"probe could not be started: {e}"

    if result.returncode == 0:
        return None

    # Decode any output the child produced before dying.
    stderr = result.stderr.decode(errors="replace").strip()
    stdout = result.stdout.decode(errors="replace").strip()
    detail = stderr or stdout or "(no output from probe)"

    rc = result.returncode
    if rc in (-11, 139):  # SIGSEGV
        return (
            f"native crash — SIGSEGV (exit {rc}) during delegate init;\n"
            f"  delegate={delegate_lib!r}\n"
            f"  options={options!r}\n"
            f"  child output:\n{detail}"
        )
    return f"delegate probe exited {rc}:\n{detail}"

# Try to import ai_edge_litert at module load time.  On dev machines this
# package is absent, so we fall back to tf.lite (which ships with tensorflow).
try:
    from ai_edge_litert.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = True
except ImportError:  # pragma: no cover
    from tensorflow.lite.python.interpreter import load_delegate as _load_delegate

    _have_ai_edge_litert = False
    logger.warning("ai_edge_litert not available — using tf.lite as fallback")


class QCS6490LiteRTBackend(LiteRTBackend):
    """TFLite runtime with QNN delegate for NPU/GPU acceleration on QCS6490.

    Extends the base LiteRTBackend to add Qualcomm QNN TFLite delegate
    support for Hexagon HTP (NPU) and Adreno GPU. When NPU/GPU is requested,
    the QNN delegate is loaded; CPU requests use no delegate (XNNPACK).
    """

    def _get_delegates(self) -> list:
        """Build the delegate list for the configured compute unit.

        Returns an empty list for CPU — no delegate loading is attempted,
        and XNNPACK acceleration is applied automatically by the runtime.

        For NPU/GPU, loads the QNN TFLite delegate from
        ``_QNN_DELEGATE_PATH`` with backend-specific options. Raises
        ``RuntimeError`` on failure so the caller can fall back to CPU.
        """
        # CPU path: no delegates needed.
        if self._unit == ComputeUnit.CPU:
            return []

        # NPU path: probe in a subprocess first to catch native crashes, then
        # load in the real process if the probe passes.
        if self._unit == ComputeUnit.NPU:
            npu_opts = {_QNN_BACKEND_KEY: _QNN_NPU_BACKEND}
            logger.debug(
                "[get_delegates] probing QNN NPU delegate: lib=%s options=%r",
                _QNN_DELEGATE_PATH,
                npu_opts,
            )
            probe_err = _probe_delegate_load(_QNN_DELEGATE_PATH, npu_opts)
            if probe_err:
                msg = f"NPU delegate unavailable: {probe_err}"
                logger.warning("[get_delegates] %s", msg)
                raise RuntimeError(msg)
            logger.debug("[get_delegates] NPU delegate probe passed — loading in main process")
            try:
                qnn = _load_delegate(_QNN_DELEGATE_PATH, npu_opts)
            except Exception as e:
                msg = f"NPU delegate unavailable: {e}"
                raise RuntimeError(msg) from e
            else:
                logger.info(
                    "[get_delegates] QNN delegate loaded from %s → Hexagon HTP/NPU",
                    _QNN_DELEGATE_PATH,
                )
                return [qnn]

        if self._unit == ComputeUnit.GPU:
            gpu_opts = {_QNN_BACKEND_KEY: _QNN_GPU_BACKEND}
            logger.debug(
                "[get_delegates] probing QNN GPU delegate: lib=%s options=%r",
                _QNN_DELEGATE_PATH,
                gpu_opts,
            )
            probe_err = _probe_delegate_load(_QNN_DELEGATE_PATH, gpu_opts)
            if probe_err:
                msg = f"GPU delegate unavailable: {probe_err}"
                logger.warning("[get_delegates] %s", msg)
                raise RuntimeError(msg)
            logger.debug("[get_delegates] GPU delegate probe passed — loading in main process")
            try:
                qnn = _load_delegate(_QNN_DELEGATE_PATH, gpu_opts)
            except Exception as e:
                msg = f"GPU delegate unavailable: {e}"
                raise RuntimeError(msg) from e
            else:
                logger.info(
                    "[get_delegates] QNN delegate loaded from %s → Adreno GPU",
                    _QNN_DELEGATE_PATH,
                )
                return [qnn]

        # Other units are not implemented.
        return []
