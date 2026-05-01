"""LiteRT backend for the QCS6490 platform with QNN accelerator support.

Extends the shared LiteRTBackend to add QNN delegate support for NPU/GPU
inference on the Qualcomm QCS6490.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

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
_QNN_GPU_OPTIONS_ENV = "MOMENT_TO_ACTION_QNN_GPU_DELEGATE_OPTIONS"

# Timeout for the subprocess delegate probe.
_DELEGATE_PROBE_TIMEOUT_S = 15

_ADSP_LIBRARY_DEFAULT = ";".join(  # noqa: FLY002
    (
        "/usr/lib/rfsa/adsp",
        "/usr/lib/rfsa/adsp/hexagon-v75",
        "/usr/lib/rfsa/adsp/hexagon-v73",
        "/usr/lib/rfsa/adsp/hexagon-v68",
    )
)


def _parse_delegate_options(env_var: str) -> dict[str, str]:
    """Parse comma-separated delegate options from *env_var*.

    Format: ``key=value,key2=value2``. Empty entries are ignored.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return {}

    options: dict[str, str] = {}
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            continue
        if "=" not in entry:
            msg = f"Invalid delegate option {entry!r} in {env_var}; expected key=value"
            raise ValueError(msg)
        key, value = entry.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            msg = f"Invalid delegate option {entry!r} in {env_var}; key is empty"
            raise ValueError(msg)
        options[key] = value
    return options


def _is_in_fastrpc_group() -> bool:
    """Check if the current process has fastrpc group membership.

    Returns:
        True if the process is in the fastrpc group (either as primary or supplementary).
    """
    try:
        import grp

        # Get the fastrpc group ID
        try:
            fastrpc_gid = grp.getgrnam("fastrpc").gr_gid
        except KeyError:
            # fastrpc group doesn't exist on this system
            return False

        # Check primary group
        if os.getgid() == fastrpc_gid:
            return True

        # Check supplementary groups
        return fastrpc_gid in os.getgroups()
    except Exception:  # noqa: BLE001
        # If we can't determine group membership, assume we don't have it
        return False


def _ensure_fastrpc_permissions() -> None:
    """Ensure the process has fastrpc group permissions for NPU access.

    Library code should fail explicitly here rather than trying to re-execute
    the current process behind the caller's back.
    """
    if _is_in_fastrpc_group():
        return

    msg = (
        "NPU access requires membership in the 'fastrpc' group. "
        "Re-run your command under that group, for example: "
        "sg fastrpc -c '<your command>'"
    )
    raise RuntimeError(msg)


def _collect_htp_diagnostics() -> str:
    """Return a compact snapshot of the HTP transport environment."""
    adsp_library_path = os.environ.get("ADSP_LIBRARY_PATH", "")

    fastrpc_nodes = [
        path
        for path in (
            "/dev/fastrpc-cdsp",
            "/dev/fastrpc-cdsp-secure",
            "/dev/fastrpc-adsp-secure",
        )
        if Path(path).exists()
    ]

    remoteprocs: list[str] = []
    for remoteproc in sorted(Path("/sys/class/remoteproc").glob("remoteproc*")):
        name_path = remoteproc / "name"
        state_path = remoteproc / "state"
        firmware_path = remoteproc / "firmware"
        if not (name_path.exists() and state_path.exists()):
            continue
        name = name_path.read_text().strip()
        state = state_path.read_text().strip()
        firmware = firmware_path.read_text().strip() if firmware_path.exists() else "?"
        remoteprocs.append(f"{name}:{state}:{firmware}")

    skels = sorted(str(path) for path in Path("/usr/lib/rfsa/adsp").glob("libQnnHtp*Skel.so"))

    return (
        "HTP diagnostics: "
        f"ADSP_LIBRARY_PATH={adsp_library_path or '<unset>'}; "
        f"recommended_ADSP_LIBRARY_PATH={_ADSP_LIBRARY_DEFAULT}; "
        f"fastrpc_nodes={fastrpc_nodes or ['<none>']}; "
        f"remoteprocs={remoteprocs or ['<none>']}; "
        f"htp_skels={skels or ['<none>']}"
    )


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
    script = "\n".join(
        [
            "import sys, faulthandler",
            "faulthandler.enable()",
            "try:",
            "    from ai_edge_litert.interpreter import load_delegate",
            "except ImportError:",
            "    from tensorflow.lite.python.interpreter import load_delegate",
            f"load_delegate({delegate_lib!r}, {opts_repr})",
            "sys.exit(0)",
        ]
    )
    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=_DELEGATE_PROBE_TIMEOUT_S,
            check=False,
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

    def _get_delegates(self) -> list:  # noqa: C901
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
            # Ensure we have fastrpc group permissions before attempting NPU access.
            _ensure_fastrpc_permissions()

            # Ensure ADSP_LIBRARY_PATH is set for HTP skeleton library discovery.
            if not os.environ.get("ADSP_LIBRARY_PATH"):
                os.environ["ADSP_LIBRARY_PATH"] = _ADSP_LIBRARY_DEFAULT
                logger.debug(
                    "[get_delegates] ADSP_LIBRARY_PATH was unset; applying default=%s",
                    _ADSP_LIBRARY_DEFAULT,
                )

            npu_opts = {
                _QNN_BACKEND_KEY: _QNN_NPU_BACKEND,
                # Burst mode (3) + HMX convolution give the best latency on Hexagon HTP.
                # htp_performance_mode accepts the integer enum value:
                #   0=default, 1=balanced, 2=low_power, 3=burst, 4=high_performance
                # Passing the string name "burst" causes SIGABRT in this delegate version.
                "htp_performance_mode": "3",
                "htp_use_conv_hmx": "1",
            }
            logger.debug("[get_delegates] %s", _collect_htp_diagnostics())
            logger.debug(
                "[get_delegates] probing QNN NPU delegate: lib=%s options=%r",
                _QNN_DELEGATE_PATH,
                npu_opts,
            )
            probe_err = _probe_delegate_load(_QNN_DELEGATE_PATH, npu_opts)
            if probe_err:
                msg = f"NPU delegate unavailable: {probe_err}\n{_collect_htp_diagnostics()}"
                logger.warning("[get_delegates] %s", msg)
                raise RuntimeError(msg)
            logger.debug("[get_delegates] NPU delegate probe passed — loading in main process")
            try:
                qnn = _load_delegate(_QNN_DELEGATE_PATH, npu_opts)
            except Exception as e:
                msg = f"NPU delegate unavailable: {e}\n{_collect_htp_diagnostics()}"
                raise RuntimeError(msg) from e
            else:
                logger.info(
                    "[get_delegates] QNN delegate loaded from %s → Hexagon HTP/NPU",
                    _QNN_DELEGATE_PATH,
                )
                return [qnn]

        if self._unit == ComputeUnit.GPU:
            gpu_opts = {
                _QNN_BACKEND_KEY: _QNN_GPU_BACKEND,
                **_parse_delegate_options(_QNN_GPU_OPTIONS_ENV),
            }
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
                    "[get_delegates] QNN delegate loaded from %s → Adreno GPU options=%r",
                    _QNN_DELEGATE_PATH,
                    gpu_opts,
                )
                return [qnn]

        # Other units are not implemented.
        return []
