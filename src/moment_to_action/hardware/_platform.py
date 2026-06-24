"""Platform — concrete hardware entry point.

``Platform`` is the **only** class the rest of the codebase imports from
``hardware/``.  It detects the current hardware at construction time, builds
the right :class:`~moment_to_action.hardware._backend.ComputeBackend` instances
for each available compute unit, and delegates all load calls.

Usage::

    platform = Platform(config, preferred_unit=ComputeUnit.NPU)
    model = platform.load_tflite(ComputeUnit.NPU, "mobileclip.tflite")
    outputs = model.run(image_tensor)
    model.unload()
    # or:
    with platform.load_dlc(ComputeUnit.NPU, "detector.dlc", dtype=DataType.W8A8) as model:
        result = model.run(inputs)
"""

from __future__ import annotations

import functools
import logging
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from moment_to_action.hardware._types import (
    BenchmarkResult,
    ComputeUnit,
    DataType,
    ModelType,
    PlatformType,
)

if TYPE_CHECKING:
    import os

    from moment_to_action.config import AppConfig
    from moment_to_action.hardware._backend import ComputeBackend
    from moment_to_action.hardware._loaded_model import LoadedModel
    from moment_to_action.hardware._resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)

# Qualcomm sysfs file containing the SoC/machine name.
_QCOM_SOC_NAME_FILE = Path("/sys/devices/soc0/machine")


@functools.cache
def _detect_platform() -> PlatformType:
    """Detect the current hardware platform (internal, cached).

    Reads ``/sys/devices/soc0/machine`` for Qualcomm SoCs, then falls back
    to ``platform.machine()`` + ``platform.system()``.

    Returns:
        The detected :class:`PlatformType`.

    Raises:
        RuntimeError: If the SoC/platform cannot be identified.
    """
    soc_name = None

    if _QCOM_SOC_NAME_FILE.exists():
        soc_name = _QCOM_SOC_NAME_FILE.read_text().strip().upper()
        logger.debug("Detected SoC: %r", soc_name)
        if "QCS6490" in soc_name:
            return PlatformType.QCS6490

    machine = platform.machine().lower()
    system = platform.system().lower()
    logger.debug("CPU architecture: %r", machine)

    if machine in {"x86_64", "amd64"}:
        logger.info("Detected x86_64 architecture")
        return PlatformType.X86_64

    if machine in {"arm64", "aarch64"} and system == "darwin":
        logger.info("Detected macOS arm64 architecture")
        return PlatformType.MACOS_ARM64

    msg = (
        f"Unrecognised platform. SoC={soc_name!r}, arch={machine!r}, os={system!r}. "
        "Add a new PlatformType member and backends to support this hardware."
    )
    raise RuntimeError(msg)


class Platform:
    """Hardware entry point — detects platform and routes inference calls.

    Construct once and reuse.  :meth:`__init__` calls :func:`_detect_platform`
    internally and eagerly builds all available
    :class:`~moment_to_action.hardware._backend.ComputeBackend` instances.
    Callers always specify the compute unit explicitly on each load call.

    Args:
        config: Application configuration used to resolve llama-server path and
            port defaults when calling :meth:`load_llama_cpp`.

    Raises:
        RuntimeError: If the current platform cannot be detected.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize Platform, detect hardware, and build all available backends.

        Args:
            config: Application configuration.
        """
        self._config = config
        self._platform_type = _detect_platform()
        self._backends: dict[ComputeUnit, ComputeBackend]
        self._resource_monitor: ResourceMonitor

        match self._platform_type:
            case PlatformType.QCS6490:
                self._init_qcs6490()
            case PlatformType.X86_64:
                self._init_x86_64()
            case PlatformType.MACOS_ARM64:
                self._init_macos_arm64()

        logger.info(
            "Platform: %s  available=%s",
            self._platform_type.name,
            ", ".join(u.name for u in self._backends),
        )

    # ------------------------------------------------------------------
    # Platform-specific initializers
    # ------------------------------------------------------------------

    def _init_qcs6490(self) -> None:
        """Build QCS6490 backends: CPU always, HTP (NPU) and GPU if available."""
        from moment_to_action.hardware._platforms.qcs6490._cpu_backend import (  # noqa: PLC0415
            QCS6490CPUBackend,
        )
        from moment_to_action.hardware._platforms.qcs6490._resources import (  # noqa: PLC0415
            QCS6490ResourceMonitor,
        )

        self._resource_monitor = QCS6490ResourceMonitor()
        self._backends = {ComputeUnit.CPU: QCS6490CPUBackend()}
        self._try_add_htp_backend()
        self._try_add_qcs6490_gpu_backend()

    def _try_add_htp_backend(self) -> None:
        """Try to register the QCS6490 HTP (NPU) backend.

        If unavailable, NPU is simply not registered — callers get a
        :class:`ValueError` if they request it.
        """
        try:
            from moment_to_action.hardware._platforms.qcs6490._htp_backend import (  # noqa: PLC0415
                QCS6490HTPBackend,
            )

            self._backends[ComputeUnit.NPU] = QCS6490HTPBackend()
        except Exception as e:  # noqa: BLE001
            logger.warning("HTP backend unavailable (%s) — NPU not registered", e)

    def _try_add_qcs6490_gpu_backend(self) -> None:
        """Try to register the QCS6490 GPU backend.

        If unavailable, GPU is simply not registered — callers get a
        :class:`ValueError` if they request it.
        """
        try:
            from moment_to_action.hardware._platforms.qcs6490._gpu_backend import (  # noqa: PLC0415
                QCS6490GPUBackend,
            )

            self._backends[ComputeUnit.GPU] = QCS6490GPUBackend()
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU backend unavailable (%s) — GPU not registered", e)

    def _init_x86_64(self) -> None:
        """Build x86_64 backends: CPU always, GPU if CUDA is available."""
        from moment_to_action.hardware._platforms.x86_64._cpu_backend import (  # noqa: PLC0415
            X86_64CPUBackend,
        )
        from moment_to_action.hardware._platforms.x86_64._resources import (  # noqa: PLC0415
            X86_64ResourceMonitor,
        )

        self._resource_monitor = X86_64ResourceMonitor()
        self._backends = {ComputeUnit.CPU: X86_64CPUBackend()}
        self._try_add_x86_64_gpu_backend()

    def _try_add_x86_64_gpu_backend(self) -> None:
        """Try to register the x86_64 GPU (CUDA) backend.

        Fails silently if CUDA is unavailable.
        """
        try:
            from moment_to_action.hardware._platforms.x86_64._gpu_backend import (  # noqa: PLC0415
                X86_64GPUBackend,
            )

            self._backends[ComputeUnit.GPU] = X86_64GPUBackend()
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU backend unavailable (%s) — GPU not registered", e)

    def _init_macos_arm64(self) -> None:
        """Build macOS arm64 backends: CPU always, GPU if MPS is available."""
        from moment_to_action.hardware._platforms.macos_arm64._cpu_backend import (  # noqa: PLC0415
            MacOSARM64CPUBackend,
        )
        from moment_to_action.hardware._platforms.macos_arm64._resources import (  # noqa: PLC0415
            MacOSARM64ResourceMonitor,
        )

        self._resource_monitor = MacOSARM64ResourceMonitor()
        self._backends = {ComputeUnit.CPU: MacOSARM64CPUBackend()}
        self._try_add_macos_arm64_gpu_backend()

    def _try_add_macos_arm64_gpu_backend(self) -> None:
        """Try to register the macOS arm64 GPU (MPS) backend.

        Fails silently if MPS is unavailable.
        """
        try:
            from moment_to_action.hardware._platforms.macos_arm64._gpu_backend import (  # noqa: PLC0415
                MacOSARM64GPUBackend,
            )

            self._backends[ComputeUnit.GPU] = MacOSARM64GPUBackend()
        except Exception as e:  # noqa: BLE001
            logger.warning("GPU backend unavailable (%s) — GPU not registered", e)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def platform_type(self) -> PlatformType:
        """The detected platform type.

        Returns:
            The :class:`PlatformType` of this machine.
        """
        return self._platform_type

    @property
    def supported_units(self) -> set[ComputeUnit]:
        """Compute units available on this platform instance.

        Returns:
            Set of ``ComputeUnit`` members for which a backend is registered.
        """
        return set(self._backends)

    @property
    def resource_monitor(self) -> ResourceMonitor:
        """The platform resource monitor (power / utilisation sampling).

        Returns:
            The :class:`~moment_to_action.hardware._resource_monitor.ResourceMonitor`
            for this platform.
        """
        return self._resource_monitor

    # ------------------------------------------------------------------
    # Load methods — dispatch to the appropriate backend
    # ------------------------------------------------------------------

    def _backend_for(self, unit: ComputeUnit) -> ComputeBackend:
        """Return the backend for *unit*, raising if unavailable.

        Args:
            unit: The requested compute unit.

        Returns:
            The registered :class:`ComputeBackend` for *unit*.

        Raises:
            ValueError: If no backend is registered for *unit*.
        """
        backend = self._backends.get(unit)
        if backend is None:
            available = ", ".join(u.name for u in self._backends)
            msg = (
                f"{unit.name} is not available on {self._platform_type.name}. "
                f"Available units: {available}"
            )
            raise ValueError(msg)
        return backend

    def supports(
        self,
        unit: ComputeUnit,
        *,
        model_type: ModelType,
        data_type: DataType | None = None,
    ) -> bool:
        """Return True if ``unit`` has a backend supporting ``model_type`` (and ``data_type``).

        Args:
            unit: Compute unit to check.
            model_type: The model format to test (e.g. ``ModelType.DLC``).
            data_type: Optional quantization type to also validate.

        Returns:
            True if the platform can load this combination on ``unit``, False otherwise.
        """
        try:
            backend = self._backend_for(unit)
        except ValueError:
            return False
        if model_type not in backend.supported_formats:
            return False
        return data_type is None or data_type in backend.supported_dtypes

    def load_tflite(
        self, unit: ComputeUnit, path: str | os.PathLike[str], *, dtype: DataType
    ) -> LoadedModel:
        """Load a TFLite model on the specified compute unit.

        Args:
            unit: The compute unit to run this model on.
            path: Path to the ``.tflite`` model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` ready for inference.

        Raises:
            ValueError: If *unit* is not available on this platform.
            NotImplementedError: If the backend does not support TFLITE.
        """
        return self._backend_for(unit).load_tflite(path, dtype=dtype)

    def load_onnx(
        self, unit: ComputeUnit, path: str | os.PathLike[str], *, dtype: DataType
    ) -> LoadedModel:
        """Load an ONNX model on the specified compute unit.

        Args:
            unit: The compute unit to run this model on.
            path: Path to the ``.onnx`` model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` ready for inference.

        Raises:
            ValueError: If *unit* is not available on this platform.
            NotImplementedError: If the backend does not support ONNX.
        """
        return self._backend_for(unit).load_onnx(path, dtype=dtype)

    def load_dlc(
        self, unit: ComputeUnit, path: str | os.PathLike[str], *, dtype: DataType
    ) -> LoadedModel:
        """Load a DLC model on the specified compute unit.

        Args:
            unit: The compute unit to run this model on.
            path: Path to the ``.dlc`` model file.
            dtype: Quantization type of the model (e.g. ``DataType.W8A8``).

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` ready for inference.

        Raises:
            ValueError: If *unit* is not available on this platform.
            NotImplementedError: If the backend does not support DLC.
        """
        return self._backend_for(unit).load_dlc(path, dtype=dtype)

    def load_torch(
        self, unit: ComputeUnit, path: str | os.PathLike[str], *, dtype: DataType
    ) -> LoadedModel:
        """Load a PyTorch model on the specified compute unit.

        Args:
            unit: The compute unit to run this model on.
            path: Path to the saved model file.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` ready for inference.

        Raises:
            ValueError: If *unit* is not available on this platform.
            NotImplementedError: If the backend does not support TORCH.
        """
        return self._backend_for(unit).load_torch(path, dtype=dtype)

    def load_llama_cpp(
        self,
        unit: ComputeUnit,
        path: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        server_path: str | os.PathLike[str] | None = None,
        port: int | None = None,
        dtype: DataType,
    ) -> LoadedModel:
        """Load a llama.cpp GGUF model on the specified compute unit.

        ``server_path`` and ``port`` fall back to ``AppConfig.llama_server_path``
        and ``AppConfig.llama_server_port`` when not explicitly provided.

        Args:
            unit: The compute unit to run this model on.
            path: Path to the ``.gguf`` model file.
            mmproj: Optional path to the multimodal projector file.
            server_path: Path to the ``llama-server`` binary. Defaults to
                ``config.llama_server_path``.
            port: Port for llama-server. Defaults to ``config.llama_server_port``.
            dtype: Data type of the model (e.g. ``DataType.FP32``).

        Returns:
            A :class:`~moment_to_action.hardware.LoadedModel` ready for inference.

        Raises:
            ValueError: If *unit* is not available on this platform.
            NotImplementedError: If the backend does not support LLAMA_CPP.
        """
        effective_server = server_path or self._config.llama_server_path
        effective_port = port or self._config.llama_server_port
        return self._backend_for(unit).load_llama_cpp(
            path, mmproj=mmproj, server_path=effective_server, port=effective_port, dtype=dtype
        )

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        model: LoadedModel,
        inputs: object,
        n_runs: int = 20,
    ) -> BenchmarkResult:
        """Run inference *n_runs* times and return latency statistics.

        Args:
            model: A loaded model returned by one of the ``load_*`` methods.
            inputs: Inputs to pass on each run.
            n_runs: Number of inference repetitions.

        Returns:
            A :class:`BenchmarkResult` with latency percentiles and metadata.
        """
        latencies = np.empty(n_runs, dtype=np.float64)
        for i in range(n_runs):
            t = time.perf_counter()
            model.run(inputs)
            latencies[i] = (time.perf_counter() - t) * 1000.0

        return BenchmarkResult(
            mean_ms=float(np.mean(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            p99_ms=float(np.percentile(latencies, 99)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            compute_unit=model.unit.name,
            n_runs=n_runs,
        )
