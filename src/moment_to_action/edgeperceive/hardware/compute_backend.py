"""Hardware Abstraction Layer for the QCS6490.

This is the single place that knows about specific runtimes.
Models call this — never TFLite or SNPE directly.
Swap the runtime or the chip: change this file only.

Runtime note:
  tf.lite.Interpreter is deprecated as of TF 2.18 and will be removed in TF 2.20.
  We use ai_edge_litert.interpreter instead, which is the official replacement.
  Install: pip install ai-edge-litert

Multi-input models (e.g. MobileCLIP):
  Pass a dict mapping input name → tensor to backend.run().
  Single-input models can still pass a plain ndarray for convenience.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from moment_to_action.edgeperceive.hardware.types import ComputeUnit

logger = logging.getLogger(__name__)

# Type alias for flexible input: single tensor or named dict of tensors
ModelInput = np.ndarray | dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# Power monitor
# ---------------------------------------------------------------------------


@dataclass
class PowerSample:
    """A single power measurement sample."""

    timestamp: float
    compute_unit: ComputeUnit
    power_mw: float
    utilization_pct: float


class PowerMonitor:
    """Reads power draw from sysfs on QCS6490.

    Falls back to estimates when sensors are unavailable.
    """

    SYSFS_POWER_PATH = "/sys/class/power_supply"

    def __init__(self) -> None:
        self._hw_available = self._check_hw_sensors()

    def _check_hw_sensors(self) -> bool:
        from pathlib import Path

        return Path(self.SYSFS_POWER_PATH).exists()

    def sample(self, unit: ComputeUnit) -> PowerSample:
        """Take a power measurement for the given compute unit."""
        if self._hw_available:
            return self._read_hw_sensor(unit)
        return self._estimate(unit)

    def _read_hw_sensor(self, unit: ComputeUnit) -> PowerSample:
        try:
            with open(f"{self.SYSFS_POWER_PATH}/battery/power_now") as f:  # noqa: PTH123
                power_uw = int(f.read().strip())
            return PowerSample(
                timestamp=time.time(),
                compute_unit=unit,
                power_mw=power_uw / 1000.0,
                utilization_pct=0.0,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning("HW power sensor read failed: %s", e)
            return self._estimate(unit)

    def _estimate(self, unit: ComputeUnit) -> PowerSample:
        estimates = {
            ComputeUnit.NPU: 500.0,
            ComputeUnit.GPU: 800.0,
            ComputeUnit.DSP: 150.0,
            ComputeUnit.CPU: 300.0,
        }
        return PowerSample(
            timestamp=time.time(),
            compute_unit=unit,
            power_mw=estimates.get(unit, 300.0),
            utilization_pct=0.0,
        )


# ---------------------------------------------------------------------------
# Runtime backends
# ---------------------------------------------------------------------------


class InferenceBackend(ABC):
    """Abstract runtime. One implementation per runtime (LiteRT, SNPE, ONNX)."""

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load a model from the given path and return a handle."""
        ...

    @abstractmethod
    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference and return output tensors."""
        ...

    @abstractmethod
    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend targets."""
        ...


def _load_interpreter(model_path: str, delegates: list) -> Any:
    """Load a TFLite model using ai_edge_litert.

    If delegate application fails (e.g. FP32 model on NPU), falls back to CPU.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter

        interp = Interpreter(model_path=model_path, experimental_delegates=delegates)
        logger.info("Loaded %s via ai_edge_litert", model_path)
    except RuntimeError as e:
        if delegates:
            logger.warning("Delegate failed (%s) — retrying on CPU", e)
            from ai_edge_litert.interpreter import Interpreter

            interp = Interpreter(model_path=model_path)
        else:
            raise
    except ImportError:
        logger.warning("ai_edge_litert not installed, falling back to tf.lite")
        import tensorflow as tf

        try:
            interp = tf.lite.Interpreter(model_path=model_path, experimental_delegates=delegates)
        except (RuntimeError, ValueError, OSError):
            interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _set_inputs(interp: Any, inputs: ModelInput) -> None:
    """Feed inputs into an interpreter.

    Single tensor (most models):
        inputs = np.array([...])
        → fed to input index 0

    Multiple tensors (MobileCLIP, multi-modal models):
        inputs = {
            'serving_default_args_0:0': image_tensor,   # [1, 3, 256, 256] float32
            'serving_default_args_1:0': token_tensor,   # [1, 77] int64
        }
        → each tensor matched to its named input slot
    """
    input_details = interp.get_input_details()

    if isinstance(inputs, np.ndarray):
        # Convenience: single array → slot 0
        interp.set_tensor(input_details[0]["index"], inputs)
        return

    # Dict path: match by name
    name_to_detail = {d["name"]: d for d in input_details}
    for name, tensor in inputs.items():
        if name not in name_to_detail:
            available = list(name_to_detail.keys())
            msg = f"Input name '{name}' not found in model. Available: {available}"
            raise KeyError(msg)
        detail = name_to_detail[name]
        # Validate dtype to catch float32/int64 mismatches early
        expected_dtype = detail["dtype"]
        if tensor.dtype != expected_dtype:
            msg = (
                f"Input '{name}' dtype mismatch: got {tensor.dtype}, model expects {expected_dtype}"
            )
            raise TypeError(msg)
        interp.set_tensor(detail["index"], tensor)


class LiteRTBackend(InferenceBackend):
    """ai_edge_litert runtime — primary backend for NPU inference.

    Replaces the deprecated tf.lite.Interpreter.
    Falls back to tf.lite if ai_edge_litert is not installed.

    NNAPI delegate routes to Hexagon NPU on QCS6490.
    """

    def __init__(self, compute_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self._unit = compute_unit
        self._interpreter_cache: dict[str, Any] = {}

    def load_model(self, model_path: str) -> Any:
        """Load a TFLite model, caching interpreters by path."""
        if model_path in self._interpreter_cache:
            logger.debug("Model cache hit: %s", model_path)
            return self._interpreter_cache[model_path]

        interp = _load_interpreter(model_path, self._get_delegates())
        self._interpreter_cache[model_path] = interp
        logger.info("Loaded %s on %s", model_path, self._unit.name)
        return interp

    def _get_delegates(self) -> list:
        delegates = []
        try:
            from ai_edge_litert.interpreter import load_delegate

            if self._unit == ComputeUnit.NPU:
                qnn = load_delegate("/usr/lib/libQnnTFLiteDelegate.so")
                delegates.append(qnn)
                logger.info("QNN delegate loaded → Hexagon HTP/NPU")
        except Exception as e:  # noqa: BLE001
            # Delegate failed → fallback to CPU interpreter
            logger.warning("NPU delegate unavailable (%s), falling back to CPU XNNPACK", e)
        return delegates

    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference. Returns all output tensors as a list.

        Callers index by position: outputs[0], outputs[1], ...
        """
        _set_inputs(model_handle, inputs)
        model_handle.invoke()
        output_details = model_handle.get_output_details()
        return [model_handle.get_tensor(d["index"]) for d in output_details]

    def get_supported_unit(self) -> ComputeUnit:
        """Return the compute unit this backend targets."""
        return self._unit


class ONNXBackend(InferenceBackend):
    """ONNX Runtime backend — used for YOLO and other ONNX models.

    Runs on CPU by default; ExecutionProvider can be swapped for GPU/NPU.
    Install: pip install onnxruntime
    """

    def __init__(self) -> None:
        self._session_cache: dict[str, Any] = {}

    def load_model(self, model_path: str) -> Any:
        """Load an ONNX model, caching sessions by path."""
        if model_path in self._session_cache:
            logger.debug("ONNX cache hit: %s", model_path)
            return self._session_cache[model_path]
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
        except ImportError as err:
            msg = "onnxruntime not installed. Run: pip install onnxruntime"
            raise RuntimeError(msg) from err
        self._session_cache[model_path] = session
        logger.info("Loaded %s via onnxruntime", model_path)
        return session

    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run ONNX inference and return output tensors."""
        input_details = model_handle.get_inputs()

        feed = {input_details[0].name: inputs} if isinstance(inputs, np.ndarray) else inputs

        return model_handle.run(None, feed)

    def get_supported_unit(self) -> ComputeUnit:
        """Return CPU as the supported compute unit."""
        return ComputeUnit.CPU


class CPUBackend(InferenceBackend):
    """Pure CPU — no acceleration. Used for dev/testing without NPU hardware."""

    def load_model(self, model_path: str) -> Any:
        """Load a TFLite model on CPU without delegates."""
        return _load_interpreter(model_path, delegates=[])

    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run CPU inference and return output tensors."""
        _set_inputs(model_handle, inputs)
        model_handle.invoke()
        output_details = model_handle.get_output_details()
        return [model_handle.get_tensor(d["index"]) for d in output_details]

    def get_supported_unit(self) -> ComputeUnit:
        """Return CPU as the supported compute unit."""
        return ComputeUnit.CPU


# ---------------------------------------------------------------------------
# ComputeBackend — what models actually call
# ---------------------------------------------------------------------------


class ComputeBackend:
    """HAL entry point. Models call this, never LiteRT or SNPE directly.

    Usage:
        backend = ComputeBackend(preferred_unit=ComputeUnit.NPU)
        handle  = backend.load_model("mobileclip.tflite")
        outputs = backend.run(handle, {
            'serving_default_args_0:0': image_tensor,
            'serving_default_args_1:0': token_tensor,
        })
        image_emb = outputs[1]   # [1, 512]
        text_emb  = outputs[0]   # [1, 512]
    """

    def __init__(self, preferred_unit: ComputeUnit = ComputeUnit.NPU) -> None:
        self.preferred_unit = preferred_unit
        self.power_monitor = PowerMonitor()
        self._backend = self._select_backend(preferred_unit)
        self._onnx_backend = ONNXBackend()
        logger.info("ComputeBackend: %s", self._backend.get_supported_unit().name)

    def _select_backend(self, unit: ComputeUnit) -> InferenceBackend:
        # ONNX models are detected at load time — see load_model()
        if unit in (ComputeUnit.NPU, ComputeUnit.GPU):
            return LiteRTBackend(compute_unit=unit)
        return CPUBackend()

    def load_model(self, model_path: str) -> Any:
        """Load model. Automatically uses ONNXBackend for .onnx files."""
        if model_path and model_path.endswith(".onnx"):
            if not isinstance(self._backend, ONNXBackend):
                self._onnx_backend = ONNXBackend()
            return self._onnx_backend.load_model(model_path)
        return self._backend.load_model(model_path)

    def run(self, model_handle: Any, inputs: ModelInput) -> list[np.ndarray]:
        """Run inference. Routes to the correct backend based on model handle type."""
        try:
            import onnxruntime as ort

            if isinstance(model_handle, ort.InferenceSession):
                return self._onnx_backend.run(model_handle, inputs)
        except ImportError:
            pass
        return self._backend.run(model_handle, inputs)

    def get_input_details(self, model_handle: Any) -> list[dict]:
        """Inspect model input slots — useful for debugging and validation."""
        return model_handle.get_input_details()

    def get_output_details(self, model_handle: Any) -> list[dict]:
        """Inspect model output slots."""
        return model_handle.get_output_details()

    @property
    def active_unit(self) -> ComputeUnit:
        """Return the compute unit of the active backend."""
        return self._backend.get_supported_unit()

    def benchmark(
        self,
        model_handle: Any,
        inputs: ModelInput,
        n_runs: int = 20,
    ) -> dict:
        """Run inference n_runs times and return latency statistics."""
        latencies = []
        for _ in range(n_runs):
            t = time.perf_counter()
            self._backend.run(model_handle, inputs)
            latencies.append((time.perf_counter() - t) * 1000)

        arr = np.array(latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "compute_unit": self.active_unit.name,
            "n_runs": n_runs,
        }
