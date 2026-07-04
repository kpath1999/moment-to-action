"""MobileNet V2 image classification model."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import cv2
import numpy as np

from moment_to_action.hardware import ModelType
from moment_to_action.models.image.classification._base import ImageClassificationModel
from moment_to_action.models.image.classification._types import Classification

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.hardware import LoadedModel, Platform
    from moment_to_action.hardware._types import ComputeUnit, DataType
    from moment_to_action.metrics import MetricsCollector

_MNV2_INPUT_SIZE = 224
_MNV2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_MNV2_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MNV2_NUM_CLASSES = 1000


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis.

    Args:
        x: Input array of any shape.

    Returns:
        Array of same shape with values in ``(0, 1)`` summing to 1 over last axis.
    """
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class MobileNetV2Model(ImageClassificationModel):
    """MobileNet V2 image classifier (ImageNet-1K, 224x224 input).

    Supports ONNX (CPU via ONNX Runtime) and DLC (NPU via QAIRT) formats.
    The model is unloaded after construction; call :meth:`load` before inference.

    Args:
        variant: Registry variant key used to identify this instance.
        path: Path to the model directory (contains ``model.onnx`` or ``model.dlc``).
        model_type: Model file format — determines which backend methods to call.
        top_k: Number of top predictions returned by :meth:`post_proc`.
    """

    IMAGENET_LABELS: ClassVar[tuple[str, ...]] = ()
    """ImageNet-1K class labels loaded lazily from torchvision on first use."""

    def __init__(
        self,
        variant: str,
        path: Path,
        model_type: ModelType,
        data_type: DataType,
        top_k: int = 5,
        *,
        backends: dict[ComputeUnit, dict[str, str]],
        input_layout: str = "NCHW",
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize an unloaded MobileNetV2Model.

        Args:
            variant: Registry variant key.
            path: Path to the model directory containing ``model.onnx`` or ``model.dlc``.
            model_type: ``ModelType.ONNX`` or ``ModelType.DLC``.
            data_type: Quantization type (e.g. ``DataType.W8A8``); required for DLC variants.
            top_k: Number of top predictions to return from :meth:`post_proc`.
            backends: Compute unit → ``{"model": filename}`` mapping.  Keys
                present are the supported units; ``load()`` indexes this with
                the explicit ``unit`` argument.
            input_layout: Input tensor layout (unused by MobileNet V2, which
                always preprocesses to NCHW; accepted for interface uniformity).
            metrics: Metrics collector used to record ``MODEL_*`` spans.
        """
        super().__init__(
            variant,
            path,
            model_type,
            data_type,
            backends=backends,
            input_layout=input_layout,
            metrics=metrics,
        )
        self._top_k = top_k
        self._handle: LoadedModel | None = None

    @property
    def top_k(self) -> int:
        """Number of top predictions returned by :meth:`post_proc`."""
        return self._top_k

    @classmethod
    def _get_label(cls, class_id: int) -> str:
        """Return the human-readable label for a class index.

        Lazily loads ImageNet-1K labels from ``torchvision`` on first call.
        Falls back to ``"class_<id>"`` if torchvision is unavailable.

        Args:
            class_id: Integer class index in ``[0, 999]``.

        Returns:
            Human-readable class name string.
        """
        if not cls.IMAGENET_LABELS:
            try:
                from torchvision.models import MobileNet_V2_Weights  # noqa: PLC0415

                cls.IMAGENET_LABELS = tuple(MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"])
            except Exception:  # noqa: BLE001
                cls.IMAGENET_LABELS = tuple(f"class_{i}" for i in range(_MNV2_NUM_CLASSES))
        if class_id < len(cls.IMAGENET_LABELS):
            return cls.IMAGENET_LABELS[class_id]
        return f"class_{class_id}"

    def _load(self, platform: Platform, unit: ComputeUnit) -> None:
        """Load model weights onto the backend.

        Selects the artifact filename from the per-unit ``backends`` table
        using ``unit``.

        Args:
            platform: Hardware platform to load the model onto.
            unit: Compute unit to target.

        Raises:
            RuntimeError: If the model is already loaded.
            KeyError: If ``unit`` is not supported by this variant.
            ValueError: If ``unit`` is not available on ``platform``.
        """
        if self._platform is not None:
            msg = f"{type(self).__name__} is already loaded; call unload() first"
            raise RuntimeError(msg)
        arts = self._backends[unit]

        # Do the load
        if self._model_type is ModelType.ONNX:
            dtype = self._data_type
            self._handle = platform.load_onnx(unit, self._artifact_path(arts["model"]), dtype=dtype)
        else:
            dtype = self._data_type
            self._handle = platform.load_dlc(unit, self._artifact_path(arts["model"]), dtype=dtype)
        self._platform = platform

    def _unload(self) -> None:
        """Release backend resources and reset internal state."""
        if self._handle is not None:
            self._handle.unload()
        self._platform = None
        self._handle = None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalize, and batch a raw BGR frame for MobileNet V2 inference.

        Args:
            frame: Raw BGR image (HxWxC, uint8).

        Returns:
            Float32 NCHW tensor of shape ``(1, 3, 224, 224)`` normalized with
            ImageNet mean and standard deviation.
        """
        resized = cv2.resize(frame, (_MNV2_INPUT_SIZE, _MNV2_INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - _MNV2_MEAN) / _MNV2_STD
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Run MobileNet V2 forward pass.

        Args:
            prepared: Batch tensor from :meth:`prepare`.

        Returns:
            List containing a single ``(1, 1000)`` float32 logits array.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self._handle is None:
            msg = "MobileNetV2Model.load() must be called before run()"
            raise RuntimeError(msg)
        if self._model_type is ModelType.ONNX:
            return cast("list[np.ndarray]", self._handle.run(prepared))
        dlc_out = cast("dict[str, np.ndarray]", self._handle.run(prepared))
        return [next(iter(dlc_out.values()))]

    def _post_proc(self, raw: list[np.ndarray]) -> list[Classification]:
        """Decode logits into top-k classification results.

        Args:
            raw: Value returned by :meth:`run` — list containing a
                ``(1, 1000)`` logits array.

        Returns:
            Up to :attr:`top_k` :class:`~.Classification` objects ordered by
            descending confidence.
        """
        if not raw or raw[0].size == 0:
            return []
        logits = raw[0][0].astype(np.float32)
        probs = _softmax(logits)
        top_indices = np.argsort(probs)[::-1][: self._top_k]
        return [
            Classification(
                label=self._get_label(int(idx)),
                confidence=float(probs[idx]),
                class_id=int(idx),
            )
            for idx in top_indices
        ]
