"""Abstract base class for image detection models."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from moment_to_action.models.image._base import ImageModel
from moment_to_action.models.image.detection._types import Detection


class ImageDetectionModel(ImageModel[list[np.ndarray], Detection]):
    """Abstract base for models that detect objects in images.

    Fixes ``_RawOutputT=list[np.ndarray]`` and ``_ResultT=Detection``, so:

    - :meth:`run` returns ``list[np.ndarray]``
    - :meth:`post_proc` takes ``list[np.ndarray]`` and returns ``list[Detection]``

    Provides a concrete :meth:`verify_outputs` implementation that compares
    raw element-wise outputs and decoded detection labels against reference data.
    """

    @abstractmethod
    def _post_proc(self, raw: list[np.ndarray]) -> list[Detection]:
        """Decode raw model output into a list of detections.

        Args:
            raw: Output returned by :meth:`~moment_to_action.models.image.ImageModel._run`.

        Returns:
            List of :class:`~moment_to_action.models.image.detection.Detection` objects.
        """
        ...

    def verify_outputs(
        self,
        inputs: np.ndarray,
        ref_outputs: list[np.ndarray],
        *,
        tol: float,
        is_npu: bool,
    ) -> tuple[bool, str]:
        """Verify model outputs against reference data.

        For CPU/GPU: checks raw element-wise diff ≤ ``tol`` AND decoded
        detection label sets match.  For NPU: skips raw diff (INT8 quantisation
        noise dominates) and checks decoded labels only.

        Args:
            inputs: Input array of shape ``(N, ...)``.
            ref_outputs: List of reference output arrays, each of shape ``(N, ...)``.
            tol: Max absolute element-wise error for CPU/GPU raw comparison.
            is_npu: When True, skip raw diff check.

        Returns:
            ``(passed, fail_reason)``.  ``passed`` is True when all samples
            pass; ``fail_reason`` is empty on success or describes the first
            failure encountered.
        """
        for i in range(len(inputs)):
            inp = inputs[i : i + 1]
            act_raw = self._run(inp)

            if not is_npu:
                for k, (act_t, ref_t) in enumerate(zip(act_raw, ref_outputs, strict=False)):
                    ref_row = ref_t[i : i + 1]
                    max_err = float(
                        np.max(np.abs(act_t.astype(np.float32) - ref_row.astype(np.float32)))
                    )
                    if max_err > tol:
                        return False, f"output_{k}[{i}] max_err={max_err:.4f} > tol={tol}"

            ref_raw = [ref_outputs[k][i : i + 1] for k in range(len(ref_outputs))]
            ref_dets = self._post_proc(ref_raw)
            act_dets = self._post_proc(act_raw)

            ref_labels = sorted(d.label for d in ref_dets)
            act_labels = sorted(d.label for d in act_dets)
            if ref_labels != act_labels:
                return False, f"decoded mismatch at image {i}"

        return True, ""
