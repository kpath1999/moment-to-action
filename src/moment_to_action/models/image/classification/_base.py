"""Abstract base class for image classification models."""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from moment_to_action.models.image._base import ImageModel
from moment_to_action.models.image.classification._types import Classification


class ImageClassificationModel(ImageModel[list[np.ndarray], Classification]):
    """Abstract base for models that classify images into discrete categories.

    Fixes ``_RawOutputT=list[np.ndarray]`` and ``_ResultT=Classification``, so:

    - :meth:`run` returns ``list[np.ndarray]``
    - :meth:`post_proc` takes ``list[np.ndarray]`` and returns ``list[Classification]``

    Provides a concrete :meth:`verify_outputs` implementation that compares
    top-1 predicted labels against reference data.
    """

    @abstractmethod
    def _post_proc(self, raw: list[np.ndarray]) -> list[Classification]:
        """Decode raw model output into a list of classifications.

        Args:
            raw: Output returned by :meth:`~moment_to_action.models.image.ImageModel._run`.

        Returns:
            List of :class:`~moment_to_action.models.image.classification.Classification`
            objects ordered by descending confidence.
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

        For CPU/GPU: checks raw element-wise diff ≤ ``tol`` AND top-1 decoded
        label matches.  For NPU: skips raw diff (INT8 quantisation noise
        dominates) and checks top-1 decoded label only.

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
            ref_top1 = self._post_proc(ref_raw)[:1]
            act_top1 = self._post_proc(act_raw)[:1]

            ref_label = ref_top1[0].label if ref_top1 else ""
            act_label = act_top1[0].label if act_top1 else ""
            if ref_label != act_label:
                return (
                    False,
                    f"top-1 mismatch at image {i}: got {act_label!r}, expected {ref_label!r}",
                )

        return True, ""
