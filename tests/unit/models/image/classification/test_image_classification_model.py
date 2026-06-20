"""Unit tests for ImageClassificationModel ABC."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.models.image.classification._base import ImageClassificationModel
from moment_to_action.models.image.classification._types import Classification


class _ConcreteClassificationModel(ImageClassificationModel):
    """Minimal concrete classification model for testing verify_outputs."""

    def __init__(self, classifications: list[Classification] | None = None) -> None:
        """Initialize with canned classifications."""
        super().__init__("default", Path("/x"), backends={})
        self._classifications = classifications or []
        self._run_output: list[np.ndarray] = []

    def load(self, platform: object, unit: object) -> None:  # type: ignore[override]
        """Load."""
        self._platform = platform  # type: ignore[assignment]

    def unload(self) -> None:
        """Unload."""
        self._platform = None

    def prepare(self, inputs: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Prepare."""
        return inputs

    def run(self, prepared: np.ndarray) -> list[np.ndarray]:
        """Return canned run output."""
        return self._run_output

    def post_proc(self, raw: list[np.ndarray]) -> list[Classification]:
        """Return canned classifications."""
        return self._classifications


@pytest.mark.unit
class TestImageClassificationModel:
    """Tests for ImageClassificationModel abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """ImageClassificationModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ImageClassificationModel("default", Path("/x"))  # type: ignore[abstract, call-arg]

    def test_abstract_methods_enforced(self) -> None:
        """Subclasses missing post_proc cannot be instantiated."""

        class _Incomplete(ImageClassificationModel):
            def load(self, backend: object, unit: object = None) -> None:  # type: ignore[override]
                """Load."""

            def unload(self) -> None:
                """Unload."""

            def prepare(self, inputs: np.ndarray) -> np.ndarray:  # type: ignore[override]
                """Prepare."""
                return inputs

            def run(self, prepared: np.ndarray) -> list[np.ndarray]:
                """Run."""
                return []

            # Missing post_proc

        with pytest.raises(TypeError):
            _Incomplete("v", Path("/x"))  # type: ignore[abstract, call-arg]


def _make_ref_outputs(n: int = 2) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build minimal reference inputs and outputs."""
    inputs = np.zeros((n, 3, 4, 4), dtype=np.float32)
    ref0 = np.zeros((n, 10), dtype=np.float32)
    return inputs, [ref0]


@pytest.mark.unit
class TestVerifyOutputs:
    """Tests for ImageClassificationModel.verify_outputs."""

    def test_passes_when_outputs_match(self) -> None:
        """Returns (True, '') when raw diff <= tol and top-1 labels match."""
        inputs, ref_outputs = _make_ref_outputs()
        model = _ConcreteClassificationModel(classifications=[Classification("cat", 0.9, 281)])
        model._run_output = [np.zeros((1, 10), dtype=np.float32)]
        model._platform = MagicMock()
        passed, reason = model.verify_outputs(inputs, ref_outputs, tol=0.01, is_npu=False)
        assert passed is True
        assert reason == ""

    def test_fails_raw_diff_exceeds_tol(self) -> None:
        """Returns (False, reason) when raw diff > tol."""
        inputs, ref_outputs = _make_ref_outputs()
        model = _ConcreteClassificationModel(classifications=[Classification("cat", 0.9, 281)])
        model._run_output = [np.ones((1, 10), dtype=np.float32) * 999.0]
        model._platform = MagicMock()
        passed, reason = model.verify_outputs(inputs, ref_outputs, tol=0.01, is_npu=False)
        assert passed is False
        assert "max_err" in reason

    def test_npu_skips_raw_diff(self) -> None:
        """NPU mode skips raw diff; passes if top-1 labels match."""
        inputs, ref_outputs = _make_ref_outputs()
        model = _ConcreteClassificationModel(classifications=[Classification("cat", 0.9, 281)])
        model._run_output = [np.ones((1, 10), dtype=np.float32) * 999.0]
        model._platform = MagicMock()
        passed, _ = model.verify_outputs(inputs, ref_outputs, tol=0.01, is_npu=True)
        assert passed is True

    def test_fails_top1_mismatch(self) -> None:
        """Returns (False, reason) when top-1 label differs."""
        inputs, ref_outputs = _make_ref_outputs(n=1)
        call_count = 0

        class _Alternating(_ConcreteClassificationModel):
            def post_proc(self, raw: list[np.ndarray]) -> list[Classification]:
                """Alternate between two label sets."""
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 1:
                    return [Classification("cat", 0.9, 281)]
                return [Classification("dog", 0.9, 207)]

        model = _Alternating()
        model._run_output = [np.zeros((1, 10), dtype=np.float32)]
        model._platform = MagicMock()
        passed, reason = model.verify_outputs(inputs, ref_outputs, tol=0.01, is_npu=False)
        assert passed is False
        assert "top-1 mismatch" in reason

    def test_empty_inputs_passes(self) -> None:
        """Zero-sample input array trivially passes."""
        inputs = np.zeros((0, 3, 4, 4), dtype=np.float32)
        ref_outputs = [np.zeros((0, 10), dtype=np.float32)]
        model = _ConcreteClassificationModel()
        model._platform = MagicMock()
        passed, reason = model.verify_outputs(inputs, ref_outputs, tol=0.01, is_npu=False)
        assert passed is True
        assert reason == ""

    def test_empty_post_proc_mismatch(self) -> None:
        """Empty top-1 on one side while other has a prediction fails."""
        inputs, ref_outputs = _make_ref_outputs(n=1)
        call_count = 0

        class _Empty(_ConcreteClassificationModel):
            def post_proc(self, raw: list[np.ndarray]) -> list[Classification]:
                """First call empty, second non-empty."""
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return []
                return [Classification("cat", 0.9, 281)]

        model = _Empty()
        model._run_output = [np.zeros((1, 10), dtype=np.float32)]
        model._platform = MagicMock()
        passed, reason = model.verify_outputs(inputs, ref_outputs, tol=0.01, is_npu=False)
        assert passed is False
        assert "top-1 mismatch" in reason
