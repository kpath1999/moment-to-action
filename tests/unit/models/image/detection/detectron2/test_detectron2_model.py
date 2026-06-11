"""Unit tests for Detectron2Model (two-stage Faster R-CNN)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from moment_to_action.models._formats import ModelFormat
from moment_to_action.models.image.detection.detectron2._model import Detectron2Model


@pytest.fixture
def onnx_model() -> Detectron2Model:
    """Return an unloaded Detectron2Model in ONNX format (NCHW)."""
    return Detectron2Model("default", Path("/fake/d2"), ModelFormat.ONNX)


@pytest.fixture
def dlc_model() -> Detectron2Model:
    """Return an unloaded Detectron2Model in DLC format (qcs6490 → NHWC)."""
    return Detectron2Model("qcs6490", Path("/fake/d2_qcs"), ModelFormat.DLC)


@pytest.fixture
def dlc_model_nchw() -> Detectron2Model:
    """Return an unloaded Detectron2Model in DLC format with NCHW layout."""
    return Detectron2Model("other", Path("/fake/d2_other"), ModelFormat.DLC)


@pytest.fixture
def mock_backend() -> MagicMock:
    """Return a mock ComputeBackend with two-graph handles."""
    backend = MagicMock()
    backend.load_model.return_value = MagicMock()
    backend.load_model_dlc.return_value = MagicMock()
    return backend


def _roi_outputs(
    boxes: list[list[float]], scores: list[float], classes: list[int]
) -> list[np.ndarray]:
    """Build a synthetic ROI-head [boxes, scores, classes] output triple."""
    return [
        np.array([boxes], dtype=np.float32),
        np.array([scores], dtype=np.float32),
        np.array([classes], dtype=np.int64),
    ]


@pytest.mark.unit
class TestPrepare:
    """Tests for Detectron2Model.prepare()."""

    def test_nchw_shape(self, onnx_model: Detectron2Model, sample_image_array: np.ndarray) -> None:
        """ONNX prepare returns NCHW (1, 3, 800, 800)."""
        assert onnx_model.prepare(sample_image_array).shape == (1, 3, 800, 800)

    def test_nhwc_shape(self, dlc_model: Detectron2Model, sample_image_array: np.ndarray) -> None:
        """qcs6490 DLC prepare returns NHWC (1, 800, 800, 3)."""
        assert dlc_model.prepare(sample_image_array).shape == (1, 800, 800, 3)

    def test_dtype_and_range(
        self, onnx_model: Detectron2Model, sample_image_array: np.ndarray
    ) -> None:
        """Prepare normalizes to float32 in [0, 1]."""
        out = onnx_model.prepare(sample_image_array)
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_stores_original_size(
        self, onnx_model: Detectron2Model, sample_image_array: np.ndarray
    ) -> None:
        """Prepare records the original frame size."""
        onnx_model.prepare(sample_image_array)
        h, w = sample_image_array.shape[:2]
        assert onnx_model._last_original_size == (h, w)


@pytest.mark.unit
class TestProperties:
    """Tests for Detectron2Model properties + layout derivation."""

    def test_default_confidence(self, onnx_model: Detectron2Model) -> None:
        """Default confidence_threshold is 0.5."""
        assert onnx_model.confidence_threshold == 0.5

    def test_custom_confidence(self) -> None:
        """Custom confidence_threshold is stored."""
        m = Detectron2Model("default", Path("/f"), ModelFormat.ONNX, confidence_threshold=0.2)
        assert m.confidence_threshold == 0.2

    def test_layout_onnx_nchw(self, onnx_model: Detectron2Model) -> None:
        """ONNX uses NCHW."""
        assert onnx_model.input_layout == "NCHW"

    def test_layout_qcs6490_nhwc(self, dlc_model: Detectron2Model) -> None:
        """qcs6490 DLC uses NHWC."""
        assert dlc_model.input_layout == "NHWC"

    def test_layout_other_dlc_nchw(self, dlc_model_nchw: Detectron2Model) -> None:
        """Non-qcs6490 DLC uses NCHW."""
        assert dlc_model_nchw.input_layout == "NCHW"

    @pytest.mark.parametrize("variant", ["qcs6490_w8a16", "qcs6490_w8a8"])
    def test_layout_qcs6490_precision_variants_nhwc(self, variant: str) -> None:
        """Both qcs6490 precision-variant keys derive NHWC layout."""
        m = Detectron2Model(variant, Path("/fake"), ModelFormat.DLC)
        assert m.input_layout == "NHWC"


@pytest.mark.unit
class TestLoadUnload:
    """Tests for load()/unload() of both component graphs."""

    def test_load_onnx_loads_both(
        self, onnx_model: Detectron2Model, mock_backend: MagicMock
    ) -> None:
        """ONNX load() loads both component graphs."""
        onnx_model.load(mock_backend)
        assert mock_backend.load_model.call_count == 2

    def test_load_dlc_resolves_both_stems(
        self, dlc_model: Detectron2Model, mock_backend: MagicMock
    ) -> None:
        """DLC load() resolves both component stems and loads both DLCs."""
        import moment_to_action.models.image.detection.detectron2._model as m

        mock_resolve = MagicMock(return_value=Path("/fake/x.dlc"))
        orig = m.resolve_backend_artifact
        m.resolve_backend_artifact = mock_resolve
        try:
            dlc_model.load(mock_backend)
        finally:
            m.resolve_backend_artifact = orig
        stems = {c.kwargs["stem"] for c in mock_resolve.call_args_list}
        assert stems == {"model.proposal_generator", "model.roi_head"}
        assert mock_backend.load_model_dlc.call_count == 2

    def test_load_twice_raises(self, onnx_model: Detectron2Model, mock_backend: MagicMock) -> None:
        """Loading twice raises RuntimeError."""
        onnx_model.load(mock_backend)
        with pytest.raises(RuntimeError, match="already loaded"):
            onnx_model.load(mock_backend)

    def test_unload_onnx_unloads_both(
        self, onnx_model: Detectron2Model, mock_backend: MagicMock
    ) -> None:
        """ONNX unload() releases both graphs and clears state."""
        onnx_model.load(mock_backend)
        onnx_model.unload()
        assert mock_backend.unload_model.call_count == 2
        assert onnx_model._backend is None
        assert onnx_model._handle_pg is None
        assert onnx_model._handle_roi is None

    def test_unload_dlc_unloads_both(
        self, dlc_model: Detectron2Model, mock_backend: MagicMock
    ) -> None:
        """DLC unload() calls unload_dlc for both graphs."""
        import moment_to_action.models.image.detection.detectron2._model as m

        orig = m.resolve_backend_artifact
        m.resolve_backend_artifact = MagicMock(return_value=Path("/fake/x.dlc"))
        try:
            dlc_model.load(mock_backend)
        finally:
            m.resolve_backend_artifact = orig
        dlc_model.unload()
        assert mock_backend.unload_dlc.call_count == 2

    def test_unload_without_load_noop(self, onnx_model: Detectron2Model) -> None:
        """unload() before load() does nothing."""
        onnx_model.unload()


@pytest.mark.unit
class TestRun:
    """Tests for the two-stage run()."""

    def test_run_without_load_raises(
        self, onnx_model: Detectron2Model, sample_image_array: np.ndarray
    ) -> None:
        """run() before load() raises."""
        prepared = onnx_model.prepare(sample_image_array)
        with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
            onnx_model.run(prepared)

    def test_run_onnx_two_stage(
        self, onnx_model: Detectron2Model, mock_backend: MagicMock, sample_image_array: np.ndarray
    ) -> None:
        """ONNX run() chains proposal generator → filter → ROI head (list outputs)."""
        feature = np.zeros((1, 4, 5, 5), dtype=np.float32)
        proposals = np.array([[[0, 0, 100, 100], [10, 10, 50, 50]]], dtype=np.float32)
        score = np.array([[0.9, 0.8]], dtype=np.float32)
        roi = _roi_outputs([[0, 0, 100, 100]], [0.9], [1])
        mock_backend.run.side_effect = [[feature, proposals, score], roi]
        onnx_model.load(mock_backend)
        prepared = onnx_model.prepare(sample_image_array)
        result = onnx_model.run(prepared)
        assert len(result) == 3
        # ROI head was called with a name→tensor dict (multi-input)
        roi_call = mock_backend.run.call_args_list[1]
        roi_inputs = roi_call.args[1]
        assert set(roi_inputs) == {"features", "proposals_boxes"}
        assert roi_inputs["proposals_boxes"].shape == (1, 200, 4)

    def test_run_dlc_two_stage(
        self, dlc_model: Detectron2Model, mock_backend: MagicMock, sample_image_array: np.ndarray
    ) -> None:
        """DLC run() chains both infer_dlc calls and remaps the dict outputs."""
        feature = np.zeros((1, 4, 5, 5), dtype=np.float32)
        proposals = np.array([[[0, 0, 100, 100]]], dtype=np.float32)
        score = np.array([[0.9]], dtype=np.float32)
        boxes = np.array([[[1, 2, 3, 4]]], dtype=np.float32)
        scores = np.array([[0.7]], dtype=np.float32)
        classes = np.array([[2]], dtype=np.int64)
        mock_backend.infer_dlc.side_effect = [
            {"feature": feature, "proposals": proposals, "score": score},
            {"boxes": boxes, "scores": scores, "classes": classes},
        ]
        import moment_to_action.models.image.detection.detectron2._model as m

        orig = m.resolve_backend_artifact
        m.resolve_backend_artifact = MagicMock(return_value=Path("/fake/x.dlc"))
        try:
            dlc_model.load(mock_backend)
        finally:
            m.resolve_backend_artifact = orig
        prepared = dlc_model.prepare(sample_image_array)
        result = dlc_model.run(prepared)
        np.testing.assert_array_equal(result[0], boxes)
        np.testing.assert_array_equal(result[1], scores)
        np.testing.assert_array_equal(result[2], classes)
        roi_inputs = mock_backend.infer_dlc.call_args_list[1].args[1]
        assert set(roi_inputs) == {"features", "proposals_boxes"}


@pytest.mark.unit
class TestFilterProposals:
    """Tests for the CPU proposal filter."""

    def test_pads_to_fixed_length(self, onnx_model: Detectron2Model) -> None:
        """Output is always [1, 200, 4] regardless of input count."""
        proposals = np.array([[[0, 0, 10, 10], [20, 20, 40, 40]]], dtype=np.float32)
        score = np.array([[0.5, 0.9]], dtype=np.float32)
        out = onnx_model._filter_proposals(proposals, score)
        assert out.shape == (1, 200, 4)

    def test_clamps_to_image(self, onnx_model: Detectron2Model) -> None:
        """Proposal coordinates are clamped to [0, 800]."""
        proposals = np.array([[[-50, -50, 900, 900]]], dtype=np.float32)
        score = np.array([[0.9]], dtype=np.float32)
        out = onnx_model._filter_proposals(proposals, score)[0, 0]
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] == 800.0
        assert out[3] == 800.0

    def test_drops_empty_boxes(self, onnx_model: Detectron2Model) -> None:
        """Zero-area proposals are removed before NMS."""
        # second box has zero width → dropped, leaving exactly one real proposal
        proposals = np.array([[[0, 0, 100, 100], [50, 0, 50, 100]]], dtype=np.float32)
        score = np.array([[0.6, 0.9]], dtype=np.float32)
        out = onnx_model._filter_proposals(proposals, score)
        nonzero_rows = int(np.count_nonzero(out[0].any(axis=1)))
        assert nonzero_rows == 1

    def test_keeps_highest_objectness_first(self, onnx_model: Detectron2Model) -> None:
        """The top-scoring proposal survives and lands first."""
        proposals = np.array([[[0, 0, 10, 10], [500, 500, 600, 600]]], dtype=np.float32)
        score = np.array([[0.1, 0.99]], dtype=np.float32)
        out = onnx_model._filter_proposals(proposals, score)[0]
        np.testing.assert_allclose(out[0], [500, 500, 600, 600])


@pytest.mark.unit
class TestPostProc:
    """Tests for post_proc()/_decode()."""

    def test_empty_outputs(self, onnx_model: Detectron2Model) -> None:
        """No surviving boxes returns an empty list."""
        assert onnx_model._decode(_roi_outputs([], [], []), original_size=None) == []

    def test_fewer_than_three_outputs(self, onnx_model: Detectron2Model) -> None:
        """Fewer than 3 output tensors returns empty."""
        assert onnx_model._decode([np.zeros((1, 2, 4)), np.zeros((1, 2))], original_size=None) == []

    def test_filters_low_confidence(self, onnx_model: Detectron2Model) -> None:
        """Detections below confidence_threshold are dropped."""
        out = onnx_model._decode(_roi_outputs([[0, 0, 50, 50]], [0.1], [0]), original_size=None)
        assert out == []

    def test_label_mapping(self, onnx_model: Detectron2Model) -> None:
        """Class index maps to the COCO label."""
        out = onnx_model._decode(_roi_outputs([[0, 0, 50, 50]], [0.9], [2]), original_size=None)
        assert out[0].label == "car"

    def test_unknown_class_uses_string(self, onnx_model: Detectron2Model) -> None:
        """Out-of-range class index renders as its string."""
        out = onnx_model._decode(_roi_outputs([[0, 0, 50, 50]], [0.9], [999]), original_size=None)
        assert out[0].label == "999"

    def test_scales_to_original(self, onnx_model: Detectron2Model) -> None:
        """Boxes scale from 800x800 to the original size."""
        out = onnx_model._decode(
            _roi_outputs([[0, 0, 800, 800]], [0.9], [0]), original_size=(1600, 1600)
        )
        assert out[0].bbox.x2 == pytest.approx(1600.0)
        assert out[0].bbox.y2 == pytest.approx(1600.0)

    def test_per_class_nms_keeps_distinct_classes(self, onnx_model: Detectron2Model) -> None:
        """Overlapping boxes of different classes are both kept (per-class NMS)."""
        out = onnx_model._decode(
            _roi_outputs([[0, 0, 100, 100], [1, 1, 99, 99]], [0.9, 0.8], [0, 1]),
            original_size=None,
        )
        assert len(out) == 2

    def test_per_class_nms_suppresses_same_class(self, onnx_model: Detectron2Model) -> None:
        """Overlapping boxes of the same class collapse to one."""
        out = onnx_model._decode(
            _roi_outputs([[0, 0, 100, 100], [1, 1, 99, 99]], [0.9, 0.8], [0, 0]),
            original_size=None,
        )
        assert len(out) == 1

    def test_post_proc_uses_recorded_size(
        self, onnx_model: Detectron2Model, sample_image_array: np.ndarray
    ) -> None:
        """post_proc scales using the size recorded by prepare()."""
        onnx_model.prepare(sample_image_array)
        h, w = sample_image_array.shape[:2]
        out = onnx_model.post_proc(_roi_outputs([[0, 0, 800, 800]], [0.9], [0]))
        assert out[0].bbox.x2 == pytest.approx(float(w))
        assert out[0].bbox.y2 == pytest.approx(float(h))

    def test_decode_accepts_iterable(self, onnx_model: Detectron2Model) -> None:
        """decode() coerces its raw arg and scales to the given size."""
        raw = _roi_outputs([[0, 0, 800, 800]], [0.9], [0])
        out = onnx_model.decode(raw, original_size=(400, 400))
        assert out[0].bbox.x2 == pytest.approx(400.0)
