"""Unit tests for model info / source / status data classes and the registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.hardware import ComputeUnit
from moment_to_action.hardware._types import DataType, ModelType
from moment_to_action.models import (
    DownloadSource,
    HuggingFaceSource,
    ModelID,
    ModelInfo,
    Variant,
    VariantStatus,
    VendoredSource,
    YOLOModel,
)
from moment_to_action.models._model_info import ModelStatus
from moment_to_action.models._registry import DEFAULT_KEY, MODEL_REGISTRY

# ---------------------------------------------------------------------------
# ModelID
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelID:
    """Tests for ModelID enum."""

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (ModelID.YOLO_V8, "yolo_v8"),
            (ModelID.MOBILECLIP_S2, "mobileclip_s2"),
            (ModelID.SMOLVLM2_2_2B, "smolvlm2_2_2b"),
        ],
    )
    def test_members_have_expected_values(self, member: ModelID, value: str) -> None:
        """Each ModelID has the expected snake_case value."""
        assert member.value == value


# ---------------------------------------------------------------------------
# ModelType
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelType:
    """Tests for ModelType enum."""

    def test_has_onnx_and_dlc(self) -> None:
        """ModelType has at least ONNX and DLC members."""
        assert ModelType.ONNX is not None
        assert ModelType.DLC is not None

    def test_has_llama_cpp(self) -> None:
        """ModelType has a LLAMA_CPP member."""
        assert ModelType.LLAMA_CPP is not None


# ---------------------------------------------------------------------------
# VendoredSource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVendoredSource:
    """Tests for VendoredSource."""

    def test_requires_path(self) -> None:
        """``path`` is required."""
        with pytest.raises(TypeError):
            VendoredSource()  # type: ignore[call-arg]

    def test_stores_path(self) -> None:
        """Constructor stores ``path``."""
        s = VendoredSource(path=Path("yolo/model.onnx"))
        assert s.path == Path("yolo/model.onnx")

    def test_is_frozen(self) -> None:
        """VendoredSource is frozen."""
        s = VendoredSource(path=Path("a"))
        with pytest.raises(AttributeError):
            s.path = Path("b")  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equal-fields VendoredSources compare equal."""
        a = VendoredSource(path=Path("a"))
        b = VendoredSource(path=Path("a"))
        c = VendoredSource(path=Path("b"))
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# DownloadSource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownloadSource:
    """Tests for DownloadSource."""

    def test_requires_all_fields(self) -> None:
        """``url`` and ``filename`` are required."""
        with pytest.raises(TypeError):
            DownloadSource()  # type: ignore[call-arg]

    def test_stores_fields(self) -> None:
        """Constructor stores all fields."""
        s = DownloadSource(
            url="https://example.com/m.onnx",
            filename="m.onnx",
        )
        assert s.url == "https://example.com/m.onnx"
        assert s.filename == "m.onnx"

    def test_is_frozen(self) -> None:
        """DownloadSource is frozen."""
        s = DownloadSource(url="u", filename="f")
        with pytest.raises(AttributeError):
            s.url = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HuggingFaceSource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceSource:
    """Tests for HuggingFaceSource."""

    def test_requires_all_fields(self) -> None:
        """``hf_repo_id``, ``files``, and ``revision`` are all required."""
        with pytest.raises(TypeError):
            HuggingFaceSource()  # type: ignore[call-arg]

    def test_stores_fields(self) -> None:
        """Constructor stores all fields; hf_subdir defaults to None."""
        s = HuggingFaceSource(
            hf_repo_id="org/repo",
            files=["a", "b"],
            revision="abc123",
        )
        assert s.hf_repo_id == "org/repo"
        assert s.files == ["a", "b"]
        assert s.revision == "abc123"
        assert s.hf_subdir is None

    def test_hf_subdir_stored(self) -> None:
        """hf_subdir is stored when provided."""
        s = HuggingFaceSource(
            hf_repo_id="org/repo",
            files=["model.bin"],
            revision="abc123",
            hf_subdir="mydir",
        )
        assert s.hf_subdir == "mydir"


# ---------------------------------------------------------------------------
# Variant
# ---------------------------------------------------------------------------


def _make_vendored_variant(units: list[ComputeUnit] | None = None) -> Variant:
    """Build a Variant with a VendoredSource for testing."""
    if units is None:
        units = [ComputeUnit.CPU]
    src = VendoredSource(path=Path("model.onnx"))
    return Variant(
        source=src,
        backends={u: {"model": "model.onnx"} for u in units},
        model_type=ModelType.ONNX,
        data_type=DataType.FP32,
    )


@pytest.mark.unit
class TestVariant:
    """Tests for the Variant frozen attrs class."""

    def test_stores_source_and_backends(self) -> None:
        """Source and backends are stored."""
        v = _make_vendored_variant([ComputeUnit.CPU, ComputeUnit.GPU])
        assert isinstance(v.source, VendoredSource)
        assert ComputeUnit.CPU in v.backends
        assert ComputeUnit.GPU in v.backends

    def test_stores_model_type_and_data_type(self) -> None:
        """model_type and data_type are stored."""
        v = _make_vendored_variant()
        assert v.model_type is ModelType.ONNX
        assert v.data_type is DataType.FP32

    def test_default_input_layout_is_none(self) -> None:
        """input_layout defaults to None (not applicable for non-image models)."""
        v = _make_vendored_variant()
        assert v.input_layout is None

    def test_custom_input_layout(self) -> None:
        """input_layout can be overridden."""
        src = VendoredSource(path=Path("x"))
        v = Variant(
            source=src,
            backends={ComputeUnit.NPU: {"model": "x"}},
            model_type=ModelType.DLC,
            data_type=DataType.W8A8,
            input_layout="NHWC",
        )
        assert v.input_layout == "NHWC"

    def test_is_frozen(self) -> None:
        """Variant is frozen (attrs.frozen)."""
        v = _make_vendored_variant()
        with pytest.raises(AttributeError):
            v.input_layout = "NHWC"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two Variants with identical fields compare equal."""
        a = _make_vendored_variant()
        b = _make_vendored_variant()
        assert a == b


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelInfo:
    """Tests for ModelInfo."""

    def test_requires_fields(self) -> None:
        """`id`, `variants`, and `model_class` are all required."""
        with pytest.raises(TypeError):
            ModelInfo(id=ModelID.YOLO_V8, variants={})  # type: ignore[call-arg]

    def test_stores_variant_map(self) -> None:
        """`variants` maps str keys to Variant objects."""
        v = _make_vendored_variant()
        info = ModelInfo(id=ModelID.YOLO_V8, variants={DEFAULT_KEY: v}, model_class=YOLOModel)
        assert info.id is ModelID.YOLO_V8
        assert info.variants == {DEFAULT_KEY: v}

    def test_stores_model_class(self) -> None:
        """`model_class` is stored verbatim."""
        v = _make_vendored_variant()
        info = ModelInfo(id=ModelID.YOLO_V8, variants={DEFAULT_KEY: v}, model_class=YOLOModel)
        assert info.model_class is YOLOModel


# ---------------------------------------------------------------------------
# VariantStatus
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVariantStatus:
    """Tests for VariantStatus."""

    def test_stores_fields(self) -> None:
        """Constructor stores all fields."""
        v = VariantStatus(
            model_id=ModelID.YOLO_V8,
            variant=DEFAULT_KEY,
            available=True,
            path=Path("/x"),
            size_bytes=10,
        )
        assert v.model_id is ModelID.YOLO_V8
        assert v.variant == DEFAULT_KEY
        assert v.available is True
        assert v.path == Path("/x")
        assert v.size_bytes == 10


# ---------------------------------------------------------------------------
# ModelStatus
# ---------------------------------------------------------------------------


def _make_variant(
    model_id: ModelID,
    variant: str,
    *,
    available: bool,
    size: int | None,
) -> VariantStatus:
    """Build a VariantStatus for testing."""
    return VariantStatus(
        model_id=model_id,
        variant=variant,
        available=available,
        path=Path("/x") if available else None,
        size_bytes=size,
    )


@pytest.mark.unit
class TestModelStatus:
    """Tests for ModelStatus."""

    def _info(self) -> ModelInfo:
        v = Variant(
            source=VendoredSource(path=Path("yolo/model.onnx")),
            backends={ComputeUnit.CPU: {"model": "model.onnx"}},
            model_type=ModelType.ONNX,
            data_type=DataType.FP32,
        )
        return ModelInfo(id=ModelID.YOLO_V8, model_class=YOLOModel, variants={DEFAULT_KEY: v})

    def test_available_true_when_any_variant_available(self) -> None:
        """`available` is True if at least one variant is available."""
        status = ModelStatus(
            info=self._info(),
            variants=[
                _make_variant(ModelID.YOLO_V8, DEFAULT_KEY, available=False, size=None),
                _make_variant(ModelID.YOLO_V8, "alt", available=True, size=10),
            ],
            path=Path("/cache/yolo"),
        )
        assert status.available is True

    def test_available_false_when_no_variant_available(self) -> None:
        """`available` is False when no variants are available."""
        status = ModelStatus(
            info=self._info(),
            variants=[
                _make_variant(ModelID.YOLO_V8, DEFAULT_KEY, available=False, size=None),
            ],
            path=None,
        )
        assert status.available is False

    def test_size_bytes_sums_available_variants(self) -> None:
        """`size_bytes` sums variant sizes (treating None as 0)."""
        status = ModelStatus(
            info=self._info(),
            variants=[
                _make_variant(ModelID.YOLO_V8, "a", available=True, size=10),
                _make_variant(ModelID.YOLO_V8, "b", available=True, size=25),
                _make_variant(ModelID.YOLO_V8, "c", available=False, size=None),
            ],
            path=Path("/cache/yolo"),
        )
        assert status.size_bytes == 35

    def test_available_variants_returns_only_available(self) -> None:
        """`available_variants` returns only those with `available=True`."""
        a = _make_variant(ModelID.YOLO_V8, "a", available=True, size=10)
        b = _make_variant(ModelID.YOLO_V8, "b", available=False, size=None)
        status = ModelStatus(info=self._info(), variants=[a, b], path=None)
        assert status.available_variants == [a]


# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_yolo_v8_registered(self) -> None:
        """YOLO_V8 is in the registry with an UltralyticsSource default variant."""
        from moment_to_action.models._sources._ultralytics import UltralyticsSource

        assert ModelID.YOLO_V8 in MODEL_REGISTRY
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert info.id is ModelID.YOLO_V8
        default = info.variants[DEFAULT_KEY]
        assert isinstance(default, Variant)
        assert isinstance(default.source, UltralyticsSource)
        assert default.source.name == "yolov8n"
        assert default.model_type is ModelType.ONNX

    def test_yolo_v8_has_model_class(self) -> None:
        """YOLO_V8 registry entry has YOLOModel as its model_class."""
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert info.model_class is YOLOModel

    def test_yolo_v8_has_qcs6490_variant(self) -> None:
        """YOLO_V8 registry has a qcs6490 DLC variant with NPU backend."""
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert "qcs6490" in info.variants
        qcs = info.variants["qcs6490"]
        assert isinstance(qcs, Variant)
        assert isinstance(qcs.source, HuggingFaceSource)
        assert qcs.model_type is ModelType.DLC
        assert ComputeUnit.NPU in qcs.backends
