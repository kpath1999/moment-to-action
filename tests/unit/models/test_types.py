"""Unit tests for model info / source / status data classes and the registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.models import (
    DownloadSource,
    HuggingFaceSource,
    ModelFormat,
    ModelID,
    ModelInfo,
    VariantStatus,
    VendoredSource,
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
# ModelFormat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelFormat:
    """Tests for ModelFormat enum."""

    def test_has_onnx_and_dlc(self) -> None:
        """ModelFormat has at least ONNX and DLC members."""
        assert ModelFormat.ONNX is not None
        assert ModelFormat.DLC is not None


# ---------------------------------------------------------------------------
# VendoredSource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVendoredSource:
    """Tests for VendoredSource."""

    def test_requires_format_and_path(self) -> None:
        """Both `format` and `path` are required."""
        with pytest.raises(TypeError):
            VendoredSource()  # type: ignore[call-arg]

    def test_stores_fields(self) -> None:
        """Constructor stores `format` and `path`."""
        s = VendoredSource(format=ModelFormat.ONNX, path=Path("yolo/model.onnx"))
        assert s.format is ModelFormat.ONNX
        assert s.path == Path("yolo/model.onnx")

    def test_is_frozen(self) -> None:
        """VendoredSource is frozen."""
        s = VendoredSource(format=ModelFormat.ONNX, path=Path("a"))
        with pytest.raises(AttributeError):
            s.path = Path("b")  # type: ignore[misc]

    def test_equality(self) -> None:
        """Equal-fields VendoredSources compare equal."""
        a = VendoredSource(format=ModelFormat.ONNX, path=Path("a"))
        b = VendoredSource(format=ModelFormat.ONNX, path=Path("a"))
        c = VendoredSource(format=ModelFormat.DLC, path=Path("a"))
        assert a == b
        assert a != c


# ---------------------------------------------------------------------------
# DownloadSource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownloadSource:
    """Tests for DownloadSource."""

    def test_requires_all_fields(self) -> None:
        """`format`, `url`, and `filename` are all required."""
        with pytest.raises(TypeError):
            DownloadSource()  # type: ignore[call-arg]

    def test_stores_fields(self) -> None:
        """Constructor stores all three fields."""
        s = DownloadSource(
            format=ModelFormat.ONNX,
            url="https://example.com/m.onnx",
            filename="m.onnx",
        )
        assert s.format is ModelFormat.ONNX
        assert s.url == "https://example.com/m.onnx"
        assert s.filename == "m.onnx"

    def test_is_frozen(self) -> None:
        """DownloadSource is frozen."""
        s = DownloadSource(format=ModelFormat.ONNX, url="u", filename="f")
        with pytest.raises(AttributeError):
            s.url = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HuggingFaceSource
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHuggingFaceSource:
    """Tests for HuggingFaceSource."""

    def test_requires_all_fields(self) -> None:
        """`format`, `hf_repo_id`, `files`, and `revision` are all required."""
        with pytest.raises(TypeError):
            HuggingFaceSource()  # type: ignore[call-arg]

    def test_stores_fields(self) -> None:
        """Constructor stores all fields."""
        s = HuggingFaceSource(
            format=ModelFormat.ONNX,
            hf_repo_id="org/repo",
            files=["a", "b"],
            revision="abc123",
        )
        assert s.hf_repo_id == "org/repo"
        assert s.files == ["a", "b"]
        assert s.revision == "abc123"


# ---------------------------------------------------------------------------
# ModelInfo
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelInfo:
    """Tests for ModelInfo."""

    def test_requires_fields(self) -> None:
        """`id` and `variants` are required."""
        with pytest.raises(TypeError):
            ModelInfo(id=ModelID.YOLO_V8)  # type: ignore[call-arg]

    def test_stores_variant_map(self) -> None:
        """`variants` is stored verbatim."""
        v = VendoredSource(format=ModelFormat.ONNX, path=Path("a"))
        info = ModelInfo(id=ModelID.YOLO_V8, variants={DEFAULT_KEY: v})
        assert info.id is ModelID.YOLO_V8
        assert info.variants == {DEFAULT_KEY: v}


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
        return ModelInfo(
            id=ModelID.YOLO_V8,
            variants={
                DEFAULT_KEY: VendoredSource(format=ModelFormat.ONNX, path=Path("yolo/model.onnx"))
            },
        )

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
        """YOLO_V8 is in the registry with a VendoredSource default variant."""
        assert ModelID.YOLO_V8 in MODEL_REGISTRY
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert info.id is ModelID.YOLO_V8
        default = info.variants[DEFAULT_KEY]
        assert isinstance(default, VendoredSource)
        assert default.path == Path("yolo/model.onnx")
        assert default.format is ModelFormat.ONNX
