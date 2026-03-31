"""Unit tests for models._types module."""

from __future__ import annotations

from pathlib import Path

import pytest

from moment_to_action.models._registry import MODEL_REGISTRY
from moment_to_action.models._types import (
    DownloadSource,
    ModelID,
    ModelInfo,
    ModelStatus,
    TransformersSource,
    VendoredSource,
)


@pytest.mark.unit
class TestModelID:
    """Tests for ModelID enum."""

    def test_model_id_has_yolo_v8(self) -> None:
        """Test that ModelID enum has YOLO_V8."""
        assert hasattr(ModelID, "YOLO_V8")
        assert ModelID.YOLO_V8.value == "yolo_v8"

    def test_model_id_has_mobileclip_s2(self) -> None:
        """Test that ModelID enum has MOBILECLIP_S2."""
        assert hasattr(ModelID, "MOBILECLIP_S2")
        assert ModelID.MOBILECLIP_S2.value == "mobileclip_s2"

    def test_model_id_enum_count(self) -> None:
        """Test that ModelID has exactly three members."""
        assert len(list(ModelID)) == 3

    @pytest.mark.parametrize(
        "model_id",
        [ModelID.YOLO_V8, ModelID.MOBILECLIP_S2, ModelID.SMOLVLM2_2_2B],
    )
    def test_model_id_has_value(self, model_id: ModelID) -> None:
        """Test that each ModelID has a value."""
        assert isinstance(model_id.value, str)
        assert len(model_id.value) > 0

    def test_model_id_has_smolvlm2_2_2b(self) -> None:
        """Test that ModelID enum has SMOLVLM2_2_2B."""
        assert hasattr(ModelID, "SMOLVLM2_2_2B")
        assert ModelID.SMOLVLM2_2_2B.value == "smolvlm2_2_2b"


@pytest.mark.unit
class TestVendoredSource:
    """Tests for VendoredSource attrs."""

    def test_vendored_source_subdir_required(self) -> None:
        """Test that VendoredSource requires subdir field."""
        with pytest.raises(TypeError):
            VendoredSource()  # type: ignore[call-arg]

    def test_vendored_source_subdir_stored(self) -> None:
        """Test that VendoredSource stores subdir correctly."""
        source = VendoredSource(subdir="test_subdir")
        assert source.subdir == "test_subdir"

    def test_vendored_source_is_frozen(self) -> None:
        """Test that VendoredSource is frozen (immutable)."""
        source = VendoredSource(subdir="test")
        with pytest.raises(AttributeError):
            source.subdir = "modified"  # type: ignore[misc]

    def test_vendored_source_equality(self) -> None:
        """Test VendoredSource equality comparison."""
        source1 = VendoredSource(subdir="yolo")
        source2 = VendoredSource(subdir="yolo")
        source3 = VendoredSource(subdir="other")
        assert source1 == source2
        assert source1 != source3


@pytest.mark.unit
class TestDownloadSource:
    """Tests for DownloadSource attrs."""

    def test_download_source_hf_repo_id_required(self) -> None:
        """Test that DownloadSource requires hf_repo_id field."""
        with pytest.raises(TypeError):
            DownloadSource(hf_filename="file.tflite")  # type: ignore[call-arg]

    def test_download_source_hf_filename_required(self) -> None:
        """Test that DownloadSource requires hf_filename field."""
        with pytest.raises(TypeError):
            DownloadSource(hf_repo_id="user/repo")  # type: ignore[call-arg]

    def test_download_source_both_fields_required(self) -> None:
        """Test that DownloadSource requires both fields."""
        with pytest.raises(TypeError):
            DownloadSource()  # type: ignore[call-arg]

    def test_download_source_stores_both_fields(self) -> None:
        """Test that DownloadSource stores both fields correctly."""
        source = DownloadSource(
            hf_repo_id="user/repo",
            hf_filename="model.tflite",
        )
        assert source.hf_repo_id == "user/repo"
        assert source.hf_filename == "model.tflite"

    def test_download_source_is_frozen(self) -> None:
        """Test that DownloadSource is frozen (immutable)."""
        source = DownloadSource(
            hf_repo_id="user/repo",
            hf_filename="model.tflite",
        )
        with pytest.raises(AttributeError):
            source.hf_repo_id = "modified"  # type: ignore[misc]

    def test_download_source_equality(self) -> None:
        """Test DownloadSource equality comparison."""
        source1 = DownloadSource(
            hf_repo_id="user/repo",
            hf_filename="model.tflite",
        )
        source2 = DownloadSource(
            hf_repo_id="user/repo",
            hf_filename="model.tflite",
        )
        source3 = DownloadSource(
            hf_repo_id="other/repo",
            hf_filename="model.tflite",
        )
        assert source1 == source2
        assert source1 != source3


@pytest.mark.unit
class TestModelInfo:
    """Tests for ModelInfo attrs."""

    def test_model_info_id_required(self) -> None:
        """Test that ModelInfo requires id field."""
        with pytest.raises(TypeError):
            ModelInfo(  # type: ignore[call-arg]
                filename="model.onnx",
                source=VendoredSource(subdir="yolo"),
            )

    def test_model_info_filename_required(self) -> None:
        """Test that ModelInfo requires filename field."""
        with pytest.raises(TypeError):
            ModelInfo(  # type: ignore[call-arg]
                id=ModelID.YOLO_V8,
                source=VendoredSource(subdir="yolo"),
            )

    def test_model_info_source_required(self) -> None:
        """Test that ModelInfo requires source field."""
        with pytest.raises(TypeError):
            ModelInfo(  # type: ignore[call-arg]
                id=ModelID.YOLO_V8,
                filename="model.onnx",
            )

    def test_model_info_stores_all_fields(self) -> None:
        """Test that ModelInfo stores all fields correctly."""
        source = VendoredSource(subdir="yolo")
        info = ModelInfo(
            id=ModelID.YOLO_V8,
            filename="model.onnx",
            source=source,
        )
        assert info.id == ModelID.YOLO_V8
        assert info.filename == "model.onnx"
        assert info.source is source

    def test_model_info_is_frozen(self) -> None:
        """Test that ModelInfo is frozen (immutable)."""
        info = ModelInfo(
            id=ModelID.YOLO_V8,
            filename="model.onnx",
            source=VendoredSource(subdir="yolo"),
        )
        with pytest.raises(AttributeError):
            info.id = ModelID.MOBILECLIP_S2  # type: ignore[misc]

    def test_model_info_with_vendored_source(self) -> None:
        """Test ModelInfo with VendoredSource."""
        source = VendoredSource(subdir="yolo")
        info = ModelInfo(
            id=ModelID.YOLO_V8,
            filename="model.onnx",
            source=source,
        )
        assert isinstance(info.source, VendoredSource)

    def test_model_info_with_download_source(self) -> None:
        """Test ModelInfo with DownloadSource."""
        source = DownloadSource(
            hf_repo_id="user/repo",
            hf_filename="model.tflite",
        )
        info = ModelInfo(
            id=ModelID.MOBILECLIP_S2,
            filename="model.tflite",
            source=source,
        )
        assert isinstance(info.source, DownloadSource)

    def test_model_info_with_transformers_source(self) -> None:
        """Test ModelInfo with TransformersSource."""
        source = TransformersSource(hf_repo_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        info = ModelInfo(
            id=ModelID.SMOLVLM2_2_2B,
            filename="",
            source=source,
        )
        assert isinstance(info.source, TransformersSource)


@pytest.mark.unit
class TestTransformersSource:
    """Tests for TransformersSource attrs."""

    def test_transformers_source_hf_repo_id_required(self) -> None:
        """Test that TransformersSource requires hf_repo_id field."""
        with pytest.raises(TypeError):
            TransformersSource()  # type: ignore[call-arg]

    def test_transformers_source_stores_hf_repo_id(self) -> None:
        """Test that TransformersSource stores hf_repo_id correctly."""
        source = TransformersSource(hf_repo_id="org/repo")
        assert source.hf_repo_id == "org/repo"

    def test_transformers_source_is_frozen(self) -> None:
        """Test that TransformersSource is frozen (immutable)."""
        source = TransformersSource(hf_repo_id="org/repo")
        with pytest.raises(AttributeError):
            source.hf_repo_id = "modified"  # type: ignore[misc]


@pytest.mark.unit
class TestModelStatus:
    """Tests for ModelStatus attrs."""

    def test_model_status_stores_all_fields(self) -> None:
        """Test that ModelStatus stores all fields correctly."""
        info = ModelInfo(
            id=ModelID.YOLO_V8,
            filename="model.onnx",
            source=VendoredSource(subdir="yolo"),
        )
        path = Path("/path/to/model.onnx")
        status = ModelStatus(
            info=info,
            available=True,
            path=path,
            size_bytes=1000,
        )
        assert status.info is info
        assert status.available is True
        assert status.path == path
        assert status.size_bytes == 1000

    def test_model_status_not_available(self) -> None:
        """Test ModelStatus with unavailable model."""
        info = ModelInfo(
            id=ModelID.MOBILECLIP_S2,
            filename="model.tflite",
            source=DownloadSource(
                hf_repo_id="user/repo",
                hf_filename="model.tflite",
            ),
        )
        status = ModelStatus(
            info=info,
            available=False,
            path=None,
            size_bytes=None,
        )
        assert status.available is False
        assert status.path is None
        assert status.size_bytes is None

    def test_model_status_is_frozen(self) -> None:
        """Test that ModelStatus is frozen (immutable)."""
        info = ModelInfo(
            id=ModelID.YOLO_V8,
            filename="model.onnx",
            source=VendoredSource(subdir="yolo"),
        )
        status = ModelStatus(
            info=info,
            available=True,
            path=Path("/path"),
            size_bytes=1000,
        )
        with pytest.raises(AttributeError):
            status.available = False  # type: ignore[misc]


@pytest.mark.unit
class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_registry_contains_yolo_v8(self) -> None:
        """Test that MODEL_REGISTRY contains YOLO_V8."""
        assert ModelID.YOLO_V8 in MODEL_REGISTRY
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert info.id == ModelID.YOLO_V8

    def test_registry_contains_mobileclip_s2(self) -> None:
        """Test that MODEL_REGISTRY contains MOBILECLIP_S2."""
        assert ModelID.MOBILECLIP_S2 in MODEL_REGISTRY
        info = MODEL_REGISTRY[ModelID.MOBILECLIP_S2]
        assert info.id == ModelID.MOBILECLIP_S2

    def test_registry_contains_smolvlm2_2_2b(self) -> None:
        """Test that MODEL_REGISTRY contains SMOLVLM2_2_2B."""
        assert ModelID.SMOLVLM2_2_2B in MODEL_REGISTRY
        info = MODEL_REGISTRY[ModelID.SMOLVLM2_2_2B]
        assert info.id == ModelID.SMOLVLM2_2_2B

    def test_registry_has_exactly_two_entries(self) -> None:
        """Test that MODEL_REGISTRY has exactly three entries."""
        assert len(MODEL_REGISTRY) == 3

    def test_yolo_v8_is_vendored(self) -> None:
        """Test that YOLO_V8 has VendoredSource."""
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert isinstance(info.source, VendoredSource)

    def test_yolo_v8_has_correct_subdir(self) -> None:
        """Test that YOLO_V8 has correct subdir."""
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert isinstance(info.source, VendoredSource)
        assert info.source.subdir == "yolo"

    def test_yolo_v8_has_correct_filename(self) -> None:
        """Test that YOLO_V8 has correct filename."""
        info = MODEL_REGISTRY[ModelID.YOLO_V8]
        assert info.filename == "model.onnx"

    def test_mobileclip_s2_is_downloadable(self) -> None:
        """Test that MOBILECLIP_S2 has DownloadSource."""
        info = MODEL_REGISTRY[ModelID.MOBILECLIP_S2]
        assert isinstance(info.source, DownloadSource)

    def test_mobileclip_s2_has_correct_hf_repo(self) -> None:
        """Test that MOBILECLIP_S2 has correct HF repo."""
        info = MODEL_REGISTRY[ModelID.MOBILECLIP_S2]
        assert isinstance(info.source, DownloadSource)
        assert info.source.hf_repo_id == "anton96vice/mobileclip2_tflite"

    def test_mobileclip_s2_has_correct_hf_filename(self) -> None:
        """Test that MOBILECLIP_S2 has correct HF filename."""
        info = MODEL_REGISTRY[ModelID.MOBILECLIP_S2]
        assert isinstance(info.source, DownloadSource)
        assert info.source.hf_filename == "mobileclip_s2_datacompdr_last.tflite"

    def test_mobileclip_s2_has_correct_filename(self) -> None:
        """Test that MOBILECLIP_S2 has correct filename."""
        info = MODEL_REGISTRY[ModelID.MOBILECLIP_S2]
        assert info.filename == "mobileclip_s2_datacompdr_last.tflite"

    def test_smolvlm2_2_2b_is_transformers_source(self) -> None:
        """Test that SMOLVLM2_2_2B has TransformersSource."""
        info = MODEL_REGISTRY[ModelID.SMOLVLM2_2_2B]
        assert isinstance(info.source, TransformersSource)

    def test_smolvlm2_2_2b_has_correct_hf_repo(self) -> None:
        """Test that SMOLVLM2_2_2B has correct HF repo."""
        info = MODEL_REGISTRY[ModelID.SMOLVLM2_2_2B]
        assert isinstance(info.source, TransformersSource)
        assert info.source.hf_repo_id == "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    def test_smolvlm2_2_2b_has_empty_filename(self) -> None:
        """Test that SMOLVLM2_2_2B has an empty filename (directory source)."""
        info = MODEL_REGISTRY[ModelID.SMOLVLM2_2_2B]
        assert info.filename == ""

    @pytest.mark.parametrize(
        "model_id",
        [ModelID.YOLO_V8, ModelID.MOBILECLIP_S2, ModelID.SMOLVLM2_2_2B],
    )
    def test_registry_entries_are_model_info(self, model_id: ModelID) -> None:
        """Test that all registry entries are ModelInfo instances."""
        info = MODEL_REGISTRY[model_id]
        assert isinstance(info, ModelInfo)

    @pytest.mark.parametrize(
        "model_id",
        [ModelID.YOLO_V8, ModelID.MOBILECLIP_S2, ModelID.SMOLVLM2_2_2B],
    )
    def test_registry_entry_id_matches_key(self, model_id: ModelID) -> None:
        """Test that registry entry ID matches its key."""
        info = MODEL_REGISTRY[model_id]
        assert info.id == model_id
