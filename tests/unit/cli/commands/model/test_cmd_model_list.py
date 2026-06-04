"""Unit tests for m2a model list command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from moment_to_action.config import AppConfig
from moment_to_action.models import ModelID
from moment_to_action.models._model_info import ModelStatus, VariantStatus
from moment_to_action.models._registry import MODEL_REGISTRY


def _make_variant_status(
    model_id: ModelID,
    variant: str,
    *,
    available: bool = False,
    path: Path | None = None,
    size_bytes: int | None = None,
) -> VariantStatus:
    """Build a VariantStatus for testing."""
    return VariantStatus(
        model_id=model_id,
        variant=variant,
        available=available,
        path=path,
        size_bytes=size_bytes,
    )


def _invoke(args: list[str], tmp_path: Path, mock_mgr: MagicMock) -> Result:
    from moment_to_action._cli import cli

    with patch("moment_to_action._cli.init_logging"):
        with patch(
            "moment_to_action._cli.PathManager",
            return_value=MagicMock(app_config_file=tmp_path / "cfg.json"),
        ):
            with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                with patch(
                    "moment_to_action._cli.commands.cmd_model.cmd_list.ModelManager",
                    return_value=mock_mgr,
                ):
                    return CliRunner().invoke(cli, ["model", "list", *args])


def _make_mgr(statuses: list[ModelStatus]) -> MagicMock:
    mgr = MagicMock()
    mgr.list_models.return_value = statuses
    return mgr


@pytest.mark.unit
class TestModelListCommand:
    """Tests for m2a model list."""

    def test_exits_zero_empty_registry(self, tmp_path: Path) -> None:
        """Command exits 0 when registry returns no models."""
        mgr = _make_mgr([])
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0

    def test_vendored_status_shown(self, tmp_path: Path) -> None:
        """Vendored variant shows 'vendored' status."""
        mid = ModelID.YOLO_V8
        vs = _make_variant_status(mid, "default", available=True)
        info = MODEL_REGISTRY[mid]
        status = ModelStatus(info=info, variants=[vs], path=None)
        mgr = _make_mgr([status])
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0
        assert "vendored" in result.output

    def test_cached_status_shown(self, tmp_path: Path) -> None:
        """Non-vendored available variant shows 'cached'."""
        mid = ModelID.YOLO_V8
        vs = _make_variant_status(
            mid, "qcs6490", available=True, path=tmp_path / "model.dlc", size_bytes=1024
        )
        # Use qcs6490 which is HuggingFaceSource (non-vendored)
        info = MODEL_REGISTRY[mid]
        status = ModelStatus(info=info, variants=[vs], path=tmp_path)
        mgr = _make_mgr([status])
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0
        assert "cached" in result.output

    def test_not_downloaded_status_shown(self, tmp_path: Path) -> None:
        """Non-vendored unavailable variant shows 'not downloaded'."""
        mid = ModelID.YOLO_V8
        vs = _make_variant_status(mid, "qcs6490", available=False)
        info = MODEL_REGISTRY[mid]
        status = ModelStatus(info=info, variants=[vs], path=None)
        mgr = _make_mgr([status])
        result = _invoke([], tmp_path, mgr)
        assert result.exit_code == 0
        assert "not downloaded" in result.output

    def test_size_shown_when_cached(self, tmp_path: Path) -> None:
        """Cached variant shows its size in bytes."""
        mid = ModelID.YOLO_V8
        vs = _make_variant_status(mid, "qcs6490", available=True, path=tmp_path, size_bytes=2048)
        info = MODEL_REGISTRY[mid]
        status = ModelStatus(info=info, variants=[vs], path=tmp_path)
        mgr = _make_mgr([status])
        result = _invoke([], tmp_path, mgr)
        assert "2,048" in result.output

    def test_model_id_shown(self, tmp_path: Path) -> None:
        """Model ID value appears in the output."""
        mid = ModelID.YOLO_V8
        vs = _make_variant_status(mid, "default", available=False)
        info = MODEL_REGISTRY[mid]
        status = ModelStatus(info=info, variants=[vs], path=None)
        mgr = _make_mgr([status])
        result = _invoke([], tmp_path, mgr)
        assert "yolo_v8" in result.output

    def test_json_flag_exits_zero(self, tmp_path: Path) -> None:
        """--json exits 0."""
        mgr = _make_mgr([])
        result = _invoke(["--json"], tmp_path, mgr)
        assert result.exit_code == 0

    def test_json_flag_outputs_valid_json(self, tmp_path: Path) -> None:
        """--json outputs a valid JSON list."""
        mgr = _make_mgr([])
        result = _invoke(["--json"], tmp_path, mgr)
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_json_includes_model_fields(self, tmp_path: Path) -> None:
        """--json rows contain expected keys."""
        mid = ModelID.YOLO_V8
        vs = _make_variant_status(mid, "qcs6490", available=True, path=tmp_path, size_bytes=512)
        info = MODEL_REGISTRY[mid]
        status = ModelStatus(info=info, variants=[vs], path=tmp_path)
        mgr = _make_mgr([status])
        result = _invoke(["--json"], tmp_path, mgr)
        rows = json.loads(result.output)
        assert len(rows) == 1
        row = rows[0]
        assert row["model_id"] == "yolo_v8"
        assert row["variant"] == "qcs6490"
        assert "format" in row
        assert row["status"] == "cached"
        assert row["size_bytes"] == 512
