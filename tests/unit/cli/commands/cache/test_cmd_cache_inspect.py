"""Unit tests for m2a cache inspect command."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from moment_to_action.config import AppConfig
from moment_to_action.paths._cache._manager import CacheInfo
from moment_to_action.paths._cache._models import CachedModelInfo, ModelCacheContents

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _make_cache_info(
    total_bytes: int = 0,
    models: dict[str, CachedModelInfo] | None = None,
    root_contents: Iterable[Path] = (),
) -> CacheInfo:
    """Return a real CacheInfo."""
    models = models or {}
    return CacheInfo(
        total_size_bytes=total_bytes,
        root_contents=list(root_contents),
        models_info=ModelCacheContents(
            total_size_bytes=sum(m.size_bytes for m in models.values()),
            models=models,
            other=[],
        ),
    )


def _patched_path_manager(info: CacheInfo) -> MagicMock:
    """Return a MagicMock standing in for PathManager(), with cache.inspect_cache → info."""
    mock_path_man = MagicMock()
    mock_path_man.cache.inspect_cache.return_value = info
    return mock_path_man


@pytest.mark.unit
class TestCacheInspectCommand:
    """Tests for the cache inspect subcommand."""

    def test_default_output_shows_table_title(self) -> None:
        """Default output prints the cache size in the table title."""
        from moment_to_action._cli import cli

        info = _make_cache_info(
            total_bytes=50_000_000,
            models={
                "yolo_v8": CachedModelInfo(
                    model_id="yolo_v8",
                    size_bytes=50_000_000,
                    variants=["default"],
                    other=[],
                ),
            },
        )
        mock_pm = _patched_path_manager(info)
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_pm):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "inspect"])

        assert result.exit_code == 0
        assert "Cache" in result.output
        assert "Subcache" in result.output
        assert "models" in result.output

    def test_json_flag_outputs_valid_json(self) -> None:
        """--json flag produces valid JSON output."""
        from moment_to_action._cli import cli

        info = _make_cache_info(0)
        mock_pm = _patched_path_manager(info)
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_pm):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert isinstance(output_json, dict)

    def test_json_output_matches_to_json(self) -> None:
        """JSON output matches CacheInfo.to_json() shape."""
        from moment_to_action._cli import cli

        info = _make_cache_info(
            total_bytes=12345,
            models={
                "yolo_v8": CachedModelInfo(
                    model_id="yolo_v8",
                    size_bytes=12345,
                    variants=["default"],
                    other=[],
                ),
            },
        )
        mock_pm = _patched_path_manager(info)
        with patch("moment_to_action._cli.init_logging"):
            with patch("moment_to_action._cli.PathManager", return_value=mock_pm):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    result = CliRunner().invoke(cli, ["cache", "inspect", "--json"])

        assert result.exit_code == 0
        output_json = json.loads(result.output)
        assert output_json["total_size_bytes"] == 12345
        assert "models_info" in output_json
        assert "yolo_v8" in output_json["models_info"]["models"]

    def test_exit_code_zero_on_success(self) -> None:
        """Command exits with code 0 on success."""
        from moment_to_action._cli import cli

        with patch("moment_to_action._cli.init_logging"):
            with patch(
                "moment_to_action._cli.PathManager",
                return_value=_patched_path_manager(_make_cache_info(0)),
            ):
                with patch("moment_to_action._cli.load_config", return_value=AppConfig()):
                    runner = CliRunner()
                    result_default = runner.invoke(cli, ["cache", "inspect"])
                    result_json = runner.invoke(cli, ["cache", "inspect", "--json"])

        assert result_default.exit_code == 0
        assert result_json.exit_code == 0
