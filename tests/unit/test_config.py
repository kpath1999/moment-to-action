"""Unit tests for config module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from moment_to_action.config import AppConfig, load_config, save_config

if TYPE_CHECKING:
    from pathlib import Path

    from moment_to_action.paths import PathManager


@pytest.mark.unit
class TestAppConfig:
    """Tests for AppConfig model."""

    def test_default_max_workers(self) -> None:
        """max_workers defaults to cpu_count (at least 1)."""
        config = AppConfig()
        assert config.max_workers >= 1

    def test_default_log_level(self) -> None:
        """log_level defaults to INFO."""
        config = AppConfig()
        assert config.log_level == "INFO"

    def test_max_workers_ge1_validation(self) -> None:
        """max_workers=0 raises ValidationError."""
        with pytest.raises(ValidationError):
            AppConfig(max_workers=0)

    def test_log_level_invalid_raises(self) -> None:
        """Unknown log level raises ValidationError."""
        with pytest.raises(ValidationError):
            AppConfig(log_level="TRACE")  # type: ignore[arg-type]

    def test_custom_values(self) -> None:
        """Custom field values are stored correctly."""
        config = AppConfig(max_workers=4, log_level="DEBUG")
        assert config.max_workers == 4
        assert config.log_level == "DEBUG"


@pytest.mark.unit
class TestLoadConfig:
    """Tests for load_config."""

    def test_load_creates_file_when_missing(self, path_manager: PathManager) -> None:
        """load_config writes default file when path does not exist."""
        path = path_manager.app_config_file
        assert not path.exists()
        load_config(path)
        assert path.exists()

    def test_load_returns_defaults_when_missing(self, path_manager: PathManager) -> None:
        """load_config returns default AppConfig when file absent."""
        config = load_config(path_manager.app_config_file)
        assert config.max_workers >= 1
        assert config.log_level == "INFO"

    def test_load_existing_file(self, path_manager: PathManager) -> None:
        """load_config parses an existing config file correctly."""
        path = path_manager.app_config_file
        save_config(AppConfig(max_workers=4, log_level="DEBUG"), path)

        config = load_config(path)
        assert config.max_workers == 4
        assert config.log_level == "DEBUG"

    def test_load_normalizes_format(self, path_manager: PathManager) -> None:
        """load_config re-saves the file to normalize its format."""
        path = path_manager.app_config_file
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write non-normalized JSON (compact, no indent)
        path.write_text('{"max_workers":2,"log_level":"WARNING"}')

        load_config(path)

        # File should now have indented format
        content = path.read_text()
        assert "\n" in content  # indented output contains newlines

    def test_load_partial_json_fills_defaults(self, path_manager: PathManager) -> None:
        """load_config fills missing fields with defaults when file has partial data."""
        path = path_manager.app_config_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"max_workers": 3}')

        config = load_config(path)
        assert config.max_workers == 3
        assert config.log_level == "INFO"


@pytest.mark.unit
class TestSaveConfig:
    """Tests for save_config."""

    def test_save_writes_json(self, tmp_path: Path) -> None:
        """save_config writes valid JSON to the given path."""
        import json

        path = tmp_path / "config.json"
        config = AppConfig(max_workers=2, log_level="WARNING")
        save_config(config, path)

        data = json.loads(path.read_text())
        assert data["max_workers"] == 2
        assert data["log_level"] == "WARNING"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_config creates parent directories if they do not exist."""
        path = tmp_path / "deep" / "nested" / "config.json"
        save_config(AppConfig(), path)
        assert path.exists()

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Save then load returns the same config values."""
        path = tmp_path / "config.json"
        original = AppConfig(max_workers=8, log_level="ERROR")
        save_config(original, path)

        loaded = load_config(path)
        assert loaded.max_workers == original.max_workers
        assert loaded.log_level == original.log_level
