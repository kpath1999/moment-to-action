"""Application configuration model and persistence."""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003
from typing import Literal

from pydantic import BaseModel, Field


class AppConfig(BaseModel, frozen=True):
    """Application configuration."""

    max_workers: int = Field(default_factory=lambda: os.cpu_count() or 1, ge=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    qairt_sdk_version: str = "2.45.0"
    qairt_sdk_path: Path | None = None


def load_config(path: Path) -> AppConfig:
    """Load config from path, writing defaults if the file does not exist."""
    if not path.exists():
        config = AppConfig()
        save_config(config, path)
        return config
    config = AppConfig.model_validate_json(path.read_text())
    save_config(config, path)  # normalize to standard format
    return config


def save_config(config: AppConfig, path: Path) -> None:
    """Save config to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))
