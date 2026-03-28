"""Pipeline configuration.

Loads config.yaml from the project root and validates all values with pydantic.
Any field can be overridden by an environment variable prefixed with M2A_
using double-underscore for nesting, e.g.:

    M2A_LLM__MODEL_PATH=/new/path uv run python -m scripts.run_yolo_pipeline

Usage:
    from moment_to_action.config import settings

    print(settings.llm.model_path)
    print(settings.llm.n_threads)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

##Loud failure. Required? or just skip to defaults if not found?
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"config.yaml not found at {_CONFIG_PATH}. "
        "Copy config.yaml to the project root before running."
    )

# ── Sub-models (one per pipeline stage) ──────────────────────────────────────

class LLMConfig(BaseModel):
    """Configuration for the LLM inference stage."""

    model_path: str = Field(
        default="./llm_models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        description="Path to the GGUF model file.",
    )
    n_ctx: int = Field(default=512, ge=64, le=8192,
        description="Context window size. Keep short to limit KV cache RAM.")
    n_threads: int = Field(default=6, ge=1, le=16,
        description="CPU threads for inference. Leave 2 cores free for pipeline.")
    n_gpu_layers: int = Field(default=0, ge=0,
        description="Layers to offload to GPU. Set to 28 to try Adreno offload.")
    max_tokens: int = Field(default=80, ge=1, le=512,
        description="Hard cap on generated tokens per inference call.")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0,
        description="Sampling temperature. 0.0 = fully deterministic.")
    verbose: bool = Field(default=False,
        description="Enable llama.cpp verbose output (shows memory breakdown).")


class YOLOConfig(BaseModel):
    """Configuration for the YOLO detection stage."""

    model_path: str = Field(
        default="./models/yolo/model.onnx",
        description="Path to the YOLO ONNX model file.",
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0,
        description="Minimum confidence to keep a detection.")
    max_detections: int = Field(default=5, ge=1, le=100,
        description="Maximum detections to pass to the LLM stage.")


class PipelineConfig(BaseModel):
    """Top-level pipeline runtime configuration."""

    input_source: str = Field(
        default="./images/test.jpg",
        description="Input source — image path, mp4 path, or 0 for live camera.",
    )
    log_level: str = Field(default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.")


# ── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """Root configuration object. Loaded from config.yaml at project root.

    Priority (highest to lowest):
        1. Explicit init arguments
        2. M2A_ environment variables  (e.g. M2A_LLM__TEMPERATURE=0.5)
        3. config.yaml
        4. Field defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="M2A_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    llm: LLMConfig = Field(default_factory=LLMConfig)
    yolo: YOLOConfig = Field(default_factory=YOLOConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=_CONFIG_PATH),
            file_secret_settings,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
# Loaded once at import time. All modules import this object directly.
settings = Settings()
