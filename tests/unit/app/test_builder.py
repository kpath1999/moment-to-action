"""Unit tests for PipelineBuilder."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from moment_to_action.app._builder import PipelineBuilder
from moment_to_action.app._handle import PipelineHandle
from moment_to_action.hardware import ComputeUnit
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelID
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message


class _NoModelStage(Stage):
    """A stage that takes no model, just records its metrics collector."""

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Yield nothing."""
        yield from ()


class _ModelStage(Stage):
    """A stage that takes a model as its first positional argument."""

    def __init__(self, model: object, *, metrics: MetricsCollector | None = None) -> None:
        """Store the model for assertions."""
        super().__init__(metrics=metrics)
        self.model = model

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Yield nothing."""
        yield from ()


def _builder(app: MagicMock | None = None) -> tuple[PipelineBuilder, MetricsCollector, MagicMock]:
    """Build a PipelineBuilder with a fake app and mocked ModelManager."""
    metrics = MetricsCollector(session_id="builder_test")
    model_manager = MagicMock()
    builder = PipelineBuilder(app or MagicMock(), "pipeline-name", metrics, model_manager)
    return builder, metrics, model_manager


@pytest.mark.unit
class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_add_stage_without_model_id_constructs_stage(self) -> None:
        """add_stage with no model_id constructs the stage with only metrics/kwargs."""
        builder, metrics, model_manager = _builder()

        builder.add_stage(_NoModelStage)

        model_manager.get_model.assert_not_called()
        assert len(builder._stages) == 1
        assert isinstance(builder._stages[0], _NoModelStage)
        assert builder._stages[0]._metrics is metrics

    def test_add_stage_with_model_id_resolves_and_passes_model(self) -> None:
        """add_stage with model_id resolves via ModelManager and passes model positionally."""
        builder, _, model_manager = _builder()
        fake_model = object()
        model_manager.get_model.return_value = fake_model

        builder.add_stage(
            _ModelStage, model_id=ModelID.YOLO_V8, compute_unit=ComputeUnit.CPU, variant="v"
        )

        model_manager.get_model.assert_called_once_with(
            ModelID.YOLO_V8, variant="v", unit=ComputeUnit.CPU
        )
        stage = builder._stages[0]
        assert isinstance(stage, _ModelStage)
        assert stage.model is fake_model

    def test_add_stage_forwards_model_kwargs(self) -> None:
        """model_kwargs are forwarded verbatim to ModelManager.get_model."""
        builder, _, model_manager = _builder()
        model_manager.get_model.return_value = object()

        builder.add_stage(
            _ModelStage,
            model_id=ModelID.YOLO_V8,
            model_kwargs={"max_tokens": 128},
        )

        model_manager.get_model.assert_called_once_with(
            ModelID.YOLO_V8, variant="default", unit=None, max_tokens=128
        )

    def test_add_stage_records_stage_unit_pair(self) -> None:
        """Each add_stage call records (stage, compute_unit) for later load/unload."""
        builder, _, _ = _builder()

        builder.add_stage(_NoModelStage, compute_unit=ComputeUnit.GPU)

        stage = builder._stages[0]
        assert builder._stage_units == [(stage, ComputeUnit.GPU)]

    def test_add_stage_returns_self_for_chaining(self) -> None:
        """add_stage returns the builder itself for fluent chaining."""
        builder, _, _ = _builder()
        result = builder.add_stage(_NoModelStage)
        assert result is builder

    def test_build_registers_handle_on_app(self) -> None:
        """build() registers the resulting handle on the app via _register_pipeline."""
        app = MagicMock()
        builder, _, _ = _builder(app)
        builder.add_stage(_NoModelStage)

        handle = builder.build()

        assert isinstance(handle, PipelineHandle)
        app._register_pipeline.assert_called_once_with(handle)

    def test_build_returns_unloaded_handle(self) -> None:
        """The handle from build() is not loaded and has no device resources touched."""
        builder, _, _ = _builder()
        builder.add_stage(_NoModelStage)

        handle = builder.build()

        assert handle.loaded is False
        assert handle.name == "pipeline-name"

    def test_build_handle_contains_added_stages(self) -> None:
        """The handle's pipeline contains every stage added via add_stage, in order."""
        builder, _, _ = _builder()
        builder.add_stage(_NoModelStage).add_stage(_NoModelStage)

        handle = builder.build()

        assert len(handle.stages) == 2
