"""Unit tests for Moment2Action."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from moment_to_action.app._app import Moment2Action
from moment_to_action.config import AppConfig
from moment_to_action.stages._base import Stage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from moment_to_action.messages import Message


class _NoModelStage(Stage):
    """A stage that takes no model — safe to load/unload without touching hardware."""

    def _process(self, items: list[Message]) -> Iterator[Message]:
        """Yield nothing."""
        yield from ()


@pytest.fixture
def app() -> Moment2Action:
    """Return a real Moment2Action instance for testing."""
    return Moment2Action()


@pytest.mark.unit
class TestMoment2Action:
    """Tests for Moment2Action."""

    def test_new_pipeline_returns_builder(self, app: Moment2Action) -> None:
        """new_pipeline() returns a PipelineBuilder that can add stages."""
        builder = app.new_pipeline("p1")
        builder.add_stage(_NoModelStage)
        handle = builder.build()
        assert handle.name == "p1"

    def test_new_pipeline_duplicate_name_raises(self, app: Moment2Action) -> None:
        """Registering the same pipeline name twice raises ValueError."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        with pytest.raises(ValueError, match="already registered"):
            app.new_pipeline("p1")

    def test_build_does_not_load(self, app: Moment2Action) -> None:
        """build() alone never loads the pipeline."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        assert handle.loaded is False

    def test_load_pipeline_loads_and_activates(self, app: Moment2Action) -> None:
        """load_pipeline() loads the named pipeline and returns its handle."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        handle = app.load_pipeline("p1")
        assert handle.loaded is True

    def test_load_pipeline_unknown_name_raises(self, app: Moment2Action) -> None:
        """load_pipeline() on an unregistered name raises KeyError."""
        with pytest.raises(KeyError):
            app.load_pipeline("missing")

    def test_load_pipeline_while_another_active_raises(self, app: Moment2Action) -> None:
        """load_pipeline() raises if a different pipeline is already active."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.new_pipeline("p2").add_stage(_NoModelStage).build()
        app.load_pipeline("p1")
        with pytest.raises(RuntimeError, match="already active"):
            app.load_pipeline("p2")

    def test_load_pipeline_same_name_twice_is_idempotent_call(self, app: Moment2Action) -> None:
        """Calling load_pipeline() again for the currently-active pipeline is allowed."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline("p1")
        # Re-entering load_pipeline for the same active name should not raise the
        # "different pipeline active" error (PipelineHandle.load itself guards
        # against double-loading).
        with pytest.raises(RuntimeError, match="already loaded"):
            app.load_pipeline("p1")

    def test_unload_pipeline_defaults_to_active(self, app: Moment2Action) -> None:
        """unload_pipeline() with no name unloads the currently active pipeline."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        handle = app.load_pipeline("p1")
        app.unload_pipeline()
        assert handle.loaded is False

    def test_unload_pipeline_keeps_registration(self, app: Moment2Action) -> None:
        """unload_pipeline() does not remove the pipeline from tracking."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline("p1")
        app.unload_pipeline("p1")
        # Still registered: load_pipeline works again without re-building.
        handle = app.load_pipeline("p1")
        assert handle.loaded is True

    def test_unload_pipeline_no_active_raises(self, app: Moment2Action) -> None:
        """unload_pipeline() with no name and nothing active raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No pipeline is currently active"):
            app.unload_pipeline()

    def test_unload_pipeline_unknown_name_raises(self, app: Moment2Action) -> None:
        """unload_pipeline() on an unregistered name raises KeyError."""
        with pytest.raises(KeyError):
            app.unload_pipeline("missing")

    def test_remove_pipeline_unloads_and_deletes(self, app: Moment2Action) -> None:
        """remove_pipeline() unloads (if loaded) and fully discards the registration."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline("p1")
        app.remove_pipeline("p1")
        with pytest.raises(KeyError):
            app.load_pipeline("p1")

    def test_remove_pipeline_allows_name_reuse(self, app: Moment2Action) -> None:
        """After remove_pipeline(), the same name can be registered again."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.remove_pipeline("p1")
        app.new_pipeline("p1").add_stage(_NoModelStage).build()  # should not raise

    def test_remove_pipeline_unknown_name_raises(self, app: Moment2Action) -> None:
        """remove_pipeline() on an unregistered name raises KeyError."""
        with pytest.raises(KeyError):
            app.remove_pipeline("missing")

    def test_metrics_report_defaults_to_active(self, app: Moment2Action) -> None:
        """metrics_report() with no name reports on the currently active pipeline."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline("p1")
        report = app.metrics_report()
        assert report is not None

    def test_metrics_report_by_name(self, app: Moment2Action) -> None:
        """metrics_report(name) reports on that specific pipeline, active or not."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        report = app.metrics_report("p1")
        assert report is not None

    def test_metrics_report_no_active_raises(self, app: Moment2Action) -> None:
        """metrics_report() with no name and nothing active raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No pipeline is currently active"):
            app.metrics_report()

    def test_metrics_report_unknown_name_raises(self, app: Moment2Action) -> None:
        """metrics_report(name) on an unregistered name raises KeyError."""
        with pytest.raises(KeyError):
            app.metrics_report("missing")

    def test_close_unloads_active_pipeline(self, app: Moment2Action) -> None:
        """close() unloads the active pipeline if one exists."""
        app.new_pipeline("p1").add_stage(_NoModelStage).build()
        handle = app.load_pipeline("p1")
        app.close()
        assert handle.loaded is False

    def test_close_with_no_active_pipeline_is_noop(self, app: Moment2Action) -> None:
        """close() with nothing active does not raise."""
        app.close()

    def test_context_manager_calls_close_on_exit(self, app: Moment2Action) -> None:
        """Using Moment2Action as a context manager closes the active pipeline on exit."""
        with app as ctx_app:
            assert ctx_app is app
            ctx_app.new_pipeline("p1").add_stage(_NoModelStage).build()
            handle = ctx_app.load_pipeline("p1")
        assert handle.loaded is False

    def test_qairt_flag_configures_env_when_sdk_path_missing(self) -> None:
        """qairt=True with no configured SDK path logs a warning but does not raise."""
        Moment2Action(qairt=True)

    def test_qairt_flag_configures_env_when_sdk_path_set(self) -> None:
        """qairt=True with a configured SDK path calls QairtSDKManager.configure_env()."""
        config = AppConfig(qairt_sdk_path=Path("/opt/qairt"))
        with patch("moment_to_action.app._app.QairtSDKManager") as mock_manager_cls:
            Moment2Action(config, qairt=True)
        mock_manager_cls.from_app_config.assert_called_once()
        mock_manager_cls.from_app_config.return_value.configure_env.assert_called_once()

    def test_qairt_flag_swallows_configure_env_runtime_error(self) -> None:
        """A RuntimeError from configure_env() is logged, not raised."""
        config = AppConfig(qairt_sdk_path=Path("/opt/qairt"))
        with patch("moment_to_action.app._app.QairtSDKManager") as mock_manager_cls:
            mock_manager_cls.from_app_config.return_value.configure_env.side_effect = RuntimeError(
                "boom"
            )
            Moment2Action(config, qairt=True)
