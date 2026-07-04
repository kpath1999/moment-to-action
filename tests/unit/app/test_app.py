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

    def test_get_pipeline_returns_registered_handle(self, app: Moment2Action) -> None:
        """get_pipeline() looks a previously built handle up by name."""
        built = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        assert app.get_pipeline("p1") is built

    def test_get_pipeline_unknown_name_raises(self, app: Moment2Action) -> None:
        """get_pipeline() on an unregistered name raises KeyError."""
        with pytest.raises(KeyError):
            app.get_pipeline("missing")

    def test_load_pipeline_loads_and_activates(self, app: Moment2Action) -> None:
        """load_pipeline() loads the given handle and returns it."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        result = app.load_pipeline(handle)
        assert result is handle
        assert handle.loaded is True

    def test_load_pipeline_unregistered_handle_raises(self, app: Moment2Action) -> None:
        """load_pipeline() on a handle not registered on this app raises ValueError."""
        other_app = Moment2Action()
        foreign = other_app.new_pipeline("p1").add_stage(_NoModelStage).build()
        with pytest.raises(ValueError, match="not registered"):
            app.load_pipeline(foreign)

    def test_load_pipeline_while_another_active_raises(self, app: Moment2Action) -> None:
        """load_pipeline() raises if a different pipeline is already active."""
        h1 = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        h2 = app.new_pipeline("p2").add_stage(_NoModelStage).build()
        app.load_pipeline(h1)
        with pytest.raises(RuntimeError, match="already active"):
            app.load_pipeline(h2)

    def test_load_pipeline_same_handle_twice_raises(self, app: Moment2Action) -> None:
        """Loading the same already-loaded handle again raises RuntimeError."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline(handle)
        # PipelineHandle.load() itself guards against double-loading.
        with pytest.raises(RuntimeError, match="already loaded"):
            app.load_pipeline(handle)

    def test_unload_pipeline_defaults_to_active(self, app: Moment2Action) -> None:
        """unload_pipeline() with no handle unloads the currently active pipeline."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline(handle)
        app.unload_pipeline()
        assert handle.loaded is False

    def test_unload_pipeline_keeps_registration(self, app: Moment2Action) -> None:
        """unload_pipeline() does not remove the pipeline from tracking."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline(handle)
        app.unload_pipeline(handle)
        # Still registered: load_pipeline works again without re-building.
        app.load_pipeline(handle)
        assert handle.loaded is True

    def test_unload_pipeline_no_active_raises(self, app: Moment2Action) -> None:
        """unload_pipeline() with no handle and nothing active raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No pipeline is currently active"):
            app.unload_pipeline()

    def test_unload_pipeline_unregistered_handle_raises(self, app: Moment2Action) -> None:
        """unload_pipeline() on a handle not registered on this app raises ValueError."""
        other_app = Moment2Action()
        foreign = other_app.new_pipeline("p1").add_stage(_NoModelStage).build()
        with pytest.raises(ValueError, match="not registered"):
            app.unload_pipeline(foreign)

    def test_remove_pipeline_unloads_and_deletes(self, app: Moment2Action) -> None:
        """remove_pipeline() unloads (if loaded) and fully discards the registration."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline(handle)
        app.remove_pipeline(handle)
        with pytest.raises(KeyError):
            app.get_pipeline("p1")

    def test_remove_pipeline_allows_name_reuse(self, app: Moment2Action) -> None:
        """After remove_pipeline(), the same name can be registered again."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.remove_pipeline(handle)
        app.new_pipeline("p1").add_stage(_NoModelStage).build()  # should not raise

    def test_remove_pipeline_unregistered_handle_raises(self, app: Moment2Action) -> None:
        """remove_pipeline() on a handle not registered on this app raises ValueError."""
        other_app = Moment2Action()
        foreign = other_app.new_pipeline("p1").add_stage(_NoModelStage).build()
        with pytest.raises(ValueError, match="not registered"):
            app.remove_pipeline(foreign)

    def test_metrics_report_defaults_to_active(self, app: Moment2Action) -> None:
        """metrics_report() with no handle reports on the currently active pipeline."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline(handle)
        report = app.metrics_report()
        assert report is not None

    def test_metrics_report_by_handle(self, app: Moment2Action) -> None:
        """metrics_report(handle) reports on that specific pipeline, active or not."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        report = app.metrics_report(handle)
        assert report is not None

    def test_metrics_report_no_active_raises(self, app: Moment2Action) -> None:
        """metrics_report() with no handle and nothing active raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No pipeline is currently active"):
            app.metrics_report()

    def test_metrics_report_unregistered_handle_raises(self, app: Moment2Action) -> None:
        """metrics_report(handle) on a handle not registered on this app raises ValueError."""
        other_app = Moment2Action()
        foreign = other_app.new_pipeline("p1").add_stage(_NoModelStage).build()
        with pytest.raises(ValueError, match="not registered"):
            app.metrics_report(foreign)

    def test_close_unloads_active_pipeline(self, app: Moment2Action) -> None:
        """close() unloads the active pipeline if one exists."""
        handle = app.new_pipeline("p1").add_stage(_NoModelStage).build()
        app.load_pipeline(handle)
        app.close()
        assert handle.loaded is False

    def test_close_with_no_active_pipeline_is_noop(self, app: Moment2Action) -> None:
        """close() with nothing active does not raise."""
        app.close()

    def test_context_manager_calls_close_on_exit(self, app: Moment2Action) -> None:
        """Using Moment2Action as a context manager closes the active pipeline on exit."""
        with app as ctx_app:
            assert ctx_app is app
            handle = ctx_app.new_pipeline("p1").add_stage(_NoModelStage).build()
            ctx_app.load_pipeline(handle)
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
