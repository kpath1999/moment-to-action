"""Moment2Action — the single application container consumers of this library import."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from moment_to_action._logging import init_logging
from moment_to_action.config import load_config
from moment_to_action.hardware import Platform
from moment_to_action.metrics import MetricsCollector
from moment_to_action.models import ModelManager
from moment_to_action.paths import PathManager
from moment_to_action.qairt import QairtSDKManager

from ._builder import PipelineBuilder

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from moment_to_action.config import AppConfig
    from moment_to_action.metrics import MetricsReport

    from ._handle import PipelineHandle


class Moment2Action:
    """Application container: the one import consumers of this library need.

    Owns path/config/logging setup, the hardware :class:`~moment_to_action.hardware.Platform`,
    and every pipeline built through it. Hides ``PathManager``, ``ModelManager``, and the raw
    :class:`~moment_to_action.pipeline.Pipeline` — build pipelines with
    :meth:`new_pipeline`/:meth:`~moment_to_action.app._builder.PipelineBuilder.add_stage`, then
    load/run/unload them via their handle (:meth:`get_pipeline` looks one up by name). Only one
    pipeline may be loaded (holding device resources) at a time.
    """

    def __init__(self, config: AppConfig | None = None, *, qairt: bool = True) -> None:
        """Set up the app: path manager, config, logging, and hardware platform.

        Args:
            config: Application configuration. When ``None``, loads (and persists
                defaults for) the config at the platform-standard location.
            qairt: Whether to configure the QAIRT SDK environment (``QAIRT_SDK_ROOT``
                etc.) before building the hardware platform. Needed for DLC/NPU
                backends; a no-op (with a logged warning) if the SDK path isn't
                configured.
        """
        self._path_manager = PathManager()
        self._config = config or load_config(self._path_manager.app_config_file)
        init_logging(log_level=self._config.log_level)
        if qairt:
            self._configure_qairt()
        self._platform = Platform(self._config)
        self._pipelines: dict[str, PipelineHandle] = {}
        self._active_pipeline: str | None = None

    def _configure_qairt(self) -> None:
        """Set up the QAIRT SDK environment for DLC/NPU backends, if configured."""
        if self._config.qairt_sdk_path is None:
            logger.warning("QAIRT SDK path not configured — DLC backends may be unavailable.")
            return
        try:
            QairtSDKManager.from_app_config(self._config, self._path_manager).configure_env()
        except RuntimeError:
            logger.warning("QAIRT env setup failed.", exc_info=True)

    def new_pipeline(self, name: str) -> PipelineBuilder:
        """Start building a pipeline registered under *name*.

        Args:
            name: Unique name for the new pipeline.

        Returns:
            A fluent :class:`~moment_to_action.app._builder.PipelineBuilder` — chain
            ``.add_stage(...)`` calls and finish with ``.build()``.

        Raises:
            ValueError: If *name* is already registered. Call :meth:`remove_pipeline`
                first to reuse a name.
        """
        if name in self._pipelines:
            msg = f"Pipeline {name!r} is already registered."
            raise ValueError(msg)
        metrics = MetricsCollector(self._platform)
        model_manager = ModelManager(self._path_manager, metrics=metrics)
        return PipelineBuilder(self, name, metrics, model_manager)

    def _register_pipeline(self, handle: PipelineHandle) -> None:
        """Register a newly built, unloaded pipeline handle.

        Args:
            handle: The handle produced by ``PipelineBuilder.build()``.
        """
        self._pipelines[handle.name] = handle

    def _check_registered(self, handle: PipelineHandle) -> None:
        """Verify *handle* is still registered on this app under its own name.

        Args:
            handle: Handle to verify.

        Raises:
            ValueError: If *handle* isn't the (or isn't a) pipeline registered
                under its name — e.g. it was already removed, or belongs to a
                different app.
        """
        if self._pipelines.get(handle.name) is not handle:
            msg = f"Pipeline handle {handle.name!r} is not registered on this app."
            raise ValueError(msg)

    def get_pipeline(self, name: str) -> PipelineHandle:
        """Look up a previously built pipeline by name.

        Args:
            name: Name passed to :meth:`new_pipeline` when it was built.

        Returns:
            The registered ``PipelineHandle``.

        Raises:
            KeyError: If *name* is not a registered pipeline.
        """
        if name not in self._pipelines:
            msg = f"No pipeline registered under {name!r}."
            raise KeyError(msg)
        return self._pipelines[name]

    def load_pipeline(self, handle: PipelineHandle) -> PipelineHandle:
        """Load every stage of *handle* and make it the active pipeline.

        Args:
            handle: A pipeline handle from ``PipelineBuilder.build()`` or
                :meth:`get_pipeline`.

        Returns:
            *handle*, now loaded.

        Raises:
            ValueError: If *handle* isn't registered on this app.
            RuntimeError: If a different pipeline is already active.
        """
        self._check_registered(handle)
        if self._active_pipeline is not None and self._active_pipeline != handle.name:
            msg = (
                f"Pipeline {self._active_pipeline!r} is already active — "
                "call unload_pipeline() first."
            )
            raise RuntimeError(msg)
        handle.load(self._platform)
        self._active_pipeline = handle.name
        return handle

    def unload_pipeline(self, handle: PipelineHandle | None = None) -> None:
        """Unload a pipeline's stages without deleting its registration.

        Args:
            handle: Pipeline to unload. Defaults to the currently active pipeline.

        Raises:
            RuntimeError: If *handle* is ``None`` and no pipeline is active.
            ValueError: If *handle* isn't registered on this app.
        """
        if handle is None:
            if self._active_pipeline is None:
                msg = "No pipeline is currently active."
                raise RuntimeError(msg)
            handle = self._pipelines[self._active_pipeline]
        else:
            self._check_registered(handle)
        handle.unload()
        if self._active_pipeline == handle.name:
            self._active_pipeline = None

    def remove_pipeline(self, handle: PipelineHandle) -> None:
        """Unload (if loaded) and fully discard a pipeline's registration.

        Args:
            handle: Pipeline to remove.

        Raises:
            ValueError: If *handle* isn't registered on this app.
        """
        self._check_registered(handle)
        if handle.loaded:
            self.unload_pipeline(handle)
        del self._pipelines[handle.name]

    def metrics_report(self, handle: PipelineHandle | None = None) -> MetricsReport:
        """Return a pipeline's own metrics report.

        Args:
            handle: Pipeline to report on. Defaults to the currently active pipeline.

        Returns:
            The report generated by that pipeline's own metrics collector.

        Raises:
            RuntimeError: If *handle* is ``None`` and no pipeline is active.
            ValueError: If *handle* isn't registered on this app.
        """
        if handle is None:
            if self._active_pipeline is None:
                msg = "No pipeline is currently active."
                raise RuntimeError(msg)
            handle = self._pipelines[self._active_pipeline]
        else:
            self._check_registered(handle)
        return handle.metrics_report()

    def close(self) -> None:
        """Unload the active pipeline, if any."""
        if self._active_pipeline is not None:
            self.unload_pipeline()

    def __enter__(self) -> Self:
        """Return self for use in a ``with`` block."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Call :meth:`close` on exit from the ``with`` block."""
        self.close()
