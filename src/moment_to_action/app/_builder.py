"""PipelineBuilder — constructs stages/models and assembles them into a PipelineHandle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
from typing_extensions import Self

from moment_to_action.models import DEFAULT_VARIANT_KEY
from moment_to_action.pipeline import Pipeline

from ._handle import PipelineHandle

if TYPE_CHECKING:
    from collections.abc import Callable

    from moment_to_action.hardware import ComputeUnit
    from moment_to_action.metrics import MetricsCollector
    from moment_to_action.models import ModelID, ModelManager
    from moment_to_action.stages._base import Stage

    from ._app import Moment2Action


@attrs.define
class PipelineBuilder:
    """Fluent builder that constructs stages/models for one named pipeline.

    Returned by :meth:`~moment_to_action.app._app.Moment2Action.new_pipeline`. Each
    builder owns its own :class:`~moment_to_action.metrics.MetricsCollector` and
    :class:`~moment_to_action.models.ModelManager`, so every stage/model added via
    :meth:`add_stage` shares one trace scoped to just this pipeline.
    """

    _app: Moment2Action
    _name: str
    _metrics: MetricsCollector
    _model_manager: ModelManager
    _stages: list[Stage] = attrs.field(factory=list)
    _stage_units: list[tuple[Stage, ComputeUnit | None]] = attrs.field(factory=list)

    def add_stage(
        self,
        stage_cls: Callable[..., Stage],
        *,
        model_id: ModelID | None = None,
        variant: str = DEFAULT_VARIANT_KEY,
        compute_unit: ComputeUnit | None = None,
        model_kwargs: dict[str, object] | None = None,
        **stage_kwargs: object,
    ) -> Self:
        """Construct a stage (unloaded) and append it to this pipeline.

        Args:
            stage_cls: The ``Stage`` subclass to construct.
            model_id: When given, resolves + constructs a model via this pipeline's
                own ``ModelManager.get_model(model_id, variant=variant,
                unit=compute_unit, **(model_kwargs or {}))`` and passes it as the
                stage's first positional constructor argument. The model is *not*
                loaded here — that happens later, for every stage in the pipeline
                at once, via ``Moment2Action.load_pipeline``.
            variant: Model variant to resolve. Ignored when ``model_id`` is None.
            compute_unit: Compute unit the model will run on. Recorded alongside
                the stage so ``load_pipeline`` knows what to pass to
                ``stage.load(platform, unit)``. Ignored when ``model_id`` is None.
            model_kwargs: Extra keyword arguments forwarded to the model
                constructor. Ignored when ``model_id`` is None.
            **stage_kwargs: Forwarded verbatim to ``stage_cls``'s constructor
                (e.g. ``grammar=``). ``metrics=`` is always injected automatically
                from this pipeline's collector.

        Returns:
            self, for fluent chaining: ``builder.add_stage(...).add_stage(...).build()``.
        """
        if model_id is not None:
            model = self._model_manager.get_model(
                model_id, variant=variant, unit=compute_unit, **(model_kwargs or {})
            )
            stage = stage_cls(model, metrics=self._metrics, **stage_kwargs)
        else:
            stage = stage_cls(metrics=self._metrics, **stage_kwargs)
        self._stages.append(stage)
        self._stage_units.append((stage, compute_unit))
        return self

    def build(self) -> PipelineHandle:
        """Freeze the accumulated stages into an unloaded, registered PipelineHandle.

        Does not load any stage and does not make the pipeline active — call
        ``Moment2Action.load_pipeline(name)`` to do that.

        Returns:
            The newly registered, unloaded ``PipelineHandle``.
        """
        pipeline = Pipeline(self._stages, metrics=self._metrics)
        handle = PipelineHandle(
            name=self._name,
            pipeline=pipeline,
            metrics=self._metrics,
            stage_units=list(self._stage_units),
        )
        self._app._register_pipeline(handle)  # noqa: SLF001
        return handle
