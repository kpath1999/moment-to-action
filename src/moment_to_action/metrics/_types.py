"""Public types for the metrics subsystem.

Dataclasses and enums used by :class:`~moment_to_action.metrics.MetricsCollector`
and surfaced to callers that consume reports.
"""

from __future__ import annotations

import typing as t
from enum import Enum, auto

import attrs

if t.TYPE_CHECKING:
    from datetime import datetime, timedelta


# Pylance decided to be difficult today
def _meta_dict() -> dict[str, t.Any]:
    """Factory function for default metadata dictionary."""
    return {}


def _span_list() -> list[Span]:
    """Factory function for default span list."""
    return []


class SpanType(Enum):
    """Types of spans we might want to track within the pipeline."""

    ### PIPELINE ###

    PIPELINE = auto()
    """End-to-end pipeline execution span."""

    STAGE = auto()
    """Individual stage execution span (e.g. trigger, vision, LLM)."""

    MODEL_INFERENCE = auto()
    """Time taken for a model inference (e.g. vision, LLM)."""


@attrs.define
class Span:
    """Represents a single excution span within the pipeline.

    Contains timing and metadata for that span. Used internally by MetricsCollector.
    """

    id_: int
    """Unique identifier for this span."""

    parent_id: int | None
    """Identifier of the parent span, if any (for nested spans)."""

    type_: SpanType
    """Type of the span (e.g. SENSOR_READ, MODEL_INFERENCE, OTHER)."""

    name: str
    """Name of the span (e.g. "YOLO inference", "camera read")."""

    start: datetime
    """When did this span start?"""

    end: datetime
    """When did this span end?"""

    metadata: dict[str, t.Any] = attrs.Factory(_meta_dict)
    """Arbitrary key/value context for this span."""

    _frozen: bool = attrs.field(default=False, init=False, repr=False)
    """Whether this span is frozen and should not be modified."""

    @property
    def latency(self) -> timedelta:
        """Latency of this span."""
        return self.end - self.start

    @property
    def latency_ms(self) -> float:
        """Latency of this span in milliseconds."""
        return self.latency.total_seconds() * 1000.0

    def summary(self) -> str:
        """Generate a human-readable summary of this span."""
        return (
            f"[{self.type_.name}] {self.name}: {self.latency_ms:.2f}ms (metadata={self.metadata})"
        )

    def __attrs_post_init__(self) -> None:
        self._frozen = True  # Freeze the span after initialization to prevent modifications

    def __setattr__(self, attr: str, value: t.Any) -> None:
        if getattr(self, "_frozen", None):
            msg = (
                f"Span '{self.name}' is frozen and cannot be modified (tried to set {attr}={value})"
            )
            raise AttributeError(msg)
        return super().__setattr__(attr, value)


@attrs.define
class Trace:
    """Represents a single end-to-end pipeline execution trace.

    Contains detailed timing and metadata for each stage, as well as
    overall pipeline events. Used internally by MetricsCollector.
    """

    id_: int
    """Unique identifier for this trace."""

    start: datetime
    """When did this trace start?"""

    end: datetime
    """When did this trace end?"""

    spans: list[Span] = attrs.Factory(_span_list)
    """List of spans recorded for this trace."""

    _frozen: bool = attrs.field(default=False, init=False, repr=False)
    """Whether this trace is frozen and should not be modified."""

    @property
    def latency(self) -> timedelta:
        """Latency of the entire trace."""
        return self.end - self.start

    @property
    def latency_ms(self) -> float:
        """Latency of the entire trace in milliseconds."""
        return self.latency.total_seconds() * 1000.0

    def summary(self, latency_budget: timedelta | None = None) -> str:
        """Generate a human-readable summary of this trace and its spans.

        Idents spans by depth in the trace (based on parent_id relationships) and includes
        latency and metadata for each span.
        """
        # Get trace summary line
        lines = [
            f"Trace {self.id_}: {self.latency_ms:.2f}ms",
            f"Start: {self.start}",
            f"End: {self.end}",
            f"Latency: {self.latency}",
            "Within latency budget: "
            + ("✅" if latency_budget is not None and self.latency <= latency_budget else "❌"),
            "",
        ]

        # Get root span
        root_spans = [span for span in self.spans if span.parent_id is None]
        if len(root_spans) != 1:
            msg = f"Trace {self.id_} has {len(root_spans)} root spans (expected exactly 1)"
            raise ValueError(msg)

        root_span = root_spans[0]

        # Recursively add span summaries with indentation based on depth in the trace
        def add_span_summary(span: Span, indent: int = 0) -> None:
            lines.append("  " * indent + span.summary())
            child_spans = [s for s in self.spans if s.parent_id == span.id_]
            for child in child_spans:
                add_span_summary(child, indent + 2)

        add_span_summary(root_span)

        return "\n".join(lines)

    def __attrs_post_init__(self) -> None:
        self._frozen = True  # Freeze the trace after initialization to prevent modifications

    def __setattr__(self, attr: str, value: t.Any) -> None:
        if getattr(self, "_frozen", None):
            msg = f"Trace {self.id_} is frozen and cannot be modified (tried to set {attr}={value})"
            raise AttributeError(msg)
        return super().__setattr__(attr, value)


@attrs.frozen
class MetricsReport:
    """Represents a report generated from the collected metrics."""

    session_id: str
    """Identifier for the session during which this report was generated."""

    latency_budget: timedelta
    """Latency budget for the pipeline (e.g. 100ms)."""

    traces: list[Trace]
    """List of traces included in this report."""

    slow_traces: list[Trace]
    """List of traces that exceeded the latency budget."""

    def summary(self) -> str:
        """Generate a human-readable summary of this report, including stats and slow traces."""
        lines = [
            f"Metrics Report for session '{self.session_id}'",
            f"Latency budget: {self.latency_budget.total_seconds() * 1000:.2f}ms",
            f"Total traces: {len(self.traces)}",
            f"Slow traces: {len(self.slow_traces)}",
            "",
        ]

        if self.slow_traces:
            lines.append("Slow Traces:")
            for trace in self.slow_traces:
                lines.append(trace.summary())
                lines.append("")  # Add extra newline between traces

        return "\n".join(lines)

    def summary_full(self) -> str:
        """Generate a full human-readable summary of this report, including all traces."""
        lines = [
            f"Metrics Report for session '{self.session_id}'",
            f"Latency budget: {self.latency_budget.total_seconds() * 1000:.2f}ms",
            f"Total traces: {len(self.traces)}",
            f"Slow traces: {len(self.slow_traces)}",
            "",
            "All Traces:",
        ]

        for trace in self.traces:
            lines.append(trace.summary())
            lines.append("")  # Add extra newline between traces

        return "\n".join(lines)
