"""Central metrics collection for the framework.

The MetricsCollector is designed to provide a flexible and extensible way to collect detailed
metrics across the entire pipeline execution.

It allows stages to create spans that are automatically associated with the current trace, enabling
a hierarchical view of the execution. At the end of the pipeline, the MetricsCollector can generate
a comprehensive report that includes latency information for each span and trace, as well as any
relevant metadata. This report can be used to identify bottlenecks and optimize performance across
the pipeline.

The MetricsCollector is implemented as a context manager, allowing for easy integration into the
pipeline stages. Stages can use the `start_span` method to create spans for specific operations, and
these spans will be automatically nested under the current trace. The collector also provides
methods for setting and getting metadata on the current span, allowing stages to add custom
information that can be included in the final report.

Traces are effectively the "root span" of a section of execution, and all spans created during that
execution are children of that trace. This allows for a clear hierarchical structure in the
collected metrics, making it easier to analyze the performance of different stages and identify any
issues.
"""

from __future__ import annotations

import contextlib
import logging
import time
import typing as t
from datetime import UTC, datetime, timedelta

from ._types import MetricsReport, Span, SpanType, Trace

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _unfreeze[T: Span | Trace](obj: T) -> t.Generator[T, None, None]:
    """Context manager to temporarily unfreeze an attrs object for modification."""
    was_frozen = getattr(obj, "_frozen", None)
    if not was_frozen:
        msg = f"Object {obj} is not frozen, cannot unfreeze"
        raise ValueError(msg)

    object.__setattr__(obj, "_frozen", False)
    try:
        yield obj
    finally:
        object.__setattr__(obj, "_frozen", True)


class MetricsCollector:
    """Collects timing, accuracy, and power metrics from across the pipeline."""

    def __init__(
        self,
        session_id: str | None = None,
        latency_budget: timedelta = timedelta(seconds=5),
    ) -> None:
        """Create a new metrics collector.

        Args:
            session_id:
                Session name. If not provided, one will be auto-generated.
            latency_budget:
                Our trace latency budget. Defaults to five seconds.
        """
        self._session_id = session_id or f"session_{int(time.time())}"
        self._latency_budget = latency_budget

        self._current_id = 0

        self._traces: dict[int, Trace] = {}
        self._spans: dict[int, Span] = {}

        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

    def _next_id(self) -> int:
        """Generate a new unique identifier for a trace or span."""
        id_ = self._current_id
        self._current_id += 1
        return id_

    @contextlib.contextmanager
    def start_trace(self) -> t.Generator[Trace, None, None]:
        """Context manager for starting and ending a new trace."""
        # Ensure no trace is currently active
        if self._current_trace is not None:
            msg = "Cannot start a new trace while another is active."
            raise RuntimeError(msg)

        # Create trace with unique ID and start time
        trace = Trace(
            id_=self._next_id(), start=datetime.now(tz=UTC), end=datetime(1970, 1, 1, tzinfo=UTC)
        )

        # Save the trace
        self._traces[trace.id_] = trace
        self._current_trace = trace

        # Give the trace back to the user, and time it
        start_ns = time.perf_counter_ns()
        try:
            yield trace
        finally:
            end_ns = time.perf_counter_ns()

            # ensure span stack is empty
            if self._span_stack:  # pragma: no cover
                msg = "Span stack is not empty at the end of the trace. Missing span ends?"
                raise RuntimeError(msg)

            # Trace is over and we have execution back, end it and clear current
            with _unfreeze(trace):
                trace.end = datetime.now(tz=UTC)
                trace.latency_ns = end_ns - start_ns

            self._current_trace = None

    @contextlib.contextmanager
    def start_span(
        self, type_: SpanType, name: str, metadata: dict[str, t.Any] | None = None
    ) -> t.Generator[Span, None, None]:
        """Context manager for starting and ending a new span within the current trace."""
        if self._current_trace is None:
            msg = "Cannot start a span without an active trace."
            raise RuntimeError(msg)

        # Create span with unique ID and start time
        span = Span(
            id_=self._next_id(),
            parent_id=self._span_stack[-1].id_ if self._span_stack else None,
            type_=type_,
            name=name,
            start=datetime.now(tz=UTC),
            end=datetime(1970, 1, 1, tzinfo=UTC),
            metadata=metadata or {},
        )

        # Save the span and add it to the current trace
        self._spans[span.id_] = span

        self._current_trace.spans.append(span)
        self._span_stack.append(span)

        # Give the span back to the user, and time it
        start_ns = time.perf_counter_ns()
        try:
            yield span
        finally:
            end_ns = time.perf_counter_ns()

            # ensure we are at the end of the current span
            if self._span_stack[-1].id_ != span.id_:  # pragma: no cover
                msg = (
                    "Span stack is out of order. "
                    "Spans must be ended in the reverse order they were started."
                )
                raise RuntimeError(msg)

            # Span is over, end it and pop from stack
            with _unfreeze(span):
                span.end = datetime.now(tz=UTC)
                span.latency_ns = end_ns - start_ns

            self._span_stack.pop()

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    @property
    def latency_budget(self) -> timedelta:
        """Return the latency budget."""
        return self._latency_budget

    @property
    def traces(self) -> list[Trace]:
        """Return the list of recorded traces."""
        return list(self._traces.values())

    @property
    def spans(self) -> list[Span]:
        """Return the list of recorded spans."""
        return list(self._spans.values())

    def get_trace(self, trace_id: int) -> Trace:
        """Get a specific trace by ID."""
        return self._traces[trace_id]

    def get_span(self, span_id: int) -> Span:
        """Get a specific span by ID."""
        return self._spans[span_id]

    @property
    def current_trace(self) -> Trace:
        """Return the currently active trace."""
        if self._current_trace is None:
            msg = "No active trace. Cannot access current trace."
            raise RuntimeError(msg)
        return self._current_trace

    @property
    def current_span(self) -> Span:
        """Return the currently active span."""
        # Ensure we have spans in the stack to return a current span
        if not self._span_stack:
            msg = "No spans have been started. Cannot access last span."
            raise RuntimeError(msg)

        # Check invariants to ensure our internal state is consistent
        if self._current_trace is None:  # pragma: no cover
            msg = "Invariant violation: current_trace should not be None if span_stack is not empty"
            raise AssertionError(msg)

        # Ensure the span has not ended (invariants should guarantee this but we check to be safe)
        if self._span_stack[-1].end != datetime(1970, 1, 1, tzinfo=UTC):  # pragma: no cover
            msg = "Invariant violation: current_span has already ended"
            raise AssertionError(msg)

        return self._span_stack[-1]

    def get_meta(self, key: str) -> t.Any:
        """Get metadata value from the current span."""
        return self.current_span.metadata.get(key)

    def set_meta(self, key: str, value: t.Any) -> None:
        """Set metadata value on the current span."""
        with _unfreeze(self.current_span) as span:
            span.metadata[key] = value

    def report(self) -> MetricsReport:
        """Generate a summary report across all collected traces and spans."""
        return MetricsReport(
            session_id=self.session_id,
            latency_budget=self.latency_budget,
            traces=self.traces,
            slow_traces=[trace for trace in self.traces if trace.latency > self.latency_budget],
        )


class NullMetricsCollector(MetricsCollector):
    """A no-op metrics collector that ignores all spans and traces."""

    def __init__(self) -> None:
        """Initialize a null metrics collector with session_id set to 'null'."""
        super().__init__(session_id="null", latency_budget=timedelta(seconds=5))

    @contextlib.contextmanager
    def start_trace(self) -> t.Generator[Trace, None, None]:
        """No-op trace context manager."""
        yield Trace(id_=0, start=datetime.now(tz=UTC), end=datetime.now(tz=UTC))

    @contextlib.contextmanager
    def start_span(
        self, type_: SpanType, name: str, metadata: dict[str, t.Any] | None = None
    ) -> t.Generator[Span, None, None]:
        """No-op span context manager."""
        yield Span(
            id_=0,
            parent_id=None,
            type_=type_,
            name=name,
            start=datetime.now(tz=UTC),
            end=datetime.now(tz=UTC),
            metadata=metadata or {},
        )

    def get_span(self, span_id: int) -> Span:  # noqa: ARG002
        """Get a span - returns dummy span for null collector."""
        return Span(
            id_=0,
            parent_id=None,
            type_=SpanType.STAGE,
            name="null",
            start=datetime.now(tz=UTC),
            end=datetime.now(tz=UTC),
            metadata={},
        )

    def get_trace(self, trace_id: int) -> Trace:  # noqa: ARG002
        """Get a trace - returns dummy trace for null collector."""
        return Trace(id_=0, start=datetime.now(tz=UTC), end=datetime.now(tz=UTC))

    def report(self) -> MetricsReport:
        """Generate an empty report."""
        return MetricsReport(
            session_id=self.session_id,
            latency_budget=self.latency_budget,
            traces=[],
            slow_traces=[],
        )
