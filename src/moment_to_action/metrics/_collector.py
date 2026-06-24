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
import copy
import logging
import time
import typing as t
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, TypeVar

import psutil

from moment_to_action.hardware._types import ComputeUnit

from ._types import (
    MemoryUsageSample,
    MetricsReport,
    ResourceUsageSample,
    Span,
    SpanType,
    Trace,
)

if TYPE_CHECKING:
    from moment_to_action.hardware import Platform

logger = logging.getLogger(__name__)

_T = TypeVar("_T", "Span", "Trace")


@contextlib.contextmanager
def _unfreeze(obj: _T) -> t.Generator[_T, None, None]:
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
    """Collects timing, accuracy, and power metrics from across the pipeline.

    Methods are thread-safe, but the collector is not designed for high contention — it's expected
    that spans will be created and ended in a mostly single-threaded manner within the pipeline
    execution, and the lock is primarily to protect against edge cases and ensure consistency of the
    internal state.
    """

    def __init__(
        self,
        compute_platform: Platform | None = None,
        session_id: str | None = None,
        latency_budget: timedelta = timedelta(seconds=5),
        resource_sample_interval: timedelta = timedelta(seconds=0.1),
    ) -> None:
        """Create a new metrics collector.

        Args:
            compute_platform:
                The compute platform to collect hardware metrics from (power, frequency, etc.).
                If None, hardware resource sampling is disabled but all timing metrics still work.
            session_id:
                Session name. If not provided, one will be auto-generated.
            latency_budget:
                Our trace latency budget. Defaults to five seconds.
            resource_sample_interval:
                How often to sample resource usage for spans. Defaults to 100ms.
        """
        # Configuration
        # all this is const and unprotected by lock
        self._session_id = session_id or f"session_{int(time.time())}"
        self._latency_budget = latency_budget
        self._resource_sample_interval = resource_sample_interval

        # ID tracking
        self._current_id = 0

        # Traces and spans
        self._traces: dict[int, Trace] = {}
        self._spans: dict[int, Span] = {}

        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

        # Sampling stuff
        self._compute_platform = compute_platform
        self._process = psutil.Process()  # current process

        if not self._compute_platform:
            logger.warning("No compute platform provided - hardware resource sampling disabled!")

        # Other state
        self._lock = Lock()  # For thread safety, protects all state (One Big Lock)

    def _sample_resource_usage(self) -> None:
        """Sample memory usage for the current trace, with psutil.

        Must hold lock to do this.
        """
        if self._current_trace is None:  # pragma: no cover
            return  # No active trace, nothing to sample

        if self._compute_platform is None:
            return  # No platform configured; hardware sampling disabled

        proc_mem_info = self._process.memory_info()

        pwr_mon = self._compute_platform.resource_monitor

        sample = ResourceUsageSample(
            timestamp=datetime.now(tz=timezone.utc),
            running_span_id=self._span_stack[-1].id_ if self._span_stack else None,
            cpu_usage=pwr_mon.sample(ComputeUnit.CPU),
            gpu_usage=pwr_mon.sample(ComputeUnit.GPU),
            npu_usage=pwr_mon.sample(ComputeUnit.NPU),
            proc_cpu_usage=self._process.cpu_percent(interval=None),
            mem_usage=MemoryUsageSample(
                rss_bytes=proc_mem_info.rss,
                vms_bytes=proc_mem_info.vms,
                shared_bytes=proc_mem_info.shared,
                text_bytes=proc_mem_info.text,
                lib_bytes=proc_mem_info.lib,
                data_bytes=proc_mem_info.data,
                dirty_bytes=proc_mem_info.dirty,
            ),
        )
        self._current_trace.resource_usage_samples.append(sample)

    def _start_resource_sampling_thread(self, evt: Event) -> Thread:
        """Start a background thread to sample memory usage at regular intervals.

        Thread safe.
        """

        def sampling_loop() -> None:
            while not evt.is_set():
                with self._lock:
                    self._sample_resource_usage()
                time.sleep(self._resource_sample_interval.total_seconds())

        thread = Thread(target=sampling_loop)
        thread.start()
        return thread

    def _next_id(self) -> int:
        """Generate a new unique identifier for a trace or span.

        Must already hold lock.
        """
        id_ = self._current_id
        self._current_id += 1
        return id_

    @contextlib.contextmanager
    def start_trace(self) -> t.Generator[Trace, None, None]:
        """Context manager for starting and ending a new trace."""
        with self._lock:
            # Ensure no trace is currently active
            if self._current_trace is not None:
                msg = "Cannot start a new trace while another is active."
                raise RuntimeError(msg)

            # Create trace with unique ID and start time
            trace = Trace(
                id_=self._next_id(),
                start=datetime.now(tz=timezone.utc),
                end=datetime(1970, 1, 1, tzinfo=timezone.utc),
            )

            # Save the trace
            self._traces[trace.id_] = trace
            self._current_trace = trace

        # Start hardware sampling thread
        # this will sample mem usage right away
        sampling_thread_event = Event()
        sampling_thread = self._start_resource_sampling_thread(sampling_thread_event)

        # Give the trace back to the user, and time it
        start_ns = time.perf_counter_ns()
        try:
            yield trace
        finally:
            # Get end time
            end_ns = time.perf_counter_ns()

            # Stop the sampling thread and wait for it to finish
            sampling_thread_event.set()
            sampling_thread.join()

            with self._lock:
                # ensure span stack is empty
                if self._span_stack:  # pragma: no cover
                    msg = "Span stack is not empty at the end of the trace. Missing span ends?"
                    raise RuntimeError(msg)

                # Trace is over and we have execution back, end it and clear current
                with _unfreeze(trace):
                    trace.end = datetime.now(tz=timezone.utc)
                    trace.latency_ns = end_ns - start_ns

                self._current_trace = None

    @contextlib.contextmanager
    def start_span(
        self, type_: SpanType, name: str, metadata: dict[str, t.Any] | None = None
    ) -> t.Generator[Span, None, None]:
        """Context manager for starting and ending a new span within the current trace."""
        with self._lock:
            if self._current_trace is None:
                msg = "Cannot start a span without an active trace."
                raise RuntimeError(msg)

            # Create span with unique ID and start time
            span = Span(
                id_=self._next_id(),
                parent_id=self._span_stack[-1].id_ if self._span_stack else None,
                type_=type_,
                name=name,
                start=datetime.now(tz=timezone.utc),
                end=datetime(1970, 1, 1, tzinfo=timezone.utc),
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

            with self._lock:
                # ensure we are at the end of the current span
                if self._span_stack[-1].id_ != span.id_:  # pragma: no cover
                    msg = (
                        "Span stack is out of order. "
                        "Spans must be ended in the reverse order they were started."
                    )
                    raise RuntimeError(msg)

                # Span is over, end it and pop from stack
                with _unfreeze(span):
                    span.end = datetime.now(tz=timezone.utc)
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
        with self._lock:
            return list(self._traces.values())

    @property
    def spans(self) -> list[Span]:
        """Return the list of recorded spans."""
        with self._lock:
            return list(self._spans.values())

    def get_trace(self, trace_id: int) -> Trace:
        """Get a specific trace by ID."""
        with self._lock:
            return self._traces[trace_id]

    def get_span(self, span_id: int) -> Span:
        """Get a specific span by ID."""
        with self._lock:
            return self._spans[span_id]

    @property
    def current_trace(self) -> Trace:
        """Return the currently active trace."""
        with self._lock:
            if self._current_trace is None:
                msg = "No active trace. Cannot access current trace."
                raise RuntimeError(msg)
            return self._current_trace

    @property
    def current_span(self) -> Span:
        """Return the currently active span."""
        with self._lock:
            # Ensure we have spans in the stack to return a current span
            if not self._span_stack:
                msg = "No spans have been started. Cannot access last span."
                raise RuntimeError(msg)

            # Check invariants to ensure our internal state is consistent
            if self._current_trace is None:  # pragma: no cover
                msg = "Invariant violation: current_trace should exist if span_stack is not empty"
                raise AssertionError(msg)

            # Ensure the span has not ended (invariants should guarantee this)
            if self._span_stack[-1].end != datetime(
                1970, 1, 1, tzinfo=timezone.utc
            ):  # pragma: no cover
                msg = "Invariant violation: current_span has already ended"
                raise AssertionError(msg)

            return self._span_stack[-1]

    def get_meta(self, key: str) -> t.Any:
        """Get metadata value from the current span."""
        with self._lock:
            # Access _span_stack directly to avoid recursive lock acquisition
            if not self._span_stack:
                msg = "No spans have been started. Cannot access last span."
                raise RuntimeError(msg)
            return self._span_stack[-1].metadata.get(key)

    def set_meta(self, key: str, value: t.Any) -> None:
        """Set metadata value on the current span."""
        with self._lock:
            # Access _span_stack directly to avoid recursive lock acquisition
            if not self._span_stack:
                msg = "No spans have been started. Cannot access last span."
                raise RuntimeError(msg)
            with _unfreeze(self._span_stack[-1]) as span:
                span.metadata[key] = value

    def report(self) -> MetricsReport:
        """Generate a summary report across all collected traces and spans."""
        with self._lock:
            # Access _traces directly (not via self.traces) to avoid recursive lock acquisition
            traces = copy.deepcopy(list(self._traces.values()))
            return MetricsReport(
                session_id=self.session_id,
                latency_budget=self.latency_budget,
                traces=traces,
                slow_traces=[trace for trace in traces if trace.latency > self.latency_budget],
            )


class NullMetricsCollector(MetricsCollector):
    """A no-op metrics collector that ignores all spans and traces."""

    def __init__(self, compute_platform: Platform | None = None) -> None:  # noqa: ARG002
        """Initialize a null metrics collector.

        The compute_platform is accepted for API compatibility but is never used —
        NullMetricsCollector is a no-op that never samples hardware metrics.
        """
        # Bypass MetricsCollector.__init__ entirely; we only need the minimal
        # state that report() and the properties access.
        self._session_id = "null"
        self._latency_budget = timedelta(seconds=5)
        self._traces: dict[int, Trace] = {}
        self._spans: dict[int, Span] = {}
        self._span_stack: list[Span] = []
        self._current_trace: Trace | None = None
        self._lock = Lock()

    @contextlib.contextmanager
    def start_trace(self) -> t.Generator[Trace, None, None]:
        """No-op trace context manager."""
        yield Trace(id_=0, start=datetime.now(tz=timezone.utc), end=datetime.now(tz=timezone.utc))

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
            start=datetime.now(tz=timezone.utc),
            end=datetime.now(tz=timezone.utc),
            metadata=metadata or {},
        )

    def get_span(self, span_id: int) -> Span:  # noqa: ARG002
        """Get a span - returns dummy span for null collector."""
        return Span(
            id_=0,
            parent_id=None,
            type_=SpanType.STAGE,
            name="null",
            start=datetime.now(tz=timezone.utc),
            end=datetime.now(tz=timezone.utc),
            metadata={},
        )

    def get_trace(self, trace_id: int) -> Trace:  # noqa: ARG002
        """Get a trace - returns dummy trace for null collector."""
        return Trace(id_=0, start=datetime.now(tz=timezone.utc), end=datetime.now(tz=timezone.utc))

    def report(self) -> MetricsReport:
        """Generate an empty report."""
        return MetricsReport(
            session_id=self.session_id,
            latency_budget=self.latency_budget,
            traces=[],
            slow_traces=[],
        )
