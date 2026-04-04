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
import os
import time
import typing as t
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

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

import attrs
import numpy as np
import psutil

from moment_to_action.metrics._types import (
    CollectorReport,
    EventRecord,
    EventType,
    LatencyBudget,
    PipelineRecord,
    PipelineStats,
    StageRecord,
    LLMRecord,
    StageStats,
    LLMStats,
)

if TYPE_CHECKING:
    from moment_to_action.hardware import ComputeBackend

logger = logging.getLogger(__name__)

_PROCESS = psutil.Process(os.getpid())


def _rss_mb() -> float:
    """Return current process RSS in MB."""
    return _PROCESS.memory_info().rss / 1024**2


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
    """Collects timing, accuracy, and power metrics from across the pipeline.

    Methods are thread-safe, but the collector is not designed for high contention — it's expected
    that spans will be created and ended in a mostly single-threaded manner within the pipeline
    execution, and the lock is primarily to protect against edge cases and ensure consistency of the
    internal state.
    """

    def __init__(
        self,
        compute_backend: ComputeBackend | None = None,
        session_id: str | None = None,
        latency_budget: timedelta = timedelta(seconds=5),
        resource_sample_interval: timedelta = timedelta(seconds=0.1),
    ) -> None:
        """Create a new metrics collector.

        Args:
            compute_backend:
                The compute backend to collect hardware metrics from (power, frequency, etc.).
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
        self._pipeline_log: list[PipelineRecord] = []
        self._stage_log: list[StageRecord] = []
        self._event_log: list[EventRecord] = []        
        self._resource_sample_interval = resource_sample_interval

        # ID tracking
        self._current_id = 0

        # Traces and spans
        self._traces: dict[int, Trace] = {}
        self._spans: dict[int, Span] = {}

        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

        # Sampling stuff
        self._compute_backend = compute_backend
        self._process = psutil.Process()  # current process

        if not self._compute_backend:
            logger.warning("No compute backend provided - hardware resource sampling disabled!")

        # Other state
        self._lock = Lock()  # For thread safety, protects all state (One Big Lock)

    def _sample_resource_usage(self) -> None:
        """Sample memory usage for the current trace, with psutil.

        Must hold lock to do this.
        """
        if self._current_trace is None:  # pragma: no cover
            return  # No active trace, nothing to sample

        if self._compute_backend is None:
            return  # No backend configured; hardware sampling disabled

        proc_mem_info = self._process.memory_info()

        pwr_mon = self._compute_backend.resource_monitor

        sample = ResourceUsageSample(
            timestamp=datetime.now(tz=UTC),
            running_span_id=self._span_stack[-1].id_ if self._span_stack else None,
            cpu_usage=pwr_mon.sample(ComputeUnit.CPU),
            gpu_usage=pwr_mon.sample(ComputeUnit.GPU),
            npu_usage=pwr_mon.sample(ComputeUnit.NPU),
            dsp_usage=pwr_mon.sample(ComputeUnit.DSP),
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
                start=datetime.now(tz=UTC),
                end=datetime(1970, 1, 1, tzinfo=UTC),
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
                    trace.end = datetime.now(tz=UTC)
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
                    span.end = datetime.now(tz=UTC)
                    span.latency_ns = end_ns - start_ns

                self._span_stack.pop()

        # Snapshot RSS at collector creation — used as the baseline
        self._baseline_rss_mb: float = _rss_mb()

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    @property
    def latency_budget(self) -> timedelta:
        """Return the latency budget."""
        return self._latency_budget

    def baseline_rss_mb(self) -> float:
        """RSS at pipeline startup — before any models loaded."""
        return self._baseline_rss_mb

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

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
            if self._span_stack[-1].end != datetime(1970, 1, 1, tzinfo=UTC):  # pragma: no cover
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

    def log_stage(
        self,
        stage_name: str,
        stage_idx: int,
        latency_ms: float,
        init_memory_bytes: int = 0,
        runtime_memory_bytes: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Record a single stage execution."""
        self._stage_log.append(
            StageRecord(
                timestamp=time.time(),
                stage_name=stage_name,
                stage_idx=stage_idx,
                latency_ms=latency_ms,
                init_memory_bytes=init_memory_bytes,
                runtime_memory_bytes=runtime_memory_bytes,
                metadata=metadata or {},
            )
        )

    # Adding a method to keep track of the llm data
    def log_llm(
        self,
        stage_name: str,
        stage_idx: int,
        latency_ms: float,
        init_memory_bytes: int = 0,
        runtime_memory_bytes: int = 0,
        prompt_ms: float = 0.0,
        gen_ms: float = 0.0,
        prompt_tokens: int = 0,
        gen_tokens: int = 0,
        kv_cache_used: int = 0,
        kv_cache_total: int = 0,
        server_rss_bytes: int = 0,
        metadata: dict | None = None,
    ) -> None:
        """Record a single LLMStage execution. Goes into the same _stage_log."""
        self._stage_log.append(
            LLMRecord(
                timestamp=time.time(),
                stage_name=stage_name,
                stage_idx=stage_idx,
                latency_ms=latency_ms,
                init_memory_bytes=init_memory_bytes,
                runtime_memory_bytes=runtime_memory_bytes,
                prompt_ms=prompt_ms,
                gen_ms=gen_ms,
                prompt_tokens=prompt_tokens,
                gen_tokens=gen_tokens,
                tokens_per_second=gen_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0,
                kv_cache_used_tokens=kv_cache_used,
                kv_cache_total_tokens=kv_cache_total,
                kv_cache_ratio=kv_cache_used / kv_cache_total if kv_cache_total else 0.0,
                server_rss_bytes=server_rss_bytes,
                metadata=metadata or {},
            )
        )

    '''
    def snapshot_memory(self, label: str) -> float:
        """Record a named RSS snapshot and return the value in MB.

        Useful for bracketing model load:
            before = metrics.snapshot_memory("before_llm_load")
            self.llm = Llama(...)
            after  = metrics.snapshot_memory("after_llm_load")
        """
        rss = _rss_mb()
        self.log_event("memory_snapshot", {"label": label, "rss_mb": round(rss, 2)})
        logger.info("Memory [%s]: %.1f MB RSS", label, rss)
        return rss
    '''

    def _per_stage_stats(self) -> dict[str, StageStats]:
        """Compute per-stage latency statistics from the stage log."""
        if not self._stage_log:
            return {}

        """This records the latencies and groups them by stages for a number of pipeline runs"""
        """
        by_stage: dict[str, list[float]] = {}
        for record in self._stage_log:
            by_stage.setdefault(record.stage_name, []).append(record.latency_ms)        
        """

        """Modifying the data structure to included stats in addition to latency, others can be easily added"""
        by_stage: dict[str, list[StageRecord]] = {}
        for record in self._stage_log:
            by_stage.setdefault(record.stage_name, []).append(record)

        return {
            stage_name: self._compute_stage_stats(records)
            for stage_name, records in by_stage.items()
        }

    def _compute_stage_stats(self, records: list[StageRecord]) -> StageStats:
        latencies = np.array([r.latency_ms for r in records])
        base = dict(
            num_calls=len(records),
            mean_ms=float(np.mean(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            min_ms=float(np.min(latencies)),
            max_ms=float(np.max(latencies)),
            init_memory_bytes=records[0].init_memory_bytes,
            mean_runtime_memory_bytes=int(np.mean([r.runtime_memory_bytes for r in records])),
        )

        if isinstance(records[0], LLMRecord):
            return self._compute_llm_stats(base, records)  # type: ignore[arg-type]
        return StageStats(**base)

    def _compute_llm_stats(self, base: dict, records: list[LLMRecord]) -> LLMStats:
        gen_arr = np.array([r.gen_ms for r in records])
        return LLMStats(
            **base,
            mean_prompt_ms=float(np.mean([r.prompt_ms for r in records])),
            mean_gen_ms=float(np.mean(gen_arr)),
            p95_gen_ms=float(np.percentile(gen_arr, 95)),
            mean_tokens_per_second=float(np.mean([r.tokens_per_second for r in records])),
            # mean_kv_cache_ratio=float(np.mean([r.kv_cache_ratio for r in records])),
            mean_kv_cache_ratio=float(np.mean([r.kv_cache_total_tokens for r in records])),
            peak_kv_cache_ratio=float(np.max([r.kv_cache_ratio for r in records])),
            mean_server_rss_bytes=int(np.mean([r.server_rss_bytes for r in records])),
            peak_server_rss_bytes=int(np.max([r.server_rss_bytes for r in records])),
        )
        """

        result: dict[str, StageStats] = {}

        for stage_name, records in by_stage.items():
            latencies = np.array([r.latency_ms for r in records])

            base = dict(
                num_calls=len(records),
                mean_ms=float(np.mean(latencies)),
                p50_ms=float(np.percentile(latencies, 50)),
                p95_ms=float(np.percentile(latencies, 95)),
                min_ms=float(np.min(latencies)),
                max_ms=float(np.max(latencies)),
                init_memory_bytes=records[0].init_memory_bytes,
                mean_runtime_memory_bytes=int(
                    np.mean([r.runtime_memory_bytes for r in records])
                    ),
                )

            result[stage_name] = StageStats(**base)

        return result
        """

        """
        return {
            stage: StageStats(
                num_calls=len(latencies),
                mean_ms=float(np.mean(arr := np.array(latencies))),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
                init_memory_bytes=0,
                mean_runtime_memory_bytes=0,
            )
            for stage, latencies in by_stage.items()
        }
        """

    def _pipeline_stats(self) -> PipelineStats:
        triggers = [r for r in self._pipeline_log if r.event_type == EventType.TRIGGER_FIRED]
        detections = [r for r in self._pipeline_log if r.event_type == EventType.DETECTION]
        false_positives = [
            r for r in self._pipeline_log if r.event_type == EventType.FALSE_POSITIVE
        ]
        return PipelineStats(
            total_triggers=len(triggers),
            total_detections=len(detections),
            total_false_positives=len(false_positives),
            trigger_rate=len(triggers) / max(1, len(self._pipeline_log)),
            false_positive_rate=len(false_positives) / max(1, len(detections)),
        )

    def _latency_budget_analysis(self) -> LatencyBudget:
        """Compute latency budget against end-to-end pipeline event times."""
        total_mean = (
            float(np.mean([r.latency_ms for r in self._pipeline_log]))
            if self._pipeline_log
            else 0.0
        )
        return LatencyBudget(
            total_mean_ms=total_mean,
            budget_ms=self._latency_budget_ms,
            headroom_ms=self._latency_budget_ms - total_mean,
            within_budget=total_mean < self._latency_budget_ms,
        )

    # ------------------------------------------------------------------
    # Memory reporting
    # ------------------------------------------------------------------
    ##Commenting out the code for now, may be removed or restructred
    """

    def _memory_records(self) -> list[dict]:
        #Extract memory snapshots from the event log.
        return [
            e.data for e in self._event_log
            if e.event_type == "memory_snapshot"
        ]

    def _stage_memory_stats(self) -> dict[str, dict]:
        #Compute per-stage peak RSS and max delta from stage log metadata.
        by_stage: dict[str, list[dict]] = {}
        for record in self._stage_log:
            if "mem_after_mb" in record.metadata:
                by_stage.setdefault(record.stage_name, []).append(record.metadata)

        result = {}
        for stage, metas in by_stage.items():
            after_values = [m["mem_after_mb"] for m in metas]
            delta_values = [m["mem_delta_mb"] for m in metas]
            result[stage] = {
                "peak_rss_mb":   round(max(after_values), 1),
                "mean_rss_mb":   round(float(np.mean(after_values)), 1),
                "max_delta_mb":  round(max(delta_values, key=abs), 1),
                "mean_delta_mb": round(float(np.mean(delta_values)), 1),
            }
        return result

    def print_memory_report(self) -> None:
        #Print a memory usage table alongside latency.
        logger.info("Entering print_memory_report")
        mem_stats = self._stage_memory_stats()
        snapshots = self._memory_records()

        logger.info("\n%-25s %12s %12s %12s", "Stage", "Peak RSS", "Mean RSS", "Max Δ")
        logger.info("─" * 65)
        for stage, stats in mem_stats.items():
            logger.info(
                "  %-23s %10.1fMB %10.1fMB %+10.1fMB",
                stage,
                stats["peak_rss_mb"],
                stats["mean_rss_mb"],
                stats["max_delta_mb"],
            )

        if snapshots:
            logger.info("\nNamed snapshots:")
            for snap in snapshots:
                logger.info("  %-30s %.1f MB", snap["label"], snap["rss_mb"])

        logger.info("Printing pipeline memory usage")
        current = _rss_mb()
        logger.info("\n  Baseline RSS:  %.1f MB", self._baseline_rss_mb)
        logger.info("  Current RSS:   %.1f MB", current)
        logger.info("  Total growth:  %+.1f MB", current - self._baseline_rss_mb)
    """

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def print_stage_latencies(self) -> None:
        """Print latency table for the most recent pipeline run."""
        if not self._stage_log:
            logger.info("No stage latencies recorded.")
            return
        total = sum(r.latency_ms for r in self._stage_log)
        logger.info("\n%-25s %10s", "Stage", "Latency")
        logger.info("─" * 37)
        for r in self._stage_log:
            logger.info("  %-23s %8.1fms", r.stage_name, r.latency_ms)
        logger.info("─" * 37)
        logger.info("  %-23s %8.1fms", "Total", total)

    def save(self, path: str) -> None:
        """Save full report to JSON."""
        Path(path).write_text(json.dumps(attrs.asdict(self.report()), indent=2))
        logger.info("Metrics saved to %s", path)

    def print_summary(self) -> None:
        """Log a human-readable summary."""
        r = self.report()
        logger.info("\n%s", "=" * 50)
        logger.info("METRICS SUMMARY  |  session: %s", r.session_id)
        logger.info("=" * 50)
        logger.info("Total stages: %d", r.total_stages)
        logger.info("\nPer-stage latency:")
        for stage, stats in r.per_stage.items():
            logger.info(
                "  %-20s  mean=%.1fms  p95=%.1fms mean_init_memory=%.1fMB mean_runtime_memory=%.1fMB",
                stage,
                stats.mean_ms,
                stats.p95_ms,
                stats.init_memory_bytes,
                stats.mean_runtime_memory_bytes,
            )

        if isinstance(stats, LLMStats):
            logger.info(
                "  %-20s  prompt=%.1fms  gen=%.1fms  p95_gen=%.1fms",
                "",
                stats.mean_prompt_ms,
                stats.mean_gen_ms,
                stats.p95_gen_ms,
            )
            logger.info(
                "  %-20s  tok/s=%.1f  kv_mean=%.2f  kv_peak=%.2f",
                "",
                stats.mean_tokens_per_second,
                stats.mean_kv_cache_ratio,
                stats.peak_kv_cache_ratio,
            )
            logger.info(
                "  %-20s  server_rss_mean=%.1fMB  server_rss_peak=%.1fMB",
                "",
                stats.mean_server_rss_bytes / 1024**2,
                stats.peak_server_rss_bytes / 1024**2,
            )

        logger.info("\nPer-stage memory:")
        # self.print_memory_report()

        budget = r.latency_budget
        logger.info("\nLatency budget (target <%.0fms):", budget.budget_ms)
        status = "✓ within budget" if budget.within_budget else "✗ over budget"
        logger.info("  Total:   %.1fms  (%s)", budget.total_mean_ms, status)
        logger.info("=" * 50)

class NullMetricsCollector(MetricsCollector):
    """A no-op metrics collector that ignores all spans and traces."""

    def __init__(self, compute_backend: ComputeBackend | None = None) -> None:  # noqa: ARG002
        """Initialize a null metrics collector.

        The compute_backend is accepted for API compatibility but is never used —
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

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> MetricsReport:
        """Generate an empty report."""
        return MetricsReport(
            session_id=self.session_id,
            latency_budget=self.latency_budget,
            traces=[],
            slow_traces=[],
        )
