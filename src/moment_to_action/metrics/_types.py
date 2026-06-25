"""Public types for the metrics subsystem.

Dataclasses and enums used by :class:`~moment_to_action.metrics.MetricsCollector`
and surfaced to callers that consume reports.
"""

from __future__ import annotations

import typing as t
from datetime import datetime, timedelta
from enum import Enum, auto

import attrs
import numpy as np

if t.TYPE_CHECKING:  # pragma: no cover
    from moment_to_action.hardware._metrics import InferenceMetrics
    from moment_to_action.hardware._types import ComputeUnitUsageSample


# Pylance decided to be difficult today
def _meta_dict() -> dict[str, t.Any]:
    """Factory function for default metadata dictionary."""
    return {}


def _span_list() -> list[Span]:
    """Factory function for default span list."""
    return []


def _resource_usage_sample_list() -> list[ResourceUsageSample]:
    """Factory function for default resource usage sample list."""
    return []


class SpanType(Enum):
    """Types of spans we might want to track within the pipeline."""

    ### PIPELINE ###

    PIPELINE = auto()
    """End-to-end pipeline execution span."""

    STAGE = auto()
    """Individual stage execution span (e.g. trigger, vision, LLM)."""

    MODEL_PREPROCESS = auto()
    """Time taken for preprocessing steps before a model inference (e.g. resampling audio)."""

    MODEL_INFERENCE = auto()
    """Time taken for a model inference (e.g. vision, LLM)."""

    MODEL_POST_PROCESS = auto()
    """Time taken for post-processing steps after a model inference (e.g. NMS, decoding)."""

    MODEL_LOAD = auto()
    """Time taken to load model weights onto a backend."""

    MODEL_UNLOAD = auto()
    """Time taken to release model weights from a backend."""


@attrs.frozen
class MemoryUsageSample:
    """Represents a single process memory usage sample taken during a trace execution."""

    rss_bytes: int
    """Resident Set Size (RSS) in bytes."""

    vms_bytes: int
    """Virtual Memory Size (VMS) in bytes."""

    shared_bytes: int
    """Shared memory size in bytes."""

    text_bytes: int
    """Text (code) memory size in bytes."""

    lib_bytes: int
    """Library memory size in bytes."""

    data_bytes: int
    """Data memory size in bytes."""

    dirty_bytes: int
    """Dirty memory size in bytes."""

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this memory usage sample."""
        return attrs.asdict(self)


@attrs.frozen
class ResourceUsageSample:
    """Represents a single resource usage sample taken during a trace execution."""

    timestamp: datetime
    """When was this resource usage sample taken?"""

    running_span_id: int | None
    """The ID of the span that was running when this resource usage sample was taken, if any."""

    cpu_usage: ComputeUnitUsageSample
    """CPU usage sample at the time of this resource usage sample."""

    gpu_usage: ComputeUnitUsageSample
    """GPU usage sample at the time of this resource usage sample."""

    npu_usage: ComputeUnitUsageSample
    """NPU usage sample at the time of this resource usage sample."""

    proc_cpu_usage: float
    """Process-specific CPU usage percentage at the time of this resource usage sample."""

    mem_usage: MemoryUsageSample
    """Process memory usage sample at the time of this resource usage sample."""

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this resource usage sample."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "running_span_id": self.running_span_id,
            "cpu_usage": self.cpu_usage.json(),
            "gpu_usage": self.gpu_usage.json(),
            "npu_usage": self.npu_usage.json(),
            "proc_cpu_usage": self.proc_cpu_usage,
            "mem_usage": self.mem_usage.json(),
        }


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

    latency_ns: int = -1
    """Latency of this span in nanoseconds, as measured by time.perf_counter_ns().

    Used for more accurate latency measurement than start/end datetimes.
    May not match the latency calculated from ``end - start``, as those clocks are not monotonic.
    """

    metadata: dict[str, t.Any] = attrs.Factory(_meta_dict)
    """Arbitrary key/value context for this span."""

    inference_metrics: InferenceMetrics | None = attrs.field(default=None)
    """Typed inference metrics attached by the model layer after inference completes.

    Set on ``MODEL_INFERENCE`` spans by model implementations that expose
    backend timing data (e.g. llama.cpp ``timings``).  ``None`` for spans
    that do not represent model inference or whose backend does not expose metrics.
    """

    _frozen: bool = attrs.field(default=False, init=False, repr=False)
    """Whether this span is frozen and should not be modified."""

    _POST_FREEZE_MUTABLE: t.ClassVar[frozenset[str]] = frozenset({"inference_metrics"})

    @property
    def latency(self) -> timedelta:
        """Latency of this span."""
        return timedelta(microseconds=self.latency_ns / 1000)

    @property
    def latency_ms(self) -> float:
        """Latency of this span in milliseconds."""
        return self.latency_ns / 1_000_000

    def summary(self) -> str:
        """Generate a human-readable summary of this span."""
        base = (
            f"[{self.type_.name}] {self.name}: {self.latency_ms:.2f}ms (metadata={self.metadata})"
        )
        if self.inference_metrics is not None:
            m = self.inference_metrics
            base += (
                f" [llm: {m.predicted_n} tok @ {m.predicted_per_second:.1f} tok/s, "
                f"prompt {m.prompt_n} tok @ {m.prompt_per_second:.1f} tok/s]"
            )
        return base

    def summary_rich(self) -> str:
        """Generate a human-readable summary of this span, with rich formatting."""
        base = (
            f"[[cyan]{self.type_.name}[/cyan]] [bold]{self.name}[/bold]: "
            f"[green]{self.latency_ms:.2f}ms[/green] (metadata={self.metadata})"
        )
        if self.inference_metrics is not None:
            m = self.inference_metrics
            base += (
                f" [[dim]llm: {m.predicted_n} tok @ {m.predicted_per_second:.1f} tok/s, "
                f"prompt {m.prompt_n} tok @ {m.prompt_per_second:.1f} tok/s[/dim]]"
            )
        return base

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this span."""
        return {
            "id": self.id_,
            "parent_id": self.parent_id,
            "type": self.type_.name,
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "latency_ns": self.latency_ns,
            "metadata": self.metadata,
            "inference_metrics": (
                self.inference_metrics.model_dump() if self.inference_metrics is not None else None
            ),
        }

    def __attrs_post_init__(self) -> None:
        self._frozen = True  # Freeze the span after initialization to prevent modifications

    def __setattr__(self, attr: str, value: t.Any) -> None:
        if getattr(self, "_frozen", None) and attr not in self._POST_FREEZE_MUTABLE:
            msg = (
                f"Span '{self.name}' is frozen and cannot be modified (tried to set {attr}={value})"
            )
            raise AttributeError(msg)
        return super().__setattr__(attr, value)


@attrs.define
class Trace:
    """Represents a single execution trace, with 1+ pipeline inferences.

    Contains detailed timing and metadata for each stage, as well as
    overall pipeline events. Used internally by MetricsCollector.
    """

    id_: int
    """Unique identifier for this trace."""

    start: datetime
    """When did this trace start?"""

    end: datetime
    """When did this trace end?"""

    latency_ns: int = -1
    """Latency of the entire trace in nanoseconds, as measured by time.perf_counter_ns().

    Used for more accurate latency measurement than start/end datetimes.
    May not match the latency calculated from ``end - start``, as those clocks are not monotonic.
    """

    spans: list[Span] = attrs.Factory(_span_list)
    """List of spans recorded for this trace."""

    resource_usage_samples: list[ResourceUsageSample] = attrs.Factory(_resource_usage_sample_list)
    """List of resource usage samples recorded for this trace."""

    _frozen: bool = attrs.field(default=False, init=False, repr=False)
    """Whether this trace is frozen and should not be modified."""

    @property
    def latency(self) -> timedelta:
        """Latency of the entire trace."""
        return timedelta(microseconds=self.latency_ns / 1000)

    @property
    def latency_ms(self) -> float:
        """Latency of the entire trace in milliseconds."""
        return self.latency_ns / 1_000_000

    def _build_summary(self, *, rich: bool, latency_budget: timedelta | None = None) -> str:
        header = (
            f"[bold]Trace {self.id_}[/bold]: [green]{self.latency_ms:.2f}ms[/green]"
            if rich
            else f"Trace {self.id_}: {self.latency_ms:.2f}ms"
        )
        lines = [
            header,
            f"Start: {self.start}",
            f"End: {self.end}",
            f"Latency: {self.latency_ms:.2f}ms",
            "Within latency budget: "
            + (
                "N/A"
                if latency_budget is None
                else ("✅" if self.latency <= latency_budget else "❌")
            ),
            "",
        ]

        # Spans without parents — the trace is their implicit parent
        root_spans = [span for span in self.spans if span.parent_id is None]

        def add_span_summary(span: Span, indent: int = 0) -> None:
            lines.append("  " * indent + (span.summary_rich() if rich else span.summary()))
            for child in [s for s in self.spans if s.parent_id == span.id_]:
                add_span_summary(child, indent + 2)

        for root_span in root_spans:
            add_span_summary(root_span)

        if self.resource_usage_samples:
            lines.append("")
            lines.append(self._resource_usage_summary(rich=rich))

        return "\n".join(lines)

    def _resource_usage_summary(self, *, rich: bool) -> str:
        """Build an aggregate resource-usage block from all samples in this trace."""
        samples = self.resource_usage_samples
        n = len(samples)

        cpu = np.array([s.cpu_usage.usage_pct for s in samples])
        gpu = np.array([s.gpu_usage.usage_pct for s in samples])
        npu = np.array([s.npu_usage.usage_pct for s in samples])
        proc = np.array([s.proc_cpu_usage for s in samples])
        rss = np.array([s.mem_usage.rss_bytes for s in samples])

        avg_cpu, peak_cpu = float(cpu.mean()), float(cpu.max())
        avg_gpu, peak_gpu = float(gpu.mean()), float(gpu.max())
        avg_npu, peak_npu = float(npu.mean()), float(npu.max())
        avg_proc_cpu, peak_proc_cpu = float(proc.mean()), float(proc.max())
        avg_rss_mb = float(rss.mean()) / 1024 / 1024
        peak_rss_mb = float(rss.max()) / 1024 / 1024

        if rich:
            header = f"[bold]Resource usage[/bold] ({n} samples)"
            rows = [
                (
                    f"  CPU:      avg [green]{avg_cpu:.1f}%[/green]"
                    f"  peak [yellow]{peak_cpu:.1f}%[/yellow]"
                ),
                (
                    f"  GPU:      avg [green]{avg_gpu:.1f}%[/green]"
                    f"  peak [yellow]{peak_gpu:.1f}%[/yellow]"
                ),
                (
                    f"  NPU:      avg [green]{avg_npu:.1f}%[/green]"
                    f"  peak [yellow]{peak_npu:.1f}%[/yellow]"
                ),
                (
                    f"  proc CPU: avg [green]{avg_proc_cpu:.1f}%[/green]"
                    f"  peak [yellow]{peak_proc_cpu:.1f}%[/yellow]"
                ),
                (
                    f"  RSS:      avg [green]{avg_rss_mb:.1f} MB[/green]"
                    f"  peak [cyan]{peak_rss_mb:.1f} MB[/cyan]"
                ),
            ]
        else:
            header = f"Resource usage ({n} samples)"
            rows = [
                (f"  CPU:      avg {avg_cpu:.1f}%  peak {peak_cpu:.1f}%"),
                (f"  GPU:      avg {avg_gpu:.1f}%  peak {peak_gpu:.1f}%"),
                (f"  NPU:      avg {avg_npu:.1f}%  peak {peak_npu:.1f}%"),
                (f"  proc CPU: avg {avg_proc_cpu:.1f}%  peak {peak_proc_cpu:.1f}%"),
                (f"  RSS:      avg {avg_rss_mb:.1f} MB  peak {peak_rss_mb:.1f} MB"),
            ]

        return "\n".join([header, *rows])

    def summary(self, latency_budget: timedelta | None = None) -> str:
        """Generate a human-readable summary of this trace and its spans.

        Idents spans by depth in the trace (based on parent_id relationships) and includes
        latency and metadata for each span.
        """
        return self._build_summary(rich=False, latency_budget=latency_budget)

    def summary_rich(self, latency_budget: timedelta | None = None) -> str:
        """Generate a human-readable summary of this trace and its spans, with rich formatting."""
        return self._build_summary(rich=True, latency_budget=latency_budget)

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this trace."""
        return {
            "id": self.id_,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "latency_ns": self.latency_ns,
            "spans": [span.json() for span in self.spans],
            "resource_usage_samples": [s.json() for s in self.resource_usage_samples],
        }

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

    def _header_lines(self, *, rich: bool) -> list[str]:
        budget_ms = self.latency_budget.total_seconds() * 1000
        if rich:
            return [
                f"[bold]Metrics Report for session '{self.session_id}'[/bold]",
                f"Latency budget: [green]{budget_ms:.2f}ms[/green]",
                f"Total traces: {len(self.traces)}",
                f"Slow traces: {len(self.slow_traces)}",
                "",
            ]
        return [
            f"Metrics Report for session '{self.session_id}'",
            f"Latency budget: {budget_ms:.2f}ms",
            f"Total traces: {len(self.traces)}",
            f"Slow traces: {len(self.slow_traces)}",
            "",
        ]

    def _trace_lines(self, traces: list[Trace], *, rich: bool) -> list[str]:
        lines: list[str] = []
        for trace in traces:
            lines.extend(
                [
                    "-" * 40,
                    "",
                    trace.summary_rich(self.latency_budget)
                    if rich
                    else trace.summary(self.latency_budget),
                    "",
                ]
            )
        return lines

    def summary(self) -> str:
        """Generate a human-readable summary of this report, including stats and slow traces."""
        lines = self._header_lines(rich=False)
        if self.slow_traces:
            lines.append("Slow Traces:\n")
            lines.extend(self._trace_lines(self.slow_traces, rich=False))
        return "\n".join(lines)

    def summary_rich(self) -> str:
        """Generate a human-readable summary of this report, with rich formatting.

        Includes stats and slow traces.
        """
        lines = self._header_lines(rich=True)
        if self.slow_traces:
            lines.append("[bold red]Slow Traces:[/bold red]")
            lines.extend(self._trace_lines(self.slow_traces, rich=True))
        return "\n".join(lines)

    def summary_full(self) -> str:
        """Generate a full human-readable summary of this report, including all traces."""
        lines = self._header_lines(rich=False)
        lines.append("All Traces:")
        lines.extend(self._trace_lines(self.traces, rich=False))
        return "\n".join(lines)

    def summary_full_rich(self) -> str:
        """Generate a full summary of this report, including all traces, with rich formatting."""
        lines = self._header_lines(rich=True)
        lines.append("[bold]All Traces:[/bold]")
        lines.extend(self._trace_lines(self.traces, rich=True))
        return "\n".join(lines)

    def json(self) -> dict[str, t.Any]:
        """Generate a JSON-serializable dictionary representation of this report."""
        return {
            "session_id": self.session_id,
            "latency_budget_ms": self.latency_budget.total_seconds() * 1000,
            "traces": [trace.json() for trace in self.traces],
            "slow_traces": [trace.json() for trace in self.slow_traces],
        }
