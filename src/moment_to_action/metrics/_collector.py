"""Central metrics collection for the framework.

Every model inference, pipeline trigger, and detection event
reports here. This is your research results table in code form.

At the end of a run, call report() to get a summary suitable
for including directly in your paper's results section.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Pipeline event types recorded by :class:`MetricsCollector`."""

    TRIGGER_FIRED = auto()
    """A pipeline trigger condition was met."""

    DETECTION = auto()
    """An object or activity was detected by a model."""

    FALSE_POSITIVE = auto()
    """A detection was later determined to be incorrect."""


@dataclass
class PipelineRecord:
    """Record of a pipeline-level event."""

    timestamp: float
    """Unix epoch timestamp of the event."""

    event_type: EventType
    """Category of pipeline event."""

    stage_idx: int
    """Pipeline stage index (1 or 2)."""

    latency_ms: float
    """End-to-end latency for this event in milliseconds."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary key/value context for this event."""


@dataclass
class StageRecord:
    """Record of a single stage execution (logged by the Stage wrapper)."""

    timestamp: float
    """Unix epoch timestamp when the stage ran."""

    stage_name: str
    """Class name of the stage."""

    latency_ms: float
    """Wall-clock time for this stage in milliseconds."""

    metadata: dict = field(default_factory=dict)
    """Arbitrary key/value context attached at log time."""


@dataclass
class EventRecord:
    """General-purpose event record (load times, config changes, etc.)."""

    timestamp: float
    """Unix epoch timestamp of the event."""

    event_type: str
    """Free-form event category string."""

    data: dict = field(default_factory=dict)
    """Arbitrary payload for this event."""


@dataclass
class ModelStats:
    """Latency statistics for a single model across all recorded inferences."""

    n_inferences: int
    """Number of inferences included in these statistics."""

    mean_ms: float
    """Mean inference latency in milliseconds."""

    p50_ms: float
    """Median (50th percentile) latency in milliseconds."""

    p95_ms: float
    """95th percentile latency in milliseconds."""

    min_ms: float
    """Minimum observed latency in milliseconds."""

    max_ms: float
    """Maximum observed latency in milliseconds."""


@dataclass
class PipelineStats:
    """High-level statistics over all recorded pipeline events."""

    total_triggers: int
    """Number of trigger-fired events."""

    total_detections: int
    """Number of detection events."""

    trigger_rate: float
    """Fraction of pipeline events that were triggers (0-1)."""


@dataclass
class StageLatencyStats:
    """Latency summary for one pipeline stage."""

    mean_ms: float
    """Mean latency in milliseconds."""

    p95_ms: float
    """95th percentile latency in milliseconds."""


@dataclass
class LatencyBudget:
    """Latency budget analysis measured against the configured target."""

    stage1: StageLatencyStats | None
    """Stage-1 (sensor/trigger model) latency statistics, or ``None`` if no data."""

    stage2: StageLatencyStats | None
    """Stage-2 (vision/LLM model) latency statistics, or ``None`` if no data."""

    total_mean_ms: float
    """Sum of stage-1 and stage-2 mean latencies."""

    budget_ms: float
    """Target latency budget in milliseconds."""

    headroom_ms: float
    """Remaining budget: ``budget_ms - total_mean_ms``."""

    within_budget: bool
    """``True`` when ``total_mean_ms < budget_ms``."""


@dataclass
class CollectorReport:
    """Full summary report produced by :meth:`MetricsCollector.report`."""

    session_id: str
    """Unique identifier for this collection session."""

    total_stages: int
    """Total number of stage executions recorded."""

    total_pipeline_events: int
    """Total number of pipeline-level events recorded."""

    per_stage: dict[str, ModelStats]
    """Per-stage latency statistics keyed by stage name."""

    pipeline: PipelineStats
    """Aggregate pipeline event statistics."""

    latency_budget: LatencyBudget
    """Latency budget analysis."""


class MetricsCollector:
    """Collects timing, accuracy, and power metrics across the pipeline.

    Thread-safe for concurrent writes from sensor threads.
    """

    def __init__(
        self,
        session_id: str | None = None,
        latency_budget_ms: float = 5000.0,
    ) -> None:
        self._session_id = session_id or f"session_{int(time.time())}"
        self._latency_budget_ms = latency_budget_ms
        self._pipeline_log: list[PipelineRecord] = []
        self._stage_log: list[StageRecord] = []
        self._event_log: list[EventRecord] = []
        self._timers: dict[str, float] = {}

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_pipeline_event(
        self,
        event_type: EventType,
        stage_idx: int,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a pipeline-level event such as a trigger or detection."""
        self._pipeline_log.append(
            PipelineRecord(
                timestamp=time.time(),
                event_type=event_type,
                stage_idx=stage_idx,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )

    def log_stage(
        self,
        stage_name: str,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a single stage execution."""
        self._stage_log.append(
            StageRecord(
                timestamp=time.time(),
                stage_name=stage_name,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )

    def log_event(self, event_type: str, data: dict) -> None:
        """General purpose event log for load times, config changes, etc."""
        self._event_log.append(
            EventRecord(
                timestamp=time.time(),
                event_type=event_type,
                data=data,
            )
        )

    # ------------------------------------------------------------------
    # Timer helpers - measure wall-clock spans
    # ------------------------------------------------------------------

    def start_timer(self, name: str) -> None:
        """Start a named wall-clock timer."""
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Return elapsed ms since :meth:`start_timer` was called."""
        if name not in self._timers:
            msg = f"Timer '{name}' was never started"
            raise KeyError(msg)
        elapsed_ms = (time.perf_counter() - self._timers.pop(name)) * 1000
        self.log_event("timer", {"name": name, "elapsed_ms": elapsed_ms})
        return elapsed_ms

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> CollectorReport:
        """Generate a summary report across all collected metrics."""
        return CollectorReport(
            session_id=self.session_id,
            total_stages=len(self._stage_log),
            total_pipeline_events=len(self._pipeline_log),
            per_stage=self._per_stage_stats(),
            pipeline=self._pipeline_stats(),
            latency_budget=self._latency_budget_analysis(),
        )

    def _per_stage_stats(self) -> dict[str, ModelStats]:
        """Compute per-stage latency statistics from the stage log."""
        if not self._stage_log:
            return {}

        by_stage: dict[str, list[float]] = {}
        for record in self._stage_log:
            by_stage.setdefault(record.stage_name, []).append(record.latency_ms)

        return {
            stage: ModelStats(
                n_inferences=len(latencies),
                mean_ms=float(np.mean(arr := np.array(latencies))),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
            )
            for stage, latencies in by_stage.items()
        }

    def _pipeline_stats(self) -> PipelineStats:
        triggers = [r for r in self._pipeline_log if r.event_type == EventType.TRIGGER_FIRED]
        detections = [r for r in self._pipeline_log if r.event_type == EventType.DETECTION]
        return PipelineStats(
            total_triggers=len(triggers),
            total_detections=len(detections),
            trigger_rate=len(triggers) / max(1, len(self._pipeline_log)),
        )

    def _latency_budget_analysis(self) -> LatencyBudget:
        """Break down latency against the configured budget target.

        Callers pass ``metadata={"stage_idx": 1}`` or ``{"stage_idx": 2}``
        to :meth:`log_stage`; records without a ``stage_idx`` are counted
        in the total but not bucketed into stage-1 or stage-2.
        """
        stage1_idx = 1
        stage2_idx = 2
        stage1_records = [r for r in self._stage_log if r.metadata.get("stage_idx") == stage1_idx]
        stage2_records = [r for r in self._stage_log if r.metadata.get("stage_idx") == stage2_idx]

        def _stats(records: list[StageRecord]) -> StageLatencyStats | None:
            if not records:
                return None
            arr = np.array([r.latency_ms for r in records])
            return StageLatencyStats(
                mean_ms=float(np.mean(arr)),
                p95_ms=float(np.percentile(arr, 95)),
            )

        s1 = _stats(stage1_records)
        s2 = _stats(stage2_records)
        total_mean = (s1.mean_ms if s1 else 0.0) + (s2.mean_ms if s2 else 0.0)

        return LatencyBudget(
            stage1=s1,
            stage2=s2,
            total_mean_ms=total_mean,
            budget_ms=self._latency_budget_ms,
            headroom_ms=self._latency_budget_ms - total_mean,
            within_budget=total_mean < self._latency_budget_ms,
        )

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
        Path(path).write_text(json.dumps(dataclasses.asdict(self.report()), indent=2))
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
                "  %-20s  mean=%.1fms  p95=%.1fms",
                stage,
                stats.mean_ms,
                stats.p95_ms,
            )
        budget = r.latency_budget
        logger.info("\nLatency budget (target <%.0fms):", budget.budget_ms)
        logger.info("  Stage 1: %.1fms", budget.stage1.mean_ms if budget.stage1 else 0.0)
        logger.info("  Stage 2: %.1fms", budget.stage2.mean_ms if budget.stage2 else 0.0)
        status = "✓ within budget" if budget.within_budget else "✗ over budget"
        logger.info("  Total:   %.1fms  (%s)", budget.total_mean_ms, status)
        logger.info("=" * 50)
