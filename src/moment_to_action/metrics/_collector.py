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
from pathlib import Path

import numpy as np

from moment_to_action.metrics._types import (
    CollectorReport,
    EventRecord,
    EventType,
    LatencyBudget,
    PipelineRecord,
    PipelineStats,
    StageLatencyStats,
    StageRecord,
    StageStats,
)

logger = logging.getLogger(__name__)


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
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a pipeline-level event such as a trigger or detection."""
        self._pipeline_log.append(
            PipelineRecord(
                timestamp=time.time(),
                event_type=event_type,
                latency_ms=latency_ms,
                metadata=metadata or {},
            )
        )

    def log_stage(
        self,
        stage_name: str,
        stage_idx: int,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a single stage execution.

        Args:
            stage_name: Class name of the stage (e.g. ``"YOLOStage"``).
            stage_idx: Pipeline stage index (1 = trigger/sensor, 2 = vision/LLM).
            latency_ms: Wall-clock time for this stage in milliseconds.
            metadata: Optional extra context to attach to the record.
        """
        self._stage_log.append(
            StageRecord(
                timestamp=time.time(),
                stage_name=stage_name,
                stage_idx=stage_idx,
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

    def _per_stage_stats(self) -> dict[str, StageStats]:
        """Compute per-stage latency statistics from the stage log."""
        if not self._stage_log:
            return {}

        by_stage: dict[str, list[float]] = {}
        for record in self._stage_log:
            by_stage.setdefault(record.stage_name, []).append(record.latency_ms)

        return {
            stage: StageStats(
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
        """Break down latency per stage_idx against the configured budget target."""
        # Group stage records by stage_idx.
        by_idx: dict[int, list[StageRecord]] = {}
        for r in self._stage_log:
            by_idx.setdefault(r.stage_idx, []).append(r)

        stages: dict[int, StageLatencyStats] = {}
        for idx, records in by_idx.items():
            arr = np.array([r.latency_ms for r in records])
            stages[idx] = StageLatencyStats(
                mean_ms=float(np.mean(arr)),
                p95_ms=float(np.percentile(arr, 95)),
            )

        total_mean = sum(s.mean_ms for s in stages.values())
        return LatencyBudget(
            stages=stages,
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
        for idx, stats in sorted(budget.stages.items()):
            logger.info("  Stage %d: %.1fms", idx, stats.mean_ms)
        status = "✓ within budget" if budget.within_budget else "✗ over budget"
        logger.info("  Total:   %.1fms  (%s)", budget.total_mean_ms, status)
        logger.info("=" * 50)
