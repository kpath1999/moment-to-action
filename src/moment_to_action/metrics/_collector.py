"""Central metrics collection for the framework.

Every model inference, pipeline trigger, and detection event
reports here. This is your research results table in code form.

At the end of a run, call report() to get a summary suitable
for including directly in your paper's results section.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

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

logger = logging.getLogger(__name__)

_PROCESS = psutil.Process(os.getpid())


def _rss_mb() -> float:
    """Return current process RSS in MB."""
    return _PROCESS.memory_info().rss / 1024**2


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

        # Snapshot RSS at collector creation — used as the baseline
        self._baseline_rss_mb: float = _rss_mb()

    @property
    def session_id(self) -> str:
        """Return the session identifier."""
        return self._session_id

    @property
    def baseline_rss_mb(self) -> float:
        """RSS at pipeline startup — before any models loaded."""
        return self._baseline_rss_mb

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
        init_memory_bytes: int = 0,
        runtime_memory_bytes: int = 0,
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

    def log_event(self, event_type: str, data: dict) -> None:
        """General purpose event log for load times, config changes, etc."""
        self._event_log.append(
            EventRecord(
                timestamp=time.time(),
                event_type=event_type,
                data=data,
            )
        )

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
