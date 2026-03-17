"""Central metrics collection for the framework.

Every model inference, pipeline trigger, and detection event
reports here. This is your research results table in code form.

At the end of a run, call report() to get a summary suitable
for including directly in your paper's results section.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from moment_to_action.hardware._types import ComputeUnit  # noqa: TC001

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
class InferenceRecord:
    """Record of a single model inference."""

    timestamp: float
    model_name: str
    latency_ms: float
    label: str
    confidence: float
    compute_unit: ComputeUnit


@dataclass
class PipelineRecord:
    """Record of a pipeline-level event."""

    timestamp: float
    event_type: EventType
    stage: int  # 1 or 2
    latency_ms: float
    metadata: dict = field(default_factory=dict)


LATENCY_BUDGET_MS = 5000


class MetricsCollector:
    """Collects timing, accuracy, and power metrics across the pipeline.

    Thread-safe for concurrent writes from sensor threads.
    """

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or f"session_{int(time.time())}"
        self._inference_log: list[InferenceRecord] = []
        self._pipeline_log: list[PipelineRecord] = []
        self._event_log: list[dict] = []
        self._timers: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_inference(
        self,
        model_name: str,
        latency_ms: float,
        label: str,
        confidence: float,
        compute_unit: ComputeUnit,
    ) -> None:
        """Record a single model inference with its timing and result."""
        self._inference_log.append(
            InferenceRecord(
                timestamp=time.time(),
                model_name=model_name,
                latency_ms=latency_ms,
                label=label,
                confidence=confidence,
                compute_unit=compute_unit,
            )
        )

    def log_pipeline_event(
        self,
        event_type: EventType,
        stage: int,
        latency_ms: float,
        metadata: dict | None = None,
    ) -> None:
        """Record a pipeline-level event such as a trigger or detection."""
        self._pipeline_log.append(
            PipelineRecord(
                timestamp=time.time(),
                event_type=event_type,
                stage=stage,
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
        self._event_log.append(
            {
                "timestamp": time.time(),
                "type": "stage",
                "stage_name": stage_name,
                "latency_ms": latency_ms,
                **(metadata or {}),
            }
        )

    def log_event(self, event_type: str, data: dict) -> None:
        """General purpose event log for load times, config changes, etc."""
        self._event_log.append(
            {
                "timestamp": time.time(),
                "type": event_type,
                **data,
            }
        )

    # ------------------------------------------------------------------
    # Timer helpers - measure wall-clock spans
    # ------------------------------------------------------------------

    def start_timer(self, name: str) -> None:
        """Start a named wall-clock timer."""
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Returns elapsed ms since start_timer(name) was called."""
        if name not in self._timers:
            msg = f"Timer '{name}' was never started"
            raise KeyError(msg)
        elapsed_ms = (time.perf_counter() - self._timers.pop(name)) * 1000
        self.log_event("timer", {"name": name, "elapsed_ms": elapsed_ms})
        return elapsed_ms

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> dict:
        """Generate a summary report across all collected metrics.

        This is your results table.
        """
        return {
            "session_id": self.session_id,
            "total_inferences": len(self._inference_log),
            "total_pipeline_events": len(self._pipeline_log),
            "per_model": self._per_model_stats(),
            "pipeline": self._pipeline_stats(),
            "latency_budget": self._latency_budget_analysis(),
        }

    def _per_model_stats(self) -> dict:
        if not self._inference_log:
            return {}

        by_model: dict[str, list[float]] = {}
        for record in self._inference_log:
            by_model.setdefault(record.model_name, []).append(record.latency_ms)

        stats = {}
        for model, latencies in by_model.items():
            arr = np.array(latencies)
            stats[model] = {
                "n_inferences": len(latencies),
                "mean_ms": float(np.mean(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
            }
        return stats

    def _pipeline_stats(self) -> dict:
        triggers = [r for r in self._pipeline_log if r.event_type == EventType.TRIGGER_FIRED]
        detections = [r for r in self._pipeline_log if r.event_type == EventType.DETECTION]

        return {
            "total_triggers": len(triggers),
            "total_detections": len(detections),
            "trigger_rate": len(triggers) / max(1, len(self._pipeline_log)),
        }

    def _latency_budget_analysis(self) -> dict:
        """Break down latency against the <5s target.

        Shows exactly where time is spent in the pipeline.
        """
        stage1 = [
            r
            for r in self._inference_log
            if any(name in r.model_name.lower() for name in ["yamnet", "imu", "ir"])
        ]
        stage2 = [
            r
            for r in self._inference_log
            if any(
                name in r.model_name.lower() for name in ["movinet", "mobileclip", "yolo", "qwen"]
            )
        ]

        def stats(records: list[InferenceRecord]) -> dict[str, float]:
            if not records:
                return {}
            arr = np.array([r.latency_ms for r in records])
            return {"mean_ms": float(np.mean(arr)), "p95_ms": float(np.percentile(arr, 95))}

        s1 = stats(stage1)
        s2 = stats(stage2)
        total_mean = s1.get("mean_ms", 0) + s2.get("mean_ms", 0)

        return {
            "stage1": s1,
            "stage2": s2,
            "total_mean_ms": total_mean,
            "budget_ms": LATENCY_BUDGET_MS,
            "headroom_ms": LATENCY_BUDGET_MS - total_mean,
            "within_budget": total_mean < LATENCY_BUDGET_MS,
        }

    def print_stage_latencies(self) -> None:
        """Print latency table for the most recent pipeline run."""
        stage_records = [e for e in self._event_log if e.get("type") == "stage"]
        if not stage_records:
            logger.info("No stage latencies recorded.")
            return
        total = sum(e["latency_ms"] for e in stage_records)
        logger.info("\n%-25s %10s", "Stage", "Latency")
        logger.info("─" * 37)
        for e in stage_records:
            logger.info("  %-23s %8.1fms", e["stage_name"], e["latency_ms"])
        logger.info("─" * 37)
        logger.info("  %-23s %8.1fms", "Total", total)

    def save(self, path: str) -> None:
        """Save full report to JSON."""
        from pathlib import Path

        Path(path).write_text(json.dumps(self.report(), indent=2))
        logger.info("Metrics saved to %s", path)

    def print_summary(self) -> None:
        """Log a human-readable summary."""
        r = self.report()
        logger.info("\n%s", "=" * 50)
        logger.info("METRICS SUMMARY  |  session: %s", r["session_id"])
        logger.info("=" * 50)
        logger.info("Total inferences: %d", r["total_inferences"])
        logger.info("\nPer-model latency:")
        for model, stats in r["per_model"].items():
            logger.info(
                "  %-20s  mean=%.1fms  p95=%.1fms",
                model,
                stats["mean_ms"],
                stats["p95_ms"],
            )
        budget = r["latency_budget"]
        logger.info("\nLatency budget (target <%dms):", LATENCY_BUDGET_MS)
        logger.info("  Stage 1: %.1fms", budget.get("stage1", {}).get("mean_ms", 0))
        logger.info("  Stage 2: %.1fms", budget.get("stage2", {}).get("mean_ms", 0))
        status = "✓ within budget" if budget["within_budget"] else "✗ over budget"
        logger.info("  Total:   %.1fms  (%s)", budget["total_mean_ms"], status)
        logger.info("=" * 50)
