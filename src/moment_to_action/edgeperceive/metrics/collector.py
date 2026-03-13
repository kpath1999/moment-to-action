"""metrics/collector.py

Central metrics collection for the framework.

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

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    timestamp: float
    model_name: str
    latency_ms: float
    label: str
    confidence: float
    compute_unit: str


@dataclass
class PipelineRecord:
    timestamp: float
    event_type: str             # "trigger_fired", "detection", "false_positive"
    stage: int                  # 1 or 2
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class MetricsCollector:
    """Collects timing, accuracy, and power metrics across the pipeline.

    Thread-safe for concurrent writes from sensor threads.
    """

    def __init__(self, session_id: str | None = None):
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
        compute_unit: str,
    ) -> None:
        self._inference_log.append(InferenceRecord(
            timestamp=time.time(),
            model_name=model_name,
            latency_ms=latency_ms,
            label=label,
            confidence=confidence,
            compute_unit=compute_unit,
        ))

    def log_pipeline_event(
        self,
        event_type: str,
        stage: int,
        latency_ms: float,
        metadata: dict = None,
    ) -> None:
        self._pipeline_log.append(PipelineRecord(
            timestamp=time.time(),
            event_type=event_type,
            stage=stage,
            latency_ms=latency_ms,
            metadata=metadata or {},
        ))

    def log_stage(
        self,
        stage_name: str,
        latency_ms: float,
        metadata: dict = None,
    ) -> None:
        """Record a single stage execution."""
        self._event_log.append({
            "timestamp": time.time(),
            "type": "stage",
            "stage_name": stage_name,
            "latency_ms": latency_ms,
            **(metadata or {}),
        })

    def log_event(self, event_type: str, data: dict) -> None:
        """General purpose event log for load times, config changes, etc."""
        self._event_log.append({
            "timestamp": time.time(),
            "type": event_type,
            **data,
        })

    # ------------------------------------------------------------------
    # Timer helpers - measure wall-clock spans
    # ------------------------------------------------------------------

    def start_timer(self, name: str) -> None:
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Returns elapsed ms since start_timer(name) was called."""
        if name not in self._timers:
            raise KeyError(f"Timer '{name}' was never started")
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
        report = {
            "session_id": self.session_id,
            "total_inferences": len(self._inference_log),
            "total_pipeline_events": len(self._pipeline_log),
            "per_model": self._per_model_stats(),
            "pipeline": self._pipeline_stats(),
            "latency_budget": self._latency_budget_analysis(),
        }
        return report

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
        triggers = [r for r in self._pipeline_log if r.event_type == "trigger_fired"]
        detections = [r for r in self._pipeline_log if r.event_type == "detection"]

        return {
            "total_triggers": len(triggers),
            "total_detections": len(detections),
            "trigger_rate": len(triggers) / max(1, len(self._pipeline_log)),
        }

    def _latency_budget_analysis(self) -> dict:
        """Break down latency against the <5s target.
        Shows exactly where time is spent in the pipeline.
        """
        stage1 = [r for r in self._inference_log
                  if any(name in r.model_name.lower() for name in ["yamnet", "imu", "ir"])]
        stage2 = [r for r in self._inference_log
                  if any(name in r.model_name.lower() for name in ["movinet", "mobileclip", "yolo", "qwen"])]

        def stats(records):
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
            "budget_ms": 5000,
            "headroom_ms": 5000 - total_mean,
            "within_budget": total_mean < 5000,
        }

    def print_stage_latencies(self) -> None:
        """Print latency table for the most recent pipeline run."""
        stage_records = [
            e for e in self._event_log
            if e.get("type") == "stage"
        ]
        if not stage_records:
            print("No stage latencies recorded.")
            return
        total = sum(e["latency_ms"] for e in stage_records)
        print(f"\n{'Stage':<25} {'Latency':>10}")
        print("─" * 37)
        for e in stage_records:
            print(f"  {e['stage_name']:<23} {e['latency_ms']:>8.1f}ms")
        print("─" * 37)
        print(f"  {'Total':<23} {total:>8.1f}ms")

    def save(self, path: str) -> None:
        """Save full report to JSON."""
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=2)
        logger.info(f"Metrics saved to {path}")

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        r = self.report()
        print(f"\n{'='*50}")
        print(f"METRICS SUMMARY  |  session: {r['session_id']}")
        print(f"{'='*50}")
        print(f"Total inferences: {r['total_inferences']}")
        print("\nPer-model latency:")
        for model, stats in r["per_model"].items():
            print(f"  {model:20s}  mean={stats['mean_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms")
        budget = r["latency_budget"]
        print("\nLatency budget (target <5000ms):")
        print(f"  Stage 1: {budget.get('stage1', {}).get('mean_ms', 0):.1f}ms")
        print(f"  Stage 2: {budget.get('stage2', {}).get('mean_ms', 0):.1f}ms")
        print(f"  Total:   {budget['total_mean_ms']:.1f}ms  "
              f"({'✓ within budget' if budget['within_budget'] else '✗ over budget'})")
        print(f"{'='*50}\n")
