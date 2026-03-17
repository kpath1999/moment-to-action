"""Public types for the metrics subsystem.

Dataclasses and enums used by :class:`~moment_to_action.metrics.MetricsCollector`
and surfaced to callers that consume reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


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

    stage_idx: int
    """Pipeline stage index (1 = trigger/sensor, 2 = vision/LLM)."""

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
class StageStats:
    """Latency statistics for a single stage across all recorded executions."""

    num_calls: int
    """Number of executions included in these statistics."""

    mean_ms: float
    """Mean latency in milliseconds."""

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
class LatencyBudget:
    """Latency budget analysis measured against the configured target."""

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

    per_stage: dict[str, StageStats]
    """Per-stage latency statistics keyed by stage name."""

    pipeline: PipelineStats
    """Aggregate pipeline event statistics."""

    latency_budget: LatencyBudget
    """Latency budget analysis."""
