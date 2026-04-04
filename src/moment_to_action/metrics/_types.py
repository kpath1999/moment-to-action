"""Public types for the metrics subsystem.

Dataclasses and enums used by :class:`~moment_to_action.metrics.MetricsCollector`
and surfaced to callers that consume reports.
"""

from __future__ import annotations

from enum import Enum, auto

import attrs


class EventType(Enum):
    """Pipeline event types recorded by :class:`MetricsCollector`."""

    TRIGGER_FIRED = auto()
    """A pipeline trigger condition was met."""

    DETECTION = auto()
    """An object or activity was detected by a model."""

    FALSE_POSITIVE = auto()
    """A detection was later determined to be incorrect."""


class PipelineRecord:
    """Record of a pipeline-level event."""

    timestamp: float
    """Unix epoch timestamp of the event."""

    event_type: EventType
    """Category of pipeline event."""

    latency_ms: float
    """End-to-end latency for this event in milliseconds."""

    metadata: dict = attrs.Factory(dict)
    """Arbitrary key/value context for this event."""


@attrs.define(kw_only=True)
class StageRecord:
    """Record of a single stage execution (logged by the Stage wrapper)."""

    timestamp: float
    """Unix epoch timestamp when the stage ran."""

    stage_name: str
    """Class name of the stage."""

    stage_idx: int
    """Pipeline stage index."""

    latency_ms: float
    """Wall-clock time for this stage in milliseconds."""

    metadata: dict = attrs.Factory(dict)
    """Arbitrary key/value context attached at log time."""

    # init_memory_bytes: int = 0 # set once at stage init time
    init_memory_bytes: int  # set once at stage init time
    """To measure memory consumed by the stage's models and other helper data"""

    # runtime_memory_bytes: int = 0
    runtime_memory_bytes: int
    """To measure the memory consumed by the stage during computation/process"""


@attrs.define
class LLMRecord(StageRecord):
    """The LLMStage requires a separate class to keep track of its metrics
    And due to its heavy resource usage. Discuss in PR
    Per-call record from llama-server, logged by LLMStage."""

    # timing (from /completion timings or measured wall-clock)
    prompt_ms: float
    gen_ms: float
    # TODO total_ms: float

    # token counts
    prompt_tokens: int
    gen_tokens: int
    tokens_per_second: float

    # server state (from /slots)
    kv_cache_used_tokens: int
    kv_cache_total_tokens: int
    kv_cache_ratio: float  # derived: used/total

    # server process memory (from psutil on llama-server pid)
    server_rss_bytes: int


@attrs.define
class EventRecord:
    """General-purpose event record (load times, config changes, etc.)."""

    timestamp: float
    """Unix epoch timestamp of the event."""

    event_type: str
    """Free-form event category string."""

    data: dict = attrs.Factory(dict)
    """Arbitrary payload for this event."""


@attrs.define
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

    init_memory_bytes: int  # constant across calls, just store it
    """Will remain roughly the same, but no harm in collection"""

    mean_runtime_memory_bytes: int


# --- new: aggregate LLM stats for CollectorReport ---
@attrs.define
class LLMStats(StageStats):
    """Aggregated statistics over all LLMRecords in a session."""

    mean_prompt_ms: float
    mean_gen_ms: float
    p95_gen_ms: float
    mean_tokens_per_second: float
    mean_kv_cache_ratio: float
    peak_kv_cache_ratio: float  # did we ever get close to OOM?
    mean_server_rss_bytes: int
    peak_server_rss_bytes: int


@attrs.define
class PipelineStats:
    """High-level statistics over all recorded pipeline events."""

    total_triggers: int
    """Number of trigger-fired events."""

    total_detections: int
    """Number of detection events."""

    total_false_positives: int
    """Number of false positive events."""

    trigger_rate: float
    """Fraction of pipeline events that were triggers (0-1)."""

    false_positive_rate: float
    """Fraction of detections that were false positives (0-1)."""


@attrs.define
class LatencyBudget:
    """Latency budget analysis measured against the configured target."""

    total_mean_ms: float
    """Mean end-to-end pipeline latency in milliseconds."""

    budget_ms: float
    """Target latency budget in milliseconds."""

    headroom_ms: float
    """Remaining budget: ``budget_ms - total_mean_ms``."""

    within_budget: bool
    """``True`` when ``total_mean_ms < budget_ms``."""


@attrs.define
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
