# Moment2Action App Container

`Moment2Action` is the single entry point for consumers of this library. It owns
path/config/logging setup and the hardware `Platform`, and hides `PathManager`,
`ModelManager`, and the raw `Pipeline` — you never construct or import those
directly.

```python
from moment_to_action import Moment2Action

app = Moment2Action()
```

Passing `config=` overrides the persisted `AppConfig`; passing `qairt=True`
configures the QAIRT SDK environment (`QAIRT_SDK_ROOT` etc.) before the platform
is built — needed for DLC/NPU backends.

## Building a pipeline

`new_pipeline(name)` returns a `PipelineBuilder`. Chain `.add_stage(...)` calls
and finish with `.build()`:

```python
from moment_to_action.hardware import ComputeUnit
from moment_to_action.models import ModelID
from moment_to_action.stages.image import ImageDetectionStage
from moment_to_action.stages.llm import DecisionStage, LLMStage
from moment_to_action.prompting import YES_NO_GRAMMAR

handle = (
    app.new_pipeline("fall-detector")
    .add_stage(
        ImageDetectionStage,
        model_id=ModelID.YOLO_V8,
        compute_unit=ComputeUnit.NPU,
    )
    .add_stage(
        LLMStage,
        model_id=ModelID.QWEN3_0_6B,
        model_kwargs={"max_tokens": 128},
        grammar=YES_NO_GRAMMAR,
        compute_unit=ComputeUnit.GPU,
    )
    .add_stage(DecisionStage)
    .build()
)
```

- `model_id=` resolves and constructs the model via this pipeline's own
  `ModelManager`; it's passed as the stage's first positional constructor
  argument. Omit it for stages that don't wrap a model (e.g. `DecisionStage`).
- `compute_unit=` is recorded alongside the stage and used later by
  `load_pipeline()` — the model is **not** loaded during `add_stage`/`build`.
- `model_kwargs=` forwards extra keyword arguments to the model constructor
  (e.g. `system_prompt=`, `max_tokens=`). Any other `**kwargs` go straight to
  the stage's own constructor (e.g. `grammar=`).
- `metrics=` is always injected automatically — every stage in one pipeline
  shares one `MetricsCollector`, so their spans nest under one trace per
  pipeline.

`build()` registers the pipeline under `name` but does **not** load it — a
freshly built pipeline holds no device resources yet.

## Loading, running, unloading

`load_pipeline`/`unload_pipeline`/`remove_pipeline`/`metrics_report` take the
`PipelineHandle` itself, not a name — `build()` already gave you one; use
`get_pipeline(name)` to look one up again later (e.g. in a different function
that only has the name). Only one pipeline may be loaded (holding device
resources) at a time.

```python
app.load_pipeline(handle)                # acquires device resources

for msg in handle.run(sensor.stream()):  # drive it
    ...

app.unload_pipeline(handle)              # releases resources, keeps registration
app.load_pipeline(handle)                # reload later without rebuilding stages
app.remove_pipeline(handle)              # unloads (if needed) and discards it

# looking a handle up again elsewhere:
handle = app.get_pipeline("fall-detector")
```

- `load_pipeline(handle)` raises `RuntimeError` if a *different* pipeline is
  already active — unload it first.
- `unload_pipeline(handle=None)` defaults to the active pipeline. It releases
  device resources but keeps the pipeline registered, so `load_pipeline` can
  bring it back without reconstructing any stage or model.
- `remove_pipeline(handle)` is the only thing that actually deletes a
  registration (unloading first if it was loaded).
- All four raise `ValueError` if the handle isn't registered on this app
  (e.g. already removed, or built by a different `Moment2Action`).

## Metrics

```python
report = app.metrics_report(handle)  # or app.metrics_report() for the active pipeline
```

Returns that pipeline's own `MetricsReport` — each pipeline has an isolated
`MetricsCollector`, so one pipeline's report never contains another's spans.

## Cleanup

```python
with Moment2Action() as app:
    ...
# equivalent to calling app.close(), which unloads the active pipeline
```

## Bounded streams: `EndOfClipMessage`

A stage that must accumulate across a bounded run of messages (e.g. "every
frame in this clip", or "every token of this generation") doesn't need to know
the run's length up front, and doesn't need a boolean flag bolted onto every
payload message. Instead, the producer sends one shared sentinel — a
payload-free control message, matched by `isinstance`, not a field — after the
last real message of the run:

- `ImageDetectionStage`/`DetectionAggregationStage`: sent after the last frame
  of a clip. `DetectionAggregationStage` keeps a running
  highest-confidence-per-label accumulation as instance state (`window=1`,
  real time, no buffering) and flushes it as one `DetectionMessage` when it
  sees this sentinel. `ImageDetectionStage` passes it through unchanged (it
  isn't a frame, so its drop predicate has to special-case it) so it reaches
  the aggregation stage.
- `LLMStage`/`VLMDescriptionStage`/`DecisionStage`: sent after the last
  `GenerationMessage` for a prompt, replacing what used to be a `done=True`
  flag on the terminal message. `DecisionStage` forwards it unchanged after
  its own last `DecisionReasoningMessage`, since the (stripped) reasoning
  stream shares the same lifecycle as the generation it's derived from.

It's the same `EndOfClipMessage` type in both cases — it carries no `prompt`
or clip ID, because a single pipeline processes one bounded run to completion
before starting the next, so there's never more than one run's sentinel live
to disambiguate. Stages that don't care about it just don't match it in their
`isinstance` dispatch — no `done`/`is_last` field to check on every message
they do care about.

## When a single `run()` isn't enough

Even with the sentinel-driven design above, `benchmark_real.py`'s LLM path
still can't be one chained `handle.run()` call end to end: `DecisionStage`
doesn't forward the raw `GenerationMessage` stream it reads (only the
extracted `DecisionMessage`/`DecisionReasoningMessage`), and this benchmark
needs the full raw text to score recall against. When you hit a case like
that, pull stages off the loaded handle and drive them directly, wrapped in
`handle.trace()` so their spans still land on that pipeline's own metrics:

```python
detection_stage, aggregation_stage, llm_stage, decision_stage = handle.stages

# detection -> aggregation chains naturally, one frame at a time:
aggregated = aggregation_stage.process(detection_stage.process(frame_stream))

with handle.trace():
    gen_messages = list(llm_stage.process(iter([aggregated_msg])))
    # keep gen_messages for the raw text, then also run decision_stage on it
    decisions = decision_stage.process(iter(gen_messages))
```

See `bench/benchmark_real.py` for the full example (`_detect_and_aggregate`,
`_run_llm_clip`).
