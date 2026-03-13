# Edge Inference

---

## Overview

Implemnetation of a perception pipeline as a sequence of **stages**. Each stage receives a typed **message**, performs one unit of computation, and emits a new typed message. 

```
SensorStage
    ↓ RawFrameMessage
VisionPreprocessStage
    ↓ TensorMessage
YOLOStage / MobileCLIPStage
    ↓ DetectionMessage / ClassificationMessage
ReasoningStage
    ↓ ReasoningMessage
```

Latency is measured automatically at every stage boundary via `MetricsCollector`.
Currently, the model stage does the model loading and inference. An alternate
idea is to have a separate model class with the necessary functions implmented and referenced
in the stage. I'm keeping it simple for now.
Future implementation: TBD

---

## Project Structure

```
edgeperceive/
├── core/
│   └── messages.py                     # Core message types
│
├── stages/
│   ├── base.py                         # Stage (abstract) and Pipeline
│   └── vision_stages/                  # Think of the files in the vision stages as forming a pipeline (final structure TBD)
│       ├── vision_preprocess_stage.py  # VisionPreprocessStage
│       ├── yolo_stage.py               # YOLOStage
│       └── mobileclip_stage.py         # MobileCLIPStage
│
├── preprocessors/
│   ├── base.py                         # BasePreprocessor
│   └── video/
│       └── video_preprocessing.py      # VideoPreprocessor
│
├── hardware/
│   ├── compute_backend.py              # HAL: ONNXBackend, LiteRTBackend, CPUBackend
│   └── types.py                        # Hardware compute units (CPU, NPU, GPU, DSP)
│
├── models/
│   ├── mobileclip_s2/
│   │   └── mobileclip_s2_datacompdr_last.tflite
│   └── yolo/
│       └── model.onnx
│
├── metrics/
│   └── collector.py                    # MetricsCollector — per-stage latency and stats
│
└── pipeline/                           # Run scripts
    ├── run_yolo_pipeline.py
    ├── run_mobileclip_pipeline.py
```

---

## Validated Pipelines

To clone the repo
```bash
git clone <repo-url>
cd moment-to-action
git checkout soma/api
```

If already using the repo
```bash
git fetch origin
git checkout soma/api
cd moment-to-action
```

```bash
uv sync
```

To run ruff
```bash
uv run ruff check src/moment_to_action
```

### YOLOv8 Object Detection

```bash
uv run python -m moment_to_action.edgeperceive.pipeline.run_yolo_pipeline \
  --image src/moment_to_action/edgeperceive/images/<choose_image>.jpg \
  --model src/moment_to_action/edgeperceive/models/yolo/model.onnx \
  --conf 0.3
```

Draw bounding boxes on the output image:

```bash
uv run python -m moment_to_action.edgeperceive.pipeline.draw_detections \
  --image src/moment_to_action/edgeperceive/images/<choose_image>.jpg \
  --model src/moment_to_action/edgeperceive/models/yolo/model.onnx \
  --conf 0.3 \
  --out result.jpg
```

### MobileCLIP-S2 Zero-Shot Classification

```bash
uv run python -m moment_to_action.edgeperceive.pipeline.run_mobileclip_pipeline \
  --image src/moment_to_action/edgeperceive/images/<choose_image>.jpg \
  --model src/moment_to_action/edgeperceive/models/mobileclip_s2/mobileclip_s2_datacompdr_last.tflite
```

Both scripts accept `--device cpu` (default) or `--device npu`.
Please use default mode for now. Working on implementing the NPU backend.

---

## Key Concepts

### Stages and Messages

A `Stage` transforms one `Message` into another. Returning `None` stops the pipeline — downstream stages do not run. This mechanism helps with gating expensive computation on confidence thresholds.

```python
from moment_to_action.edgeperceive.stages.base import Stage
from moment_to_action.edgeperceive.core.messages import TensorMessage, DetectionMessage

class MyModelStage(Stage):
    def process(self, msg: TensorMessage) -> DetectionMessage | None:
        # return None to stop the pipeline here
        ...
```

### Building a Pipeline

```python
from moment_to_action.edgeperceive.stages.base import Pipeline
from moment_to_action.edgeperceive.core.messages import RawFrameMessage
from moment_to_action.edgeperceive.stages.vision_stages import VisionPreprocessStage
from moment_to_action.edgeperceive.stages.vision_stages import YOLOStage
from moment_to_action.edgeperceive.metrics.collector import MetricsCollector

metrics = MetricsCollector()

pipeline = Pipeline(stages=[
    SensorStage(),
    VisionPreprocessStage(target_size=(640, 640), letterbox=True),
    YOLOStage(model_path="models/yolo/model.onnx", confidence_threshold=0.3),
], metrics=metrics)

result = pipeline.run(RawFrameMessage(frame=None, timestamp=0., source="image.jpg"))
metrics.print_stage_latencies()
```

### Swapping Models

Change the model stage without touching anything else:
This works well for models that ingest the same format of data.
In MobileCLIP vs YOLO, the PreprocessStage would be passed different arguments corresponding to the different image dimensions they deal with.

```python
# Object detection
pipeline = Pipeline([SensorStage(), VisionPreprocessStage(...), YOLOStage(...)])

# Zero-shot classification
pipeline = Pipeline([SensorStage(), VisionPreprocessStage(...), MobileCLIPStage(...)])
```

### Compute Backend

The `ComputeBackend` selects the right runtime automatically based on file extension:

| Model format | Runtime | Compute unit |
|---|---|---|
| `.onnx` | onnxruntime | CPU |
| `.tflite` | ai_edge_litert | CPU / NPU via QNN delegate |

### Metrics

`MetricsCollector` is injected into `Pipeline` and receives latency from every stage automatically.

```python
metrics = MetricsCollector()
pipeline = Pipeline(stages=[...], metrics=metrics)
pipeline.run(msg)

metrics.print_stage_latencies()  # per-stage latency table
metrics.print_summary()          # full report with p50/p95
metrics.save("results.json")     # persist to disk
```

Sample output:
```
Stage                      Latency
─────────────────────────────────────
  SensorStage                  11.2ms
  VisionPreprocessStage         8.4ms
  YOLOStage                   134.1ms
─────────────────────────────────────
  Total                       153.7ms
```

---

## Supported Models

| Model | Task | Format | Status |
|---|---|---|---|
| YOLOv8 | Object detection | ONNX | Validated |
| MobileCLIP-S2 | Zero-shot classification | TFLite | Validated |
| YAMNet | Audio event detection | TFLite | In progress |
| Qwen 0.8B / TinyLlama | Scene reasoning | TFLite | In progress |

---
