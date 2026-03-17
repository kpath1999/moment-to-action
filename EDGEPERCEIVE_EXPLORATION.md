# EdgePerceive Framework - Complete Exploration & Refactoring Guide

> **Generated**: Comprehensive analysis of `/src/moment_to_action/edgeperceive/`
> **Scope**: 23 Python files, 47 classes, 100+ methods across 7 modules

---

## 📋 Executive Summary

EdgePerceive is a **message-driven perception pipeline framework** optimized for edge deployment on QCS6490 (Snapdragon) devices. It provides:

- ✅ **Message-based architecture** with typed data contracts
- ✅ **Hardware abstraction** supporting TFLite (NPU/CPU) and ONNX
- ✅ **Generic preprocessing** with zero-copy buffer pools
- ✅ **Modular stages** (sensor, preprocess, YOLO, MobileCLIP, LLM, reasoning)
- ✅ **Automatic metrics** collection (per-stage latency, per-model stats)
- ✅ **Clean dependency graph** with circular import prevention

---

## 📁 Directory Structure

```
edgeperceive/  (23 .py files across 7 modules)
├── core/                          → Messages (6 dataclasses)
├── hardware/                      → Compute backends (3 files, 9 classes)
├── metrics/                       → Metrics collector (timing & stats)
├── preprocessors/                 → Generic preprocessing (5 files)
├── stages/                        → Pipeline stages (7 files)
└── pipeline/                      → Executable scripts (4 files)
```

---

## 🔑 Core Components

### 1. Messages (core/messages.py)
**6 typed data contracts** flowing through the pipeline:

| Message | From | To | Purpose |
|---------|------|-----|---------|
| RawFrameMessage | SensorStage | PreprocessorStage | Raw sensor input (path or array) |
| TensorMessage | PreprocessorStage | YOLOStage, MobileCLIPStage | Model-ready tensor |
| DetectionMessage | YOLOStage | ReasoningStage, draw_detections | Bounding boxes from detection |
| ClassificationMessage | MobileCLIPStage | Pipeline end | Zero-shot classification scores |
| ReasoningMessage | ReasoningStage | Pipeline end | LLM reasoning output |
| BoundingBox | (internal) | DetectionMessage | Single detection annotation |

**Key Feature**: Any stage can return `None` to stop the pipeline (early exit/gating).

### 2. Hardware Abstraction (hardware/)

**ComputeBackend** = Single entry point for all inference
- Auto-detects model format: `.onnx` → ONNXBackend, `.tflite` → TFLite/LiteRT
- **LiteRTBackend**: TFLite with QNN delegate for Hexagon NPU (or CPU fallback)
- **ONNXBackend**: ONNX models on CPU only
- **CPUBackend**: Pure CPU, no hardware acceleration
- Model caching by path, benchmarking (p50/p95/p99)

**ComputeUnit Enum**: CPU, NPU, GPU (reserved), DSP (reserved)

**PowerMonitor**: Reads battery power from sysfs or estimates by unit

### 3. Preprocessing (preprocessors/)

**Generic Framework**: `BasePreprocessor[InputT, OutputT]`
- **BufferPool**: Pre-allocated numpy arrays (zero-copy on hot path)
- **ComputeDispatcher**: Routes ops to CPU or DSP (DSP ready when SDK available)
- **ImagePreprocessor**: Concrete implementation
  - Pipeline: BGR→RGB → resize (±letterbox) → center crop → normalize
  - Config-driven (target_size, crop_size, mean/std, etc.)
  - Auto-timing and metrics logging

### 4. Stages (stages/)

**Stage** = Message transformer `process(msg) → Message | None`

**Implemented Stages**:
1. **SensorStage**: Load image from disk → RawFrameMessage
2. **PreprocessorStage**: Apply ImagePreprocessor → TensorMessage
3. **YOLOStage**: YOLO inference + NMS → DetectionMessage | None
4. **ReasoningStage**: Format detections into LLM prompt (stub) → ReasoningMessage
5. **MobileCLIPStage**: Zero-shot classification → ClassificationMessage

All stages auto-tracked by Pipeline if MetricsCollector injected.

### 5. Metrics (metrics/collector.py)

**MetricsCollector** aggregates timing and performance data:
- Per-stage latency tracking
- Per-model statistics (mean, p50, p95, p99, min, max)
- Latency budget analysis vs 5s target
- JSON export and human-readable reports

### 6. Pipeline Orchestrator (stages/base.py)

**Pipeline** runs stages sequentially:
```python
for stage in stages:
    msg = stage.run(msg, metrics=metrics)  # timing auto-tracked
    if msg is None:
        return None  # stop pipeline
return msg
```

---

## 🔀 Complete Class Inventory (47 classes)

### Messages (6 dataclasses)
- RawFrameMessage, TensorMessage, BoundingBox, DetectionMessage, ReasoningMessage, ClassificationMessage

### Hardware (9)
- ComputeUnit (enum), PowerSample, PowerMonitor
- InferenceBackend (ABC), LiteRTBackend, ONNXBackend, CPUBackend, ComputeBackend

### Preprocessing (7)
- BufferSpec, BufferPool, ComputeDispatcher, BasePreprocessor[InputT, OutputT] (ABC)
- ProcessedFrame, ImagePreprocessConfig, VideoPreprocessConfig, ImagePreprocessor

### Stages (7)
- Stage (ABC), Pipeline, SensorStage, PreprocessorStage, YOLOStage, ReasoningStage, MobileCLIPStage
- (+ CaptureStage custom in draw_detections.py)

### Metrics (3)
- InferenceRecord, PipelineRecord, MetricsCollector

---

## 📊 Data Flow Examples

### YOLO Detection Pipeline
```
Image File
  ↓ SensorStage (cv2.imread)
RawFrameMessage(frame=uint8[HxWxC], source=path)
  ↓ PreprocessorStage
  • ImagePreprocessor: BGR→RGB, resize(640×640 letterbox), normalize
TensorMessage(tensor=float32[1,C,640,640], original_size=(H,W))
  ↓ YOLOStage
  • ComputeBackend.run(yolo_model, tensor)
  • Parse YOLOv8 3-output format: boxes[1,N,4], scores[1,N], class_ids[1,N]
  • Apply NMS, scale to original image size
  • Return None if no detections above threshold
DetectionMessage(boxes=[BoundingBox...], latency_ms=X) | None
  ↓ ReasoningStage (stub)
ReasoningMessage(response=..., prompt=..., latency_ms=Y)
```

### MobileCLIP Zero-Shot Classification Pipeline
```
Image File
  ↓ SensorStage (cv2.imread)
RawFrameMessage(frame=uint8[HxWxC])
  ↓ PreprocessorStage(target_size=(256,256), no letterbox)
  • ImagePreprocessor: BGR→RGB, resize(256×256), normalize
TensorMessage(tensor=float32[1,3,256,256])
  ↓ MobileCLIPStage
  For each text_prompt in ["gun", "walking", "fun", "fight", "distress"]:
    1. tokenize(prompt) → [1,77] int64 via open_clip
    2. ComputeBackend.run(model, {
         'serving_default_args_0:0': image_tensor,    # [1,3,256,256]
         'serving_default_args_1:0': token_tensor     # [1,77]
       })
    3. outputs[0]=text_emb[512], outputs[1]=image_emb[512]
    4. cosine_similarity(image_emb, text_emb)
  Apply softmax over similarities
ClassificationMessage(label=best, confidence=score, all_scores={...})
```

---

## 🔄 Import Dependency Graph

```
core/messages.py (standalone)
    ↑ imported by: preprocessors, stages, pipeline scripts

hardware/types.py (ComputeUnit enum)
    ↑ imported by: compute_backend, preprocessors/base, stages

hardware/compute_backend.py
    ↑ imported by: YOLOStage, ReasoningStage, MobileCLIPStage

preprocessors/base.py
    ↑ imported by: preprocessors/video/video_preprocessing.py

stages/base.py (Stage, Pipeline)
    ↑ imported by: all vision stages and pipeline scripts

metrics/collector.py
    ↑ imported by: pipeline scripts
```

**Circular Import Prevention**:
- `types.py` kept separate from `compute_backend.py`
- `stages/base.py` uses `TYPE_CHECKING` for message imports

---

## 🎯 Configuration Presets

### ImagePreprocessor Configs by Model
| Model | target_size | crop_size | mean/std | letterbox |
|-------|-------------|-----------|----------|-----------|
| MobileCLIP-S2 | 256×256 | 224×224 | (0,0,0)/(1,1,1) | ✗ |
| YOLO | 640×640 | None | ImageNet | ✓ |
| MoViNet | 172×172 | None | ImageNet | ✗ |

### Model I/O Specs
- **YOLO input**: [1, C, 640, 640] float32 or [1, 640, 640, C]
- **YOLO output**: boxes [1,N,4], scores [1,N], class_ids [1,N]
- **MobileCLIP input**: image [1,3,256,256] + tokens [1,77]
- **MobileCLIP output**: text_emb [512] + image_emb [512]

---

## 🚀 Running the Pipelines

```bash
# YOLO detection with metrics
uv run python -m moment_to_action.edgeperceive.pipeline.run_yolo_pipeline \
  --image images/test.jpg --model models/yolo/model.onnx --conf 0.3 --device cpu

# YOLO with visualization
uv run python -m moment_to_action.edgeperceive.pipeline.draw_detections \
  --image images/test.jpg --model models/yolo/model.onnx --conf 0.3 --out result.jpg

# MobileCLIP classification
uv run python -m moment_to_action.edgeperceive.pipeline.run_mobileclip_pipeline \
  --image images/test.jpg --model models/mobileclip_s2/mobileclip_s2_datacompdr_last.tflite
```

---

## ⚠️ Known Limitations & Stubs

| Component | Status | Notes |
|-----------|--------|-------|
| DSP Dispatch | ✗ Stub | Currently routes to CPU; Hexagon SDK not wired |
| LLM Reasoning | ✗ Stub | ReasoningStage returns `"[LLM stub]..."`; Qwen/TinyLlama not integrated |
| Audio Preprocessing | ⚠️ In Progress | YAMNet mentioned but not implemented |
| Real Sensors | ✗ Stub | SensorStage loads from disk, not camera/device |
| Batching | ✗ Not Supported | Single-frame only |
| Multi-Modal | ✗ Not Supported | No audio+video fusion yet |

---

## 💡 Design Patterns & Best Practices

1. **Message-Driven**: Typed flow between stages; no shared mutable state
2. **Composition Over Inheritance**: Stages composed into Pipeline
3. **Generic Preprocessing**: `BasePreprocessor[InputT, OutputT]` avoids code duplication
4. **Hardware Abstraction**: Single entry point (ComputeBackend) for all inference
5. **Optional Instrumentation**: MetricsCollector injected, not embedded
6. **Pre-allocation**: BufferPool for zero-copy on hot path
7. **Graceful Degradation**: Delegate failures fall back to CPU
8. **Early Exit**: Return `None` to stop pipeline (gating expensive computation)

---

## 🔧 Refactoring Priorities

### Tier 1 (High Impact)
1. **Extract Model Classes**: Move token padding, embedding logic to reusable classes
2. **Real DSP Backend**: Implement Hexagon SDK dispatch
3. **Audio Pipeline**: Mirror vision preprocessing for audio resampling/windowing
4. **Unified Configuration**: Single config file for all preprocessing params
5. **Plugin System**: Dynamic stage registration

### Tier 2 (Medium Impact)
6. **Batching Support**: Handle multiple images per run
7. **Sensor Adapters**: Camera, audio device, simulator abstractions
8. **Model Registry**: Named model loading + versioning
9. **Quantization Tracking**: Instrument FQ/QAT in metrics
10. **Pipeline Builder**: DSL or config-driven construction

### Tier 3 (Nice to Have)
11. Multi-modal fusion (audio+video)
12. Real-time streaming mode
13. Model optimization tracking
14. A/B testing framework

---

## 📌 Key Constants

```python
LATENCY_BUDGET_MS = 5000                # Target total latency (5 seconds)
COCO_LABELS = (80 strings)              # YOLOStage class names
ComputeUnit = {CPU, NPU, GPU, DSP}      # Available compute platforms
Power estimates (mW) = {                # PowerMonitor fallback estimates
    CPU: 300, NPU: 500, GPU: 800, DSP: 150
}
```

---

## ✅ Validation Status

| Model | Task | Format | Status |
|-------|------|--------|--------|
| YOLOv8 | Object detection | ONNX | ✓ Validated |
| MobileCLIP-S2 | Zero-shot classification | TFLite | ✓ Validated |
| YAMNet | Audio event detection | TFLite | ⚠️ In progress |
| Qwen 0.8B / TinyLlama | Scene reasoning | TFLite | ⚠️ In progress |

---

## 🎓 Getting Started with Refactoring

1. **Start Here**: `core/messages.py`
   - Understand message contracts
   - Identify message flow

2. **Then Study**: `hardware/compute_backend.py`
   - How hardware abstraction works
   - How backends are selected

3. **Next**: `preprocessors/base.py` and `preprocessors/video/video_preprocessing.py`
   - Generic preprocessing framework
   - ImagePreprocessor concrete implementation

4. **Then**: `stages/base.py` and `stages/vision_stages/`
   - Pipeline orchestration
   - Individual stage implementations

5. **Finally**: `pipeline/*.py`
   - How stages are composed
   - How to run end-to-end

---

## 📚 Additional Documentation

- `README.md` in edgeperceive/ - High-level overview
- Docstrings in each file - Detailed implementation notes
- Type hints throughout - Self-documenting interfaces

---

## 🤝 Contributing

When refactoring:
1. Maintain message-driven architecture
2. Preserve dependency graph structure
3. Keep compute abstraction layer isolated
4. Don't embed metrics collection in stages
5. Use TYPE_CHECKING for circular import prevention
6. Pre-allocate buffers when adding new preprocessors
7. Test with both CPU and NPU backends (when available)

---

**Last Updated**: [Exploration completed]
**Total Files Analyzed**: 23
**Total Classes**: 47
**Total Methods/Properties**: 100+
