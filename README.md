# Workspace Organization

This workspace contains AI/ML development projects and tools for embedded and edge devices.

## Directory Structure

### 📁 `projects/`
Active development projects and code repositories.
- `edge-impulse/` - Edge Impulse model inference scripts and configurations
- `yolo/` - YOLO object detection implementations and NPU optimizations
- `aihub-demo/` - AI Hub demonstration projects
- `aihub-npu/` - NPU-accelerated AI applications
- `nexa-sdk/` - Nexa SDK integration and examples
- `mta/` - Additional project workspace

### 📁 `models/`
Trained models, weights, and model artifacts.
- `onnx/` - ONNX format models and labels
- `rubikpi/` - RUBIKPi model collection (TFLite, DLC, etc.)
- `archives/` - Compressed model packages

### 📁 `test-data/`
Test images and datasets for model validation and inference testing.

### 📁 `experiments/`
Experimental scripts, benchmarks, and profiling results.
- Inference benchmarking tools
- Accelerator profiling scripts
- Performance analysis results

### 📁 `export_assets/`
Exported model artifacts from various build/export processes.
- Quantized models for specific hardware targets
- Model metadata and versioning information

### 📁 `tools/`
SDKs, frameworks, and development tools.
- `qairt/` - Qualcomm AI Runtime toolchain
- `rubikpi-script/` - RUBIKPi utility scripts
- `miniconda3/` - Python package manager

### 📁 `venvs/`
Python virtual environments for dependency isolation.
- `litert_venv/` - Main LiteRT/TensorFlow Lite environment
- `.venv/` - General purpose virtual environment
- `.venv-onnxruntime-demo/` - ONNX Runtime specific environment

### 📁 `archives/`
Downloaded archives and compressed packages (SDKs, model zips, etc.)

### 🔗 `litert_venv` (symlink)
Convenience symlink to `venvs/litert_venv/` for backward compatibility.

## Quick Start

```bash
# Activate virtual environment
source venvs/litert_venv/bin/activate

# Run YOLO inference
python projects/yolo/yolo_npu.py

# Run Edge Impulse model
python projects/edge-impulse/inference_onnx.py
```
