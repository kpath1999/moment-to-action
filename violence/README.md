# Violence Detection System

A unified deep learning system for detecting violence in videos using MobileNetV2 + BiLSTM architecture.

## Features

- **Unified Pipeline**: Combines dataset parsing, model training, testing, and inference
- **Command-line Interface**: Easy-to-use CLI with multiple operation modes
- **Comprehensive Logging**: Detailed logs for all operations
- **Batch Processing**: Process multiple video directories at once
- **Model Evaluation**: Automatic generation of metrics, plots, and reports

## Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Temporal Model**: Bidirectional LSTM
- **Input**: Sequences of 16 frames (64x64 pixels)
- **Output**: Binary classification (Violence / Non-Violence)

## Setup

### 1. Install Conda Environment

Run the setup script to create the `vio` conda environment:

```bash
cd /Users/kausar/Documents/moment-to-action/violence
bash setup_env.sh
```

This script will:
- Remove all conda environments except `base`
- Create a new `vio` environment with Python 3.10
- Install all required dependencies

### 2. Activate Environment

```bash
conda activate vio
```

## Usage

The system has three operation modes. Use the wrapper script `run_violence_detection.sh` to ensure the correct conda environment is used:

### Mode 1: Training Only

Train a new model on the violence dataset:

```bash
./run_violence_detection.sh \
  --mode train \
  --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset" \
  --output ./output \
  --test-split 0.1
```

### Mode 2: Inference Only

Run inference on unseen videos using a trained model:

```bash
./run_violence_detection.sh \
  --mode inference \
  --model ./output/violence_detection_model.h5 \
  --test-dirs "/Volumes/KAUSAR/rwf2000/RWF-2000/train" "/Volumes/KAUSAR/rwf2000/RWF-2000/val" \
  --output ./output \
  --analyze
```

### Mode 3: Full Pipeline (Recommended)

Train, evaluate, and run inference in one go:

```bash
./run_violence_detection.sh \
  --mode full \
  --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset" \
  --test-dirs "/Volumes/KAUSAR/rwf2000/RWF-2000/train" "/Volumes/KAUSAR/rwf2000/RWF-2000/val" \
  --output ./output \
  --analyze
```

### Mode 4: Analyze Results

Analyze previously generated inference results against ground truth labels:

```bash
./run_violence_detection.sh \
  --mode analyze \
  --output ./output
```

This mode finds the most recent inference results file and evaluates predictions against RWF-2000 ground truth labels extracted from directory structure.

**Alternative:** If you have issues with the wrapper script, you can run directly:

```bash
conda activate vio
$(conda info --base)/envs/vio/bin/python violence_detection.py --mode full --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset"
```

## Command-line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mode` | Yes | - | Operation mode: `train`, `inference`, `full`, or `analyze` |
| `--dataset` | For train/full | `/Volumes/KAUSAR/kaggle/Real Life Violence Dataset` | Path to training dataset |
| `--test-dirs` | For inference/full | `/Volumes/KAUSAR/rwf2000/RWF-2000/train` `/Volumes/KAUSAR/rwf2000/RWF-2000/val` | Directories with test videos |
| `--model` | For inference | - | Path to saved model file |
| `--output` | No | `./output` | Output directory |
| `--cache` | No | `./cache` | Dataset cache directory |
| `--force-recreate` | No | False | Force dataset recreation |
| `--test-split` | No | 0.1 | Test set split ratio |
| `--analyze` | No | False | Evaluate predictions against ground truth (for inference/full modes) |

## Output Files

The system generates several output files:

### Training Mode
- `violence_detection_model.h5` - Trained Keras model
- `training_history.json` - Training metrics per epoch
- `loss_plot.png` - Training/validation loss visualization
- `accuracy_plot.png` - Training/validation accuracy visualization
- `confusion_matrix.png` - Test set confusion matrix
- `classification_report.txt` - Detailed performance metrics

### Inference Mode
- `inference_results_YYYYMMDD_HHMMSS.json` - Predictions for all videos

### RWF-2000 Evaluation (with --analyze flag)
- `rwf2000_analysis.txt` - Detailed evaluation report including:
  - Overall metrics (accuracy, precision, recall, F1 score)
  - Confusion matrix statistics
  - List of all misclassified videos with confidence scores
- `rwf2000_confusion_matrix.png` - Visual confusion matrix heatmap

### Logs
- `violence_detection_YYYYMMDD_HHMMSS.log` - Comprehensive execution log

## Dataset Structure

### Training Dataset Structure

Expected directory structure for training dataset:

```
Real Life Violence Dataset/
├── Violence/
│   ├── V_1.mp4
│   ├── V_2.mp4
│   └── ...
└── NonViolence/
    ├── NV_1.mp4
    ├── NV_2.mp4
    └── ...
```

### RWF-2000 Dataset Structure

Expected directory structure for RWF-2000 evaluation dataset:

```
RWF-2000/
├── train/
│   ├── Train_Fight/
│   │   ├── video_1.avi
│   │   └── ...
│   └── Train_NonFight/
│       ├── video_1.avi
│       └── ...
└── val/
    ├── Val_Fight/
    │   ├── video_1.avi
    │   └── ...
    └── Val_NonFight/
        ├── video_1.avi
        └── ...
```

The analyze mode automatically extracts ground truth labels from the directory names (Fight/NonFight).

## Inference Results Format

The inference results JSON contains:

```json
[
  {
    "video_path": "/path/to/video.mp4",
    "video_name": "video.mp4",
    "total_frames": 150,
    "predicted_class": "Violence",
    "confidence": 0.9234
  },
  ...
]
```

## RWF-2000 Label Extraction

When using `--analyze` mode with RWF-2000 dataset, ground truth labels are automatically extracted from the directory path:

- Directories containing **"Fight"** (but not "NonFight") → labeled as **Violence**
- Directories containing **"NonFight"** → labeled as **NonViolence**

Examples:
- `/Volumes/KAUSAR/rwf2000/RWF-2000/train/Train_Fight/video1.avi` → Violence
- `/Volumes/KAUSAR/rwf2000/RWF-2000/val/Val_NonFight/video2.avi` → NonViolence

The system then compares predicted labels against these ground truth labels to compute accuracy, precision, recall, F1 score, and generate confusion matrices.

## System Requirements

- **Python**: 3.10
- **TensorFlow**: 2.15.0
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~2GB for model and cached data

## Troubleshooting

### Issue: "Directory not found"
- Verify that the dataset and test directories exist
- Check that paths are correctly escaped (use quotes for paths with spaces)

### Issue: "Insufficient frames in video"
- Some videos may be too short or corrupted
- These are automatically skipped with a warning in the logs

### Issue: CUDA/GPU errors
- The system works on CPU by default
- For GPU support, install `tensorflow-gpu` instead

## Quick Reference

**Training a new model:**
```bash
./run_violence_detection.sh --mode train --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset" --output ./output
```

**Running inference on RWF-2000:**
```bash
./run_violence_detection.sh --mode inference --model ./output/violence_detection_model.h5 --test-dirs "/Volumes/KAUSAR/rwf2000/RWF-2000/train" "/Volumes/KAUSAR/rwf2000/RWF-2000/val" --output ./output --analyze
```

**Analyzing existing results:**
```bash
./run_violence_detection.sh --mode analyze --output ./output
```
