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
  --test-dirs "/Volumes/KAUSAR/1-1004" "/Volumes/KAUSAR/videos/" \
  --output ./output
```

### Mode 3: Full Pipeline (Recommended)

Train, evaluate, and run inference in one go:

```bash
./run_violence_detection.sh \
  --mode full \
  --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset" \
  --test-dirs "/Volumes/KAUSAR/1-1004" "/Volumes/KAUSAR/videos/" \
  --output ./output
```

**Alternative:** If you have issues with the wrapper script, you can run directly:

```bash
conda activate vio
$(conda info --base)/envs/vio/bin/python violence_detection.py --mode full --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset"
```

## Command-line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mode` | Yes | - | Operation mode: `train`, `inference`, or `full` |
| `--dataset` | For train/full | `/Volumes/KAUSAR/kaggle/Real Life Violence Dataset` | Path to training dataset |
| `--test-dirs` | For inference/full | `/Volumes/KAUSAR/1-1004` `/Volumes/KAUSAR/videos/` | Directories with test videos |
| `--model` | For inference | - | Path to saved model file |
| `--output` | No | `./output` | Output directory |
| `--cache` | No | `./cache` | Dataset cache directory |
| `--force-recreate` | No | False | Force dataset recreation |
| `--test-split` | No | 0.1 | Test set split ratio |

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

### Logs
- `violence_detection_YYYYMMDD_HHMMSS.log` - Comprehensive execution log

## Dataset Structure

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

## Model Performance

# Violence Detection

Minimal workflow to train, evaluate, and run inference on violence videos.

## Setup
- Run once: `bash setup_env.sh` (creates `vio` conda env and installs deps)
- Use wrapper: `./run_violence_detection.sh` (activates env automatically)

## Run
- Full pipeline: `./run_violence_detection.sh --mode full --dataset "/Volumes/KAUSAR/kaggle/Real Life Violence Dataset" --test-dirs "/Volumes/KAUSAR/1-1004" "/Volumes/KAUSAR/videos/" --output ./output`
- Train only: `./run_violence_detection.sh --mode train --dataset "<dataset>" --output ./output`
- Inference only: `./run_violence_detection.sh --mode inference --model ./output/violence_detection_model.h5 --test-dirs "<dir1>" "<dir2>" --output ./output`
- Analyze latest results: `./run_violence_detection.sh --mode analyze --output ./output`
- Inference with analysis: add `--analyze` flag to inference or full mode

## Arguments
- `--mode`: train | inference | full
- `--dataset`: dataset root (needs Violence/ and NonViolence/ subfolders)
- `--test-dirs`: one or more folders with .mp4 files
- `--model`: required for inference mode
- `--output`: where to write models/results (default `./output`)
- `--cache`: dataset cache dir (default `./cache`)
- `--force-recreate`: rebuild cache
- `--test-split`: test fraction (default 0.1)
- `--analyze`: evaluate predictions against weak labels (XD-Violence: A/B1/B2/B4→NonViolence, B5/B6/G→Violence)

## Outputs
- `output/violence_detection_model.h5`
- `output/training_history.json`, `loss_plot.png`, `accuracy_plot.png`
- `output/confusion_matrix.png`, `classification_report.txt`
- `output/inference_results_YYYYMMDD_HHMMSS.json`
- `output/weak_label_analysis.txt`, `weak_label_confusion_matrix.png` (if `--analyze` used)
- Logs: `violence_detection_*.log` (in this folder)
