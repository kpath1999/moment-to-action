#!/usr/bin/env bash
# Download and convert YOLOv8 ONNX → quantized DLC for Qualcomm targets.
#
# Requires QAIRT SDK (run `m2a qairt install` first) and a directory of
# calibration images for INT8 quantization.
#
# Usage:
#   ./scripts/convert_yolo.sh -o ./out/ -c ./calib/
#   ./scripts/convert_yolo.sh -o ./out/ -c ./calib/ --variant default

set -euo pipefail

OUTPUT_DIR=""
CALIB_DIR=""
VARIANT="default"

usage() {
    echo "Usage: $0 -o OUTPUT_DIR -c CALIB_DIR [--variant VARIANT]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -c|--calibration-dir) CALIB_DIR="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$OUTPUT_DIR" || -z "$CALIB_DIR" ]] && usage

m2a model download yolo_v8 --variant "$VARIANT"
m2a model convert yolo_v8 \
    -o "$OUTPUT_DIR" \
    --variant "$VARIANT" \
    --calibration-dir "$CALIB_DIR"
