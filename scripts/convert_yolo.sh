#!/usr/bin/env bash
# Convert YOLOv8 to a quantized DLC + three context binaries via Qualcomm AI Hub.
#
# Requires QAI_HUB_API_TOKEN in the environment and the [host] extra:
#   uv sync --extra host
#
# Usage:
#   ./scripts/convert_yolo.sh -o ./out/ -c ./calib/
#   ./scripts/convert_yolo.sh -o ./out/ -c ./calib/ --chipset qualcomm-qcs6490 --precision w8a8

set -euo pipefail

OUTPUT_DIR=""
CALIB_DIR=""
CHIPSET="qualcomm-qcs6490"
PRECISION="w8a8"

usage() {
    echo "Usage: $0 -o OUTPUT_DIR -c CALIB_DIR [--chipset CHIPSET] [--precision PRECISION]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        -c|--calibration-dir) CALIB_DIR="$2";  shift 2 ;;
        --chipset)             CHIPSET="$2";    shift 2 ;;
        --precision)           PRECISION="$2";  shift 2 ;;
        *) usage ;;
    esac
done

[[ -z "$OUTPUT_DIR" || -z "$CALIB_DIR" ]] && usage

if [[ -z "${QAI_HUB_API_TOKEN:-}" && -z "${QAI_HUB_API_KEY:-}" ]]; then
    echo "Error: QAI_HUB_API_TOKEN is not set." >&2
    exit 1
fi

m2a model convert-aihub yolo_v8 \
    -o "$OUTPUT_DIR" \
    --calibration-dir "$CALIB_DIR" \
    --chipset "$CHIPSET" \
    --precision "$PRECISION"
