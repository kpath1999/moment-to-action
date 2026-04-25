#!/bin/bash
# post_uv_sync.sh
# Download and install onnxruntime_qnn wheel on Linux/aarch64 (Rubik Pi)

set -e

if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "This script should not be run on macOS. Exiting."
    exit 0
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This script is intended for aarch64 (ARM64) platforms only. Exiting."
    exit 0
fi

WHEEL="onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl"
URL="https://cdn.edgeimpulse.com/qc-ai-docs/wheels/$WHEEL"

if [[ ! -f "$WHEEL" ]]; then
    echo "Downloading $WHEEL..."
    wget "$URL"
else
    echo "$WHEEL already exists, skipping download."
fi

echo "Installing $WHEEL with uv pip..."
uv pip install "$WHEEL"
