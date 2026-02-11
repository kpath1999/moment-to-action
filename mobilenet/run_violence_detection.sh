#!/bin/bash
# Wrapper script to run violence detection in the correct conda environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda
eval "$(conda shell.bash hook)"

# Activate the vio environment
conda activate vio

# Use the conda environment's python explicitly
CONDA_PYTHON="$(conda info --base)/envs/vio/bin/python"

# Run the script with all arguments passed to this wrapper
"$CONDA_PYTHON" "$SCRIPT_DIR/violence_detection.py" "$@"
