#!/bin/bash
# Setup script for Violence Detection conda environment

set -e  # Exit on error

echo "=================================="
echo "Violence Detection Environment Setup"
echo "=================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo ""
echo "Step 1: Cleaning up existing conda environments (keeping only 'base')..."
echo ""

# Get list of all environments except base
envs=$(conda env list | grep -v "^#" | grep -v "^base" | awk '{print $1}' | grep -v "^$")

if [ -z "$envs" ]; then
    echo "No environments to remove (only 'base' exists)"
else
    for env in $envs; do
        echo "Removing environment: $env"
        conda env remove -n "$env" -y
    done
    echo "All non-base environments removed successfully"
fi

echo ""
echo "Step 2: Creating new 'vio' environment with Python 3.10..."
echo ""

conda create -n vio python=3.10 -y

echo ""
echo "Step 3: Installing required packages..."
echo ""

# Activate environment and install packages
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vio

# Install pip packages
pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "=================================="
echo "Setup completed successfully!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate vio"
echo ""
echo "To run the violence detection system:"
echo "  python violence_detection.py --mode full --dataset /Volumes/KAUSAR/kaggle/Real\\ Life\\ Violence\\ Dataset"
echo ""
