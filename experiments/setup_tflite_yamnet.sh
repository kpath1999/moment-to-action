#!/bin/bash
# Setup TensorFlow Lite YamNet (lightweight, works on ARM)

echo "================================"
echo "Setting up TFLite YamNet"
echo "================================"
echo ""

# Install TFLite Runtime (much lighter than full TensorFlow)
echo "Installing TFLite Runtime..."
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Install other dependencies
pip3 install soundfile resampy numpy

# Download YamNet TFLite model
echo ""
echo "Downloading YamNet TFLite model..."
if [ ! -f "yamnet.tflite" ]; then
    wget -O yamnet.tflite https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite
    echo "✓ Model downloaded: yamnet.tflite"
else
    echo "Model already exists"
fi

# Download class names
echo ""
echo "Downloading class names..."
if [ ! -f "yamnet_class_map.csv" ]; then
    wget -O yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv
    echo "✓ Class map downloaded"
else
    echo "Class map already exists"
fi

echo ""
echo "================================"
echo "Setup Complete!"
echo "================================"
echo ""
echo "Test with:"
echo "  python3 yamnet_tflite.py --model yamnet.tflite --audio your_audio.wav --classes yamnet_class_map.csv"
echo ""
