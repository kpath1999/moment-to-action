import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import soundfile as sf
import librosa
import csv
import os
import time
import sys

def curr_ms():
    return round(time.time() * 1000)

# Configuration
use_npu = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-npu' else False
MODEL_PATH = "yamnet_int8_quantized.tflite"
LABELS_PATH = "yamnet_class_map.csv"
AUDIO_PATH = "test_audio.wav"  # Your audio file

# Load class labels
def load_yamnet_labels(csv_path):
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['index'])] = row['display_name']
    return labels

labels = load_yamnet_labels(LABELS_PATH)

# Audio preprocessing for YAMNet
def preprocess_audio(audio_path, target_sr=16000, target_length=15600):
    """
    YAMNet expects:
    - 16 kHz sample rate
    - Mono channel
    - Waveform in range [-1.0, 1.0]
    - Exactly 15600 samples (0.975 seconds)
    """
    # Load audio file
    waveform, sr = sf.read(audio_path, dtype='float32')
    
    # Resample to 16kHz if needed
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Pad or trim to exact length
    if len(waveform) < target_length:
        # Pad with zeros
        waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
    elif len(waveform) > target_length:
        # Trim to target length
        waveform = waveform[:target_length]
    
    return waveform

# Load delegate if using NPU
experimental_delegates = []
if use_npu:
    experimental_delegates = [
        load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
    ]

# Load TFLite model
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=experimental_delegates
)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output dtype: {output_details[0]['dtype']}")

# Load and preprocess audio
waveform = preprocess_audio(AUDIO_PATH)
print(f"Waveform shape: {waveform.shape}")
print(f"Waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]")

# Prepare input data - YAMNet expects 1D array [15600]
input_data = waveform.astype(np.float32)

# For quantized models, you may need to quantize the input
if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    print(f"Quantizing input: scale={scale}, zero_point={zero_point}")
    input_data = (input_data / scale + zero_point).astype(input_details[0]['dtype'])

# Ensure it's 1D
if len(input_data.shape) > 1:
    input_data = input_data.squeeze()

print(f"Final input shape: {input_data.shape}")

# Set tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)

# Warmup run
interpreter.invoke()

# Timed runs
start = curr_ms()
for i in range(10):
    interpreter.invoke()
end = curr_ms()

# Get predictions
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"Raw output shape: {output_data.shape}")

# Dequantize output if needed
if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
    scale, zero_point = output_details[0]['quantization']
    print(f"Dequantizing output: scale={scale}, zero_point={zero_point}")
    output_data = (output_data.astype(np.float32) - zero_point) * scale

# Get top predictions
def get_top_predictions(scores, labels, top_k=5):
    """Get top-k predictions with their scores"""
    # If output has multiple dimensions, handle accordingly
    if len(scores.shape) > 1:
        # If it's [batch, classes], take first batch
        if scores.shape[0] == 1:
            scores = scores[0]
        # If it's [time_frames, classes], average across time
        else:
            scores = np.mean(scores, axis=0)
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    predictions = []
    for idx in top_indices:
        predictions.append((labels[idx], scores[idx]))
    return predictions

# Display results
print(f"\n{'='*50}")
print(f"Running on {'NPU' if use_npu else 'CPU'}")
print(f"{'='*50}")
print("\nTop-5 predictions:")
top_predictions = get_top_predictions(output_data, labels)
for label, score in top_predictions:
    print(f"  {label:40s}: {score:.4f}")

print(f'\nInference time (average): {(end - start) / 10:.1f}ms per audio clip')
