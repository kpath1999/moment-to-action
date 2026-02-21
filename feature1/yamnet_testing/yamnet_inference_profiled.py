#!/usr/bin/env python3
"""
YAMNet Inference with Power Profiling
Compares CPU vs NPU performance and power consumption
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import soundfile as sf
import librosa
import csv
import os
import time
import sys
import json
from pathlib import Path

# Import power profiler
try:
    from power_profiler import PowerProfiler
    POWER_PROFILING_AVAILABLE = True
except ImportError:
    print("⚠️  power_profiler.py not found - power profiling disabled")
    POWER_PROFILING_AVAILABLE = False

def curr_ms():
    return round(time.time() * 1000)

# Configuration
use_npu = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-npu' else False
MODEL_PATH = "yamnet_quantized.tflite"
LABELS_PATH = "yamnet_class_map.csv"
AUDIO_PATH = "test_audio.wav"
NUM_WARMUP_RUNS = 5
NUM_TIMED_RUNS = 100  # More runs for better statistics

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
    waveform, sr = sf.read(audio_path, dtype='float32')
    
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
    elif len(waveform) > target_length:
        waveform = waveform[:target_length]
    
    return waveform

print(f"\n{'='*70}")
print(f"  YAMNet Inference - {'NPU' if use_npu else 'CPU'} Mode")
print(f"{'='*70}\n")

# Load delegate if using NPU
experimental_delegates = []
if use_npu:
    print("Loading NPU delegate...")
    try:
        experimental_delegates = [
            load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
        ]
        print("✓ NPU delegate loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load NPU delegate: {e}")
        print("   Falling back to CPU...\n")
        use_npu = False

# Load TFLite model
print(f"Loading model: {MODEL_PATH}")
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=experimental_delegates
)
interpreter.allocate_tensors()
print("✓ Model loaded\n")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Information:")
print(f"  Input shape:  {input_details[0]['shape']}")
print(f"  Input dtype:  {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Check quantization
is_quantized = input_details[0]['dtype'] in [np.uint8, np.int8]
print(f"  Quantized:    {is_quantized}")
if is_quantized:
    in_scale, in_zero = input_details[0]['quantization']
    out_scale, out_zero = output_details[0]['quantization']
    print(f"  Input quantization:  scale={in_scale:.6f}, zero_point={in_zero}")
    print(f"  Output quantization: scale={out_scale:.6f}, zero_point={out_zero}")
print()

# Load and preprocess audio
print(f"Loading audio: {AUDIO_PATH}")
waveform = preprocess_audio(AUDIO_PATH)
print(f"  Waveform shape: {waveform.shape}")
print(f"  Waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]\n")

# Prepare input data
input_data = waveform.astype(np.float32)

# Quantize input if needed
if is_quantized:
    scale, zero_point = input_details[0]['quantization']
    input_data = (input_data / scale + zero_point).astype(input_details[0]['dtype'])

# Ensure 1D
if len(input_data.shape) > 1:
    input_data = input_data.squeeze()

# Set tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# ============================================================================
# WARMUP RUNS
# ============================================================================
print(f"Warmup: Running {NUM_WARMUP_RUNS} iterations...")
for i in range(NUM_WARMUP_RUNS):
    interpreter.invoke()
print("✓ Warmup complete\n")

# ============================================================================
# TIMED RUNS WITH POWER PROFILING
# ============================================================================

# Initialize power profiler
profiler = None
if POWER_PROFILING_AVAILABLE:
    profiler = PowerProfiler()
    if profiler.power_paths:
        print(f"⚡ Power profiling enabled")
        print(f"   Monitoring: {', '.join(profiler.power_paths.keys())}\n")
    else:
        print("⚠️  No power monitoring available - timing only\n")
        profiler = None
else:
    print("⚠️  Power profiling not available - timing only\n")

print(f"Benchmark: Running {NUM_TIMED_RUNS} iterations...")

# Start power profiling
if profiler:
    profiler.start_profiling(sample_interval_ms=10)  # Sample every 10ms

# Timed inference runs
inference_times = []
start_time = time.time()

for i in range(NUM_TIMED_RUNS):
    iter_start = time.perf_counter()
    interpreter.invoke()
    iter_end = time.perf_counter()
    inference_times.append((iter_end - iter_start) * 1000)  # Convert to ms

end_time = time.time()

# Stop power profiling
if profiler:
    profiler.stop_profiling()

# ============================================================================
# RESULTS
# ============================================================================

# Get output for verification
output_data = interpreter.get_tensor(output_details[0]['index'])

# Dequantize output if needed
if is_quantized:
    scale, zero_point = output_details[0]['quantization']
    output_data = (output_data.astype(np.float32) - zero_point) * scale

# Get top predictions
def get_top_predictions(scores, labels, top_k=5):
    if len(scores.shape) > 1:
        if scores.shape[0] == 1:
            scores = scores[0]
        else:
            scores = np.mean(scores, axis=0)
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    predictions = []
    for idx in top_indices:
        predictions.append((labels[idx], scores[idx]))
    return predictions

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*70)
print(f"  INFERENCE BENCHMARK RESULTS - {'NPU' if use_npu else 'CPU'}")
print("="*70)

# Timing statistics
import statistics
mean_time = statistics.mean(inference_times)
median_time = statistics.median(inference_times)
stdev_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
min_time = min(inference_times)
max_time = max(inference_times)

print(f"\nTiming Statistics ({NUM_TIMED_RUNS} runs):")
print(f"  Mean:       {mean_time:.3f} ms")
print(f"  Median:     {median_time:.3f} ms")
print(f"  Std Dev:    {stdev_time:.3f} ms")
print(f"  Min:        {min_time:.3f} ms")
print(f"  Max:        {max_time:.3f} ms")
print(f"  Throughput: {1000/mean_time:.1f} inferences/sec")

# Power statistics
power_stats = None
if profiler:
    power_stats = profiler.get_statistics()
    
    if power_stats and "error" not in power_stats:
        print(f"\nPower Consumption:")
        
        # Print battery power if available
        for metric in ["battery_power_mw", "battery_power_uw", "cpu_power_uw"]:
            if metric in power_stats:
                data = power_stats[metric]
                print(f"  {metric}:")
                print(f"    Mean:   {data['mean']:>10.2f} {data['unit']}")
                print(f"    Median: {data['median']:>10.2f} {data['unit']}")
                print(f"    StdDev: {data['stdev']:>10.2f} {data['unit']}")
        
        # Calculate energy consumed (mJ or μJ)
        total_time_sec = end_time - start_time
        if "battery_power_mw" in power_stats:
            avg_power_mw = power_stats["battery_power_mw"]["mean"]
            energy_mj = avg_power_mw * total_time_sec
            print(f"\n  Energy per inference: {energy_mj / NUM_TIMED_RUNS:.3f} mJ")
        elif "battery_power_uw" in power_stats:
            avg_power_uw = power_stats["battery_power_uw"]["mean"]
            energy_uj = avg_power_uw * total_time_sec
            print(f"\n  Energy per inference: {energy_uj / NUM_TIMED_RUNS:.3f} μJ")

# Top predictions
print(f"\nTop-5 Predictions (sample inference):")
top_predictions = get_top_predictions(output_data, labels)
for label, score in top_predictions:
    print(f"  {label:40s}: {score:.4f}")

print("\n" + "="*70)

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    "config": {
        "device": "thundercomm_rubix_pi3",
        "model": MODEL_PATH,
        "compute_unit": "NPU" if use_npu else "CPU",
        "quantized": is_quantized,
        "num_runs": NUM_TIMED_RUNS,
    },
    "timing": {
        "mean_ms": mean_time,
        "median_ms": median_time,
        "stdev_ms": stdev_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "throughput_fps": 1000 / mean_time,
    },
    "power": power_stats if power_stats else {},
}

# Save to JSON
result_filename = f"yamnet_benchmark_{'npu' if use_npu else 'cpu'}.json"
with open(result_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to {result_filename}")

# Export power profile CSV if available
if profiler and profiler.power_samples:
    csv_filename = f"yamnet_power_profile_{'npu' if use_npu else 'cpu'}.csv"
    profiler.export_csv(csv_filename)
    print(f"✓ Power profile saved to {csv_filename}")

print()
