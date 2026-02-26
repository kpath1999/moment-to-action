import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import soundfile as sf
import librosa
import csv
import os
import time
import sys

def curr_us():
    """Return current time in microseconds for higher precision"""
    return time.perf_counter() * 1000000

class PerformanceTimer:
    """Track timing for different pipeline stages"""
    def __init__(self):
        self.times = {
            'audio_loading': [],
            'preprocessing': [],
            'inference': [],
            'postprocessing': [],
            'end_to_end': []
        }
    
    def add(self, stage, duration_us):
        """Add a timing measurement (in microseconds)"""
        self.times[stage].append(duration_us / 1000)  # Convert to ms
    
    def print_summary(self):
        """Print timing summary"""
        print(f"\n{'='*70}")
        print("PERFORMANCE BREAKDOWN")
        print(f"{'='*70}")
        print(f"{'Stage':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-"*70)
        
        for stage, times in self.times.items():
            if len(times) > 0:
                mean = np.mean(times)
                std = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                print(f"{stage:<20} {mean:>10.2f}ms {std:>10.2f}ms {min_time:>10.2f}ms {max_time:>10.2f}ms")
        
        print("-"*70)
        
        # Calculate percentages
        if len(self.times['end_to_end']) > 0:
            total = np.mean(self.times['end_to_end'])
            print(f"\nPercentage of total time:")
            for stage in ['audio_loading', 'preprocessing', 'inference', 'postprocessing']:
                if len(self.times[stage]) > 0:
                    stage_mean = np.mean(self.times[stage])
                    percentage = (stage_mean / total) * 100
                    print(f"  {stage:<20} {percentage:>5.1f}%")
        print(f"{'='*70}")

# Configuration
use_npu = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-npu' else False
MODEL_PATH = "yamnet_quantized.tflite"
LABELS_PATH = "yamnet_class_map.csv"
AUDIO_PATH = "test_audio.wav"  # Your audio file
NUM_ITERATIONS = 10  # Number of times to run for averaging

# Initialize performance timer
perf_timer = PerformanceTimer()

# Load class labels
def load_yamnet_labels(csv_path):
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['index'])] = row['display_name']
    return labels

print("Loading labels...")
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
    t_start = curr_us()
    
    # Load audio file
    waveform, sr = sf.read(audio_path, dtype='float32')
    
    t_load = curr_us()
    perf_timer.add('audio_loading', t_load - t_start)
    
    # Preprocessing start
    t_prep_start = curr_us()
    
    # Resample to 16kHz if needed
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Normalize to [-1, 1]
    if np.abs(waveform).max() > 0:
        waveform = waveform / np.abs(waveform).max()
    
    # Pad or trim to exact length
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
    elif len(waveform) > target_length:
        waveform = waveform[:target_length]
    
    t_prep_end = curr_us()
    perf_timer.add('preprocessing', t_prep_end - t_prep_start)
    
    return waveform

# Load delegate if using NPU
print(f"\nLoading model ({'NPU' if use_npu else 'CPU'})...")
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

# Check if model is quantized
if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    print(f"Model is QUANTIZED: scale={scale:.6f}, zero_point={zero_point}")
else:
    print(f"Model is FLOAT32 (NPU cannot be used)")

# Load and preprocess audio (first time - includes loading time)
print(f"\nLoading and preprocessing audio...")
waveform = preprocess_audio(AUDIO_PATH)
print(f"Waveform shape: {waveform.shape}")
print(f"Waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]")

# Prepare input data - YAMNet expects 1D array [15600]
input_data = waveform.astype(np.float32)

# For quantized models, quantize the input
if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
    scale, zero_point = input_details[0]['quantization']
    input_data = (input_data / scale + zero_point).astype(input_details[0]['dtype'])

# Ensure it's 1D
if len(input_data.shape) > 1:
    input_data = input_data.squeeze()

print(f"Final input shape: {input_data.shape}")
print(f"Final input dtype: {input_data.dtype}")

# Warmup run
print(f"\nWarming up (5 iterations)...")
interpreter.set_tensor(input_details[0]['index'], input_data)
for _ in range(5):
    interpreter.invoke()

# Get top predictions function
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

# Run benchmarking
print(f"\nBenchmarking ({NUM_ITERATIONS} iterations)...")
print("This measures the complete end-to-end pipeline for each iteration")

for iteration in range(NUM_ITERATIONS):
    # Measure end-to-end time (from setting input to getting predictions)
    t_e2e_start = curr_us()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Inference
    t_inf_start = curr_us()
    interpreter.invoke()
    t_inf_end = curr_us()
    perf_timer.add('inference', t_inf_end - t_inf_start)
    
    # Postprocessing
    t_post_start = curr_us()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output if needed
    if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale
    
    # Get predictions
    predictions = get_top_predictions(output_data, labels)
    
    t_post_end = curr_us()
    perf_timer.add('postprocessing', t_post_end - t_post_start)
    
    # End-to-end time
    t_e2e_end = curr_us()
    perf_timer.add('end_to_end', t_e2e_end - t_e2e_start)

# Display results
print(f"\n{'='*70}")
print(f"Running on {'NPU' if use_npu else 'CPU'}")
print(f"{'='*70}")
print("\nTop-5 predictions:")
for label, score in predictions:
    print(f"  {label:40s}: {score:.4f}")

# Print performance summary
perf_timer.print_summary()

# Additional analysis
print(f"\n{'='*70}")
print("BOTTLENECK ANALYSIS")
print(f"{'='*70}")

if len(perf_timer.times['end_to_end']) > 0:
    total_time = np.mean(perf_timer.times['end_to_end'])
    inference_time = np.mean(perf_timer.times['inference'])
    overhead = total_time - inference_time
    
    print(f"\nTotal end-to-end time:  {total_time:.2f}ms")
    print(f"Inference time:         {inference_time:.2f}ms  ({(inference_time/total_time)*100:.1f}%)")
    print(f"Overhead (pre+post):    {overhead:.2f}ms  ({(overhead/total_time)*100:.1f}%)")
    
    if overhead > inference_time:
        print(f"\n⚠️  WARNING: Overhead ({overhead:.2f}ms) is greater than inference ({inference_time:.2f}ms)!")
        print(f"   Pipeline is bottlenecked by pre/postprocessing, not inference.")
    else:
        print(f"\n✓ Inference is the main component ({(inference_time/total_time)*100:.1f}% of total time)")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("OPTIMIZATION OPPORTUNITIES")
    print(f"{'='*70}")
    
    preprocessing_time = np.mean(perf_timer.times['preprocessing'])
    postprocessing_time = np.mean(perf_timer.times['postprocessing'])
    
    if preprocessing_time > 5.0:
        print(f"\n• Preprocessing is slow ({preprocessing_time:.2f}ms)")
        print(f"  - Consider pre-converting audio to 16kHz")
        print(f"  - Cache preprocessed audio if processing same file multiple times")
    
    if postprocessing_time > 2.0:
        print(f"\n• Postprocessing is slow ({postprocessing_time:.2f}ms)")
        print(f"  - Consider using fewer classes for sorting")
        print(f"  - Cache labels dictionary")
    
    if use_npu and inference_time > 5.0:
        print(f"\n• NPU inference seems slow ({inference_time:.2f}ms)")
        print(f"  - Verify model is properly quantized (INT8)")
        print(f"  - Check that nodes are being delegated to NPU")
        print(f"  - Run: python3 diagnose_npu.py")
    
    if not use_npu and inference_time > 8.0:
        print(f"\n• CPU inference is slow ({inference_time:.2f}ms)")
        print(f"  - Try enabling NPU: --use-npu")
        print(f"  - Expected speedup: 3-5x")

print(f"\n{'='*70}")
