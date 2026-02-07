import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import soundfile as sf
import librosa
import csv
import os
import time
import sys
import urllib.request
from pathlib import Path

def curr_ms():
    return round(time.time() * 1000)

# Configuration
use_npu = '--use-npu' in sys.argv
MODEL_PATH = "yamnet_quantized.tflite"
LABELS_PATH = "yamnet_class_map.csv"

# Test audio sources (download these or use your own)
TEST_AUDIO_URLS = {
    "dog_bark": "https://freesound.org/data/previews/231/231762_231762-lq.mp3",
    "cat_meow": "https://freesound.org/data/previews/110/110011_110011-lq.mp3",
    "bird_chirping": "https://freesound.org/data/previews/416/416179_416179-lq.mp3",
    "horse_neigh": "https://freesound.org/data/previews/387/387240_387240-lq.mp3",
    "cow_moo": "https://freesound.org/data/previews/442/442903_442903-lq.mp3",
}

# Create audio directory
AUDIO_DIR = Path("test_audio")
AUDIO_DIR.mkdir(exist_ok=True)

def download_test_audio(urls_dict):
    """Download test audio files if they don't exist"""
    downloaded_files = {}
    for name, url in urls_dict.items():
        filename = AUDIO_DIR / f"{name}.mp3"
        if not filename.exists():
            try:
                print(f"Downloading {name}...")
                urllib.request.urlretrieve(url, filename)
                downloaded_files[name] = str(filename)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        else:
            downloaded_files[name] = str(filename)
    return downloaded_files

# Load class labels
def load_yamnet_labels(csv_path):
    labels = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row['index'])] = row['display_name']
    return labels

# Audio preprocessing for YAMNet
def preprocess_audio(audio_path, target_sr=16000, target_length=15600):
    """
    YAMNet expects:
    - 16 kHz sample rate
    - Mono channel
    - Waveform in range [-1.0, 1.0]
    - Exactly 15600 samples (0.975 seconds)
    """
    try:
        # Load audio file
        waveform, sr = sf.read(audio_path, dtype='float32')
        
        # Resample to 16kHz if needed
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        
        # Convert to mono if stereo
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        # Normalize audio to [-1, 1] range
        if np.abs(waveform).max() > 0:
            waveform = waveform / np.abs(waveform).max()
        
        # Pad or trim to exact length
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)), mode='constant')
        elif len(waveform) > target_length:
            # Take middle portion for better results
            start_idx = (len(waveform) - target_length) // 2
            waveform = waveform[start_idx:start_idx + target_length]
        
        return waveform
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def run_inference(interpreter, input_details, output_details, waveform):
    """Run inference on preprocessed audio"""
    input_data = waveform.astype(np.float32)
    
    # Quantize if needed
    if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
        scale, zero_point = input_details[0]['quantization']
        input_data = (input_data / scale + zero_point).astype(input_details[0]['dtype'])
    
    # Set tensor and invoke
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    start = curr_ms()
    interpreter.invoke()
    inference_time = curr_ms() - start
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if needed
    if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32) - zero_point) * scale
    
    return output_data, inference_time

def get_top_predictions(scores, labels, top_k=5):
    """Get top-k predictions with their scores"""
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

# Main execution
if __name__ == "__main__":
    print("YAMNet Audio Classifier Test Suite")
    print("=" * 60)
    
    # Download test audio
    print("\n1. Downloading test audio files...")
    audio_files = download_test_audio(TEST_AUDIO_URLS)
    
    # Load labels
    print("\n2. Loading YAMNet labels...")
    labels = load_yamnet_labels(LABELS_PATH)
    print(f"   Loaded {len(labels)} class labels")
    
    # Load model
    print(f"\n3. Loading model (using {'NPU' if use_npu else 'CPU'})...")
    experimental_delegates = []
    if use_npu:
        experimental_delegates = [
            load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
        ]
    
    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=experimental_delegates
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Warmup
    print("\n4. Warming up model...")
    dummy_audio = np.zeros(15600, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy_audio)
    interpreter.invoke()
    
    # Test each audio file
    print("\n5. Running inference on test files...")
    print("=" * 60)
    
    results = []
    for name, filepath in audio_files.items():
        print(f"\n📁 Testing: {name}")
        print("-" * 60)
        
        # Preprocess
        waveform = preprocess_audio(filepath)
        if waveform is None:
            print(f"   ❌ Failed to process audio")
            continue
        
        # Run inference
        output_data, inference_time = run_inference(
            interpreter, input_details, output_details, waveform
        )
        
        # Get predictions
        predictions = get_top_predictions(output_data, labels, top_k=5)
        
        # Display results
        print(f"   ⏱️  Inference time: {inference_time}ms")
        print(f"   🎯 Top 5 predictions:")
        for i, (label, score) in enumerate(predictions, 1):
            confidence = score * 100
            bar = "█" * int(confidence / 5)
            print(f"      {i}. {label:35s} {confidence:5.1f}% {bar}")
        
        results.append({
            'name': name,
            'predictions': predictions,
            'time': inference_time
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_time = np.mean([r['time'] for r in results])
    print(f"Average inference time: {avg_time:.1f}ms")
    print(f"Files processed: {len(results)}")
    print(f"Backend: {'NPU' if use_npu else 'CPU'}")
