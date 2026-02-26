#!/usr/bin/env python3
"""
Backend Comparison Benchmark for YAMNet
Systematically compares CPU, NPU, and DSP performance
"""
import sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import soundfile as sf
import librosa
import csv
import time
import os
from pathlib import Path
import json

class BackendBenchmark:
    """Benchmark different inference backends"""
    
    def __init__(self, model_path, labels_path):
        self.model_path = model_path
        self.labels_path = labels_path
        self.labels = self.load_labels(labels_path)
        
        # Test configurations
        self.backends = ['cpu', 'npu_htp', 'npu_dsp']
        self.results = {}
    
    def load_labels(self, csv_path):
        """Load YAMNet labels"""
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row['index'])] = row['display_name']
        return labels
    
    def load_interpreter(self, backend='cpu'):
        """Load interpreter with specified backend"""
        print(f"\nLoading interpreter with backend: {backend}")
        
        experimental_delegates = []
        
        if backend == 'npu_htp':
            try:
                delegate = load_delegate(
                    "libQnnTFLiteDelegate.so",
                    options={"backend_type": "htp"}
                )
                experimental_delegates = [delegate]
                print("  ✓ NPU HTP delegate loaded")
            except Exception as e:
                print(f"  ✗ Failed to load NPU HTP delegate: {e}")
                return None
                
        elif backend == 'npu_dsp':
            try:
                delegate = load_delegate(
                    "libQnnTFLiteDelegate.so",
                    options={"backend_type": "dsp"}
                )
                experimental_delegates = [delegate]
                print("  ✓ NPU DSP delegate loaded")
            except Exception as e:
                print(f"  ✗ Failed to load NPU DSP delegate: {e}")
                return None
        
        # CPU uses no delegates
        try:
            interpreter = Interpreter(
                model_path=self.model_path,
                experimental_delegates=experimental_delegates
            )
            interpreter.allocate_tensors()
            print("  ✓ Interpreter allocated")
            return interpreter
        except Exception as e:
            print(f"  ✗ Failed to create interpreter: {e}")
            return None
    
    def preprocess_audio(self, audio_path, target_sr=16000, target_length=15600):
        """Preprocess audio for YAMNet"""
        waveform, sr = sf.read(audio_path, dtype='float32')
        
        if sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        
        if np.abs(waveform).max() > 0:
            waveform = waveform / np.abs(waveform).max()
        
        if len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        elif len(waveform) > target_length:
            start_idx = (len(waveform) - target_length) // 2
            waveform = waveform[start_idx:start_idx + target_length]
        
        return waveform.astype(np.float32)
    
    def benchmark_backend(self, backend, audio_files, num_iterations=100):
        """Benchmark a specific backend"""
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {backend.upper()}")
        print(f"{'='*80}")
        
        # Load interpreter
        interpreter = self.load_interpreter(backend)
        if interpreter is None:
            print(f"Skipping {backend} - failed to load")
            return None
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        print(f"\nWarming up ({5} iterations)...")
        dummy = np.zeros(15600, dtype=np.float32)
        for _ in range(5):
            interpreter.set_tensor(input_details[0]['index'], dummy)
            interpreter.invoke()
        
        # Benchmark results storage
        results = {
            'backend': backend,
            'inference_times': [],
            'preprocessing_times': [],
            'total_times': [],
            'audio_results': {}
        }
        
        # Test each audio file
        print(f"\nTesting {len(audio_files)} audio files...")
        for audio_name, audio_path in audio_files.items():
            print(f"\n  Testing: {audio_name}")
            
            # Preprocess once
            t_prep_start = time.perf_counter()
            waveform = self.preprocess_audio(audio_path)
            t_prep_end = time.perf_counter()
            prep_time = (t_prep_end - t_prep_start) * 1000
            
            # Run multiple iterations
            inference_times = []
            for i in range(num_iterations):
                # Inference
                t_inf_start = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], waveform)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                t_inf_end = time.perf_counter()
                
                inf_time = (t_inf_end - t_inf_start) * 1000
                inference_times.append(inf_time)
            
            # Get predictions (from last run)
            scores = output[0] if len(output.shape) > 1 else output
            top_idx = np.argsort(scores)[-5:][::-1]
            predictions = [(self.labels[idx], float(scores[idx])) for idx in top_idx]
            
            # Store results
            results['audio_results'][audio_name] = {
                'preprocessing_time_ms': prep_time,
                'inference_times_ms': inference_times,
                'mean_inference_ms': np.mean(inference_times),
                'std_inference_ms': np.std(inference_times),
                'min_inference_ms': np.min(inference_times),
                'max_inference_ms': np.max(inference_times),
                'median_inference_ms': np.median(inference_times),
                'predictions': predictions
            }
            
            results['preprocessing_times'].append(prep_time)
            results['inference_times'].extend(inference_times)
            results['total_times'].extend([prep_time + t for t in inference_times])
            
            # Print progress
            print(f"    Preprocessing: {prep_time:.2f}ms")
            print(f"    Inference: {np.mean(inference_times):.2f}ms ± {np.std(inference_times):.2f}ms")
            print(f"    Top prediction: {predictions[0][0]} ({predictions[0][1]*100:.1f}%)")
        
        # Calculate overall statistics
        results['overall'] = {
            'mean_preprocessing_ms': np.mean(results['preprocessing_times']),
            'mean_inference_ms': np.mean(results['inference_times']),
            'std_inference_ms': np.std(results['inference_times']),
            'min_inference_ms': np.min(results['inference_times']),
            'max_inference_ms': np.max(results['inference_times']),
            'median_inference_ms': np.median(results['inference_times']),
            'p95_inference_ms': np.percentile(results['inference_times'], 95),
            'p99_inference_ms': np.percentile(results['inference_times'], 99),
            'total_inferences': len(results['inference_times'])
        }
        
        return results
    
    def run_comparison(self, audio_files, num_iterations=100):
        """Run comparison across all backends"""
        print("\n" + "="*80)
        print("BACKEND COMPARISON BENCHMARK")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Audio files: {len(audio_files)}")
        print(f"  Iterations per file: {num_iterations}")
        print(f"  Total inferences per backend: {len(audio_files) * num_iterations}")
        
        # Run benchmarks
        for backend in self.backends:
            result = self.benchmark_backend(backend, audio_files, num_iterations)
            if result:
                self.results[backend] = result
        
        # Generate comparison report
        self.print_comparison_report()
        
        # Save detailed results
        self.save_results()
    
    def print_comparison_report(self):
        """Print comparison report across backends"""
        print("\n" + "="*80)
        print("COMPARISON REPORT")
        print("="*80)
        
        if not self.results:
            print("No results to compare")
            return
        
        # Overall comparison table
        print("\nOVERALL PERFORMANCE:")
        print("-"*80)
        print(f"{'Backend':<15} {'Mean (ms)':<12} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'P95 (ms)':<12}")
        print("-"*80)
        
        for backend, result in self.results.items():
            overall = result['overall']
            print(f"{backend:<15} "
                  f"{overall['mean_inference_ms']:<12.2f} "
                  f"{overall['std_inference_ms']:<12.2f} "
                  f"{overall['min_inference_ms']:<12.2f} "
                  f"{overall['max_inference_ms']:<12.2f} "
                  f"{overall['p95_inference_ms']:<12.2f}")
        
        # Speedup comparison
        print("\n\nSPEEDUP COMPARISON (vs CPU):")
        print("-"*80)
        
        if 'cpu' in self.results:
            cpu_mean = self.results['cpu']['overall']['mean_inference_ms']
            
            for backend, result in self.results.items():
                if backend == 'cpu':
                    continue
                
                backend_mean = result['overall']['mean_inference_ms']
                speedup = cpu_mean / backend_mean
                
                print(f"{backend:<15}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        # Per-audio comparison
        print("\n\nPER-AUDIO FILE COMPARISON:")
        print("-"*80)
        
        # Get all audio files
        audio_files = set()
        for result in self.results.values():
            audio_files.update(result['audio_results'].keys())
        
        for audio_name in sorted(audio_files):
            print(f"\n{audio_name}:")
            print(f"  {'Backend':<15} {'Inference (ms)':<20} {'Top Prediction':<40}")
            print(f"  {'-'*75}")
            
            for backend, result in self.results.items():
                if audio_name in result['audio_results']:
                    audio_result = result['audio_results'][audio_name]
                    mean_time = audio_result['mean_inference_ms']
                    std_time = audio_result['std_inference_ms']
                    top_pred = audio_result['predictions'][0]
                    
                    print(f"  {backend:<15} {mean_time:>6.2f} ± {std_time:>5.2f}      "
                          f"{top_pred[0]:<30} ({top_pred[1]*100:>5.1f}%)")
        
        print("\n" + "="*80)
    
    def save_results(self, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_serializable = convert_numpy(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")


def main():
    """Main entry point"""
    
    # Configuration
    MODEL_PATH = "yamnet_quantized.tflite"
    LABELS_PATH = "yamnet_class_map.csv"
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        return
    
    if not os.path.exists(LABELS_PATH):
        print(f"Error: Labels file not found: {LABELS_PATH}")
        return
    
    # Get test audio files
    audio_dir = Path("test_audio")
    if not audio_dir.exists():
        print(f"Error: Test audio directory not found: {audio_dir}")
        print("Please create test_audio/ directory and add some audio files")
        return
    
    # Find audio files
    audio_files = {}
    for ext in ['*.wav', '*.mp3', '*.flac']:
        for file in audio_dir.glob(ext):
            audio_files[file.stem] = str(file)
    
    if not audio_files:
        print(f"Error: No audio files found in {audio_dir}")
        print("Supported formats: WAV, MP3, FLAC")
        return
    
    print(f"Found {len(audio_files)} audio files:")
    for name in audio_files:
        print(f"  - {name}")
    
    # Get number of iterations from command line or use default
    num_iterations = 100
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of iterations: {sys.argv[1]}")
            print("Using default: 100")
    
    # Run benchmark
    benchmark = BackendBenchmark(MODEL_PATH, LABELS_PATH)
    benchmark.run_comparison(audio_files, num_iterations)


if __name__ == "__main__":
    main()
