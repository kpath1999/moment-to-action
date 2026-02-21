#!/usr/bin/env python3
"""
Real-time YAMNet Audio Classification
Continuously captures audio and runs inference
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import sounddevice as sd
import csv
import time
import sys
from collections import deque
import threading
import queue

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 15600  # 0.975 seconds at 16kHz (YAMNet input size)
OVERLAP = 7800  # 50% overlap for smoother detection
CHANNELS = 1

class AudioConfig:
    """Configuration for audio processing"""
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.chunk_size = CHUNK_SIZE
        self.overlap = OVERLAP
        self.channels = CHANNELS
        self.use_npu = '--use-npu' in sys.argv
        self.model_path = "yamnet_quantized.tflite"
        self.labels_path = "yamnet_class_map.csv"
        
        # Performance monitoring
        self.enable_profiling = '--profile' in sys.argv
        
        # Wake word / trigger settings
        self.use_trigger = '--trigger' in sys.argv
        self.trigger_threshold = 0.3  # Confidence threshold to trigger processing
        self.trigger_classes = [
            'Speech', 'Laughter', 'Scream', 'Shout', 
            'Crying, sobbing', 'Baby cry, infant cry'
        ]  # Classes that trigger full processing

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.metrics = {
            'audio_capture': deque(maxlen=100),
            'preprocessing': deque(maxlen=100),
            'inference': deque(maxlen=100),
            'postprocessing': deque(maxlen=100),
            'total': deque(maxlen=100),
        }
    
    def record(self, stage, duration_ms):
        if self.enabled:
            self.metrics[stage].append(duration_ms)
    
    def get_stats(self, stage):
        if not self.metrics[stage]:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        
        data = list(self.metrics[stage])
        return {
            'mean': np.mean(data),
            'min': np.min(data),
            'max': np.max(data),
            'std': np.std(data),
            'count': len(data)
        }
    
    def print_report(self):
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        for stage, _ in self.metrics.items():
            stats = self.get_stats(stage)
            print(f"{stage:20s}: {stats['mean']:6.2f}ms avg | "
                  f"{stats['min']:6.2f}ms min | {stats['max']:6.2f}ms max | "
                  f"±{stats['std']:5.2f}ms std")
        print("="*70)

class RealtimeYAMNet:
    """Real-time audio classification with YAMNet"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Circular buffer for audio
        self.audio_buffer = deque(maxlen=config.chunk_size * 2)
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(config.enable_profiling)
        
        # Load labels
        self.labels = self.load_labels(config.labels_path)
        
        # Load model
        self.load_model()
        
        # Statistics
        self.total_inferences = 0
        self.start_time = None
        
    def load_labels(self, csv_path):
        """Load YAMNet class labels"""
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row['index'])] = row['display_name']
        return labels
    
    def load_model(self):
        """Load TFLite model with optional NPU acceleration"""
        print(f"Loading model (backend: {'NPU' if self.config.use_npu else 'CPU'})...")
        
        experimental_delegates = []
        if self.config.use_npu:
            try:
                experimental_delegates = [
                    load_delegate("libQnnTFLiteDelegate.so", 
                                options={"backend_type": "htp"})
                ]
                print("✓ NPU delegate loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load NPU delegate: {e}")
                print("  Falling back to CPU")
        
        self.interpreter = Interpreter(
            model_path=self.config.model_path,
            experimental_delegates=experimental_delegates
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Warmup
        print("Warming up model...")
        dummy = np.zeros(self.config.chunk_size, dtype=np.float32)
        self.run_inference(dummy)
        print("✓ Model ready\n")
    
    def preprocess_audio(self, audio_chunk):
        """Preprocess audio for YAMNet"""
        t_start = time.perf_counter()
        
        # Ensure correct shape
        if len(audio_chunk) != self.config.chunk_size:
            if len(audio_chunk) < self.config.chunk_size:
                audio_chunk = np.pad(audio_chunk, 
                                   (0, self.config.chunk_size - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:self.config.chunk_size]
        
        # Normalize
        if np.abs(audio_chunk).max() > 0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        # Convert to float32
        audio_chunk = audio_chunk.astype(np.float32)
        
        t_end = time.perf_counter()
        self.perf_monitor.record('preprocessing', (t_end - t_start) * 1000)
        
        return audio_chunk
    
    def run_inference(self, audio_chunk):
        """Run inference on audio chunk"""
        t_start = time.perf_counter()
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], audio_chunk)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        t_end = time.perf_counter()
        self.perf_monitor.record('inference', (t_end - t_start) * 1000)
        
        return output_data
    
    def postprocess_output(self, output_data, top_k=5):
        """Post-process model output"""
        t_start = time.perf_counter()
        
        # Handle different output shapes
        scores = output_data
        if len(scores.shape) > 1:
            if scores.shape[0] == 1:
                scores = scores[0]
            else:
                scores = np.mean(scores, axis=0)
        
        # Get top predictions
        top_indices = np.argsort(scores)[-top_k:][::-1]
        predictions = [(self.labels[idx], scores[idx]) for idx in top_indices]
        
        t_end = time.perf_counter()
        self.perf_monitor.record('postprocessing', (t_end - t_start) * 1000)
        
        return predictions
    
    def should_process(self, predictions):
        """Determine if we should continue processing based on trigger classes"""
        if not self.config.use_trigger:
            return True
        
        for label, score in predictions:
            if label in self.config.trigger_classes and score >= self.config.trigger_threshold:
                return True
        return False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Add to queue (make a copy to avoid data being overwritten)
        self.audio_queue.put(indata.copy())
    
    def processing_thread(self):
        """Thread for processing audio chunks"""
        print("Processing thread started")
        
        while self.running:
            try:
                # Get audio from queue (with timeout to allow checking self.running)
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                t_total_start = time.perf_counter()
                
                # Add to buffer
                self.audio_buffer.extend(audio_chunk.flatten())
                
                # Process when we have enough data
                if len(self.audio_buffer) >= self.config.chunk_size:
                    # Get chunk to process
                    chunk_to_process = np.array(list(self.audio_buffer)[:self.config.chunk_size])
                    
                    # Preprocess
                    processed_chunk = self.preprocess_audio(chunk_to_process)
                    
                    # Run inference
                    output = self.run_inference(processed_chunk)
                    
                    # Postprocess
                    predictions = self.postprocess_output(output)
                    
                    # Check if we should continue processing
                    should_process = self.should_process(predictions)
                    
                    # Record total time
                    t_total_end = time.perf_counter()
                    total_time = (t_total_end - t_total_start) * 1000
                    self.perf_monitor.record('total', total_time)
                    
                    self.total_inferences += 1
                    
                    # Put results in queue
                    self.result_queue.put({
                        'predictions': predictions,
                        'should_process': should_process,
                        'inference_num': self.total_inferences,
                        'total_time': total_time
                    })
                    
                    # Remove processed samples (with overlap)
                    for _ in range(self.config.chunk_size - self.config.overlap):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
    
    def display_thread(self):
        """Thread for displaying results"""
        print("Display thread started")
        last_display_time = time.time()
        display_interval = 0.5  # Update display every 0.5 seconds
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    # Clear screen (optional)
                    # print("\033[2J\033[H", end='')
                    
                    print(f"\n{'='*70}")
                    print(f"Inference #{result['inference_num']} | "
                          f"Total time: {result['total_time']:.2f}ms")
                    
                    if self.config.use_trigger:
                        trigger_status = "🔴 TRIGGERED" if result['should_process'] else "⚪ Idle"
                        print(f"Status: {trigger_status}")
                    
                    print(f"{'='*70}")
                    print("Top 5 Predictions:")
                    
                    for i, (label, score) in enumerate(result['predictions'], 1):
                        confidence = score * 100
                        bar_length = int(confidence / 2)
                        bar = "█" * bar_length
                        
                        # Highlight trigger classes
                        prefix = "🎯 " if label in self.config.trigger_classes else "   "
                        
                        print(f"{prefix}{i}. {label:35s} {confidence:5.1f}% {bar}")
                    
                    # Show performance stats every 20 inferences
                    if result['inference_num'] % 20 == 0:
                        self.perf_monitor.print_report()
                    
                    last_display_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display thread: {e}")
    
    def start(self):
        """Start real-time processing"""
        self.running = True
        self.start_time = time.time()
        
        # Start processing thread
        self.proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        self.proc_thread.start()
        
        # Start display thread
        self.disp_thread = threading.Thread(target=self.display_thread, daemon=True)
        self.disp_thread.start()
        
        print(f"\n{'='*70}")
        print("REAL-TIME AUDIO CLASSIFICATION")
        print(f"{'='*70}")
        print(f"Sample rate: {self.config.sample_rate} Hz")
        print(f"Chunk size: {self.config.chunk_size} samples ({self.config.chunk_size/self.config.sample_rate:.3f}s)")
        print(f"Overlap: {self.config.overlap} samples ({self.config.overlap/self.config.sample_rate:.3f}s)")
        print(f"Backend: {'NPU' if self.config.use_npu else 'CPU'}")
        print(f"Profiling: {'Enabled' if self.config.enable_profiling else 'Disabled'}")
        print(f"Trigger mode: {'Enabled' if self.config.use_trigger else 'Disabled'}")
        if self.config.use_trigger:
            print(f"Trigger classes: {', '.join(self.config.trigger_classes)}")
        print(f"{'='*70}")
        print("\nListening... (Press Ctrl+C to stop)\n")
        
        # Start audio stream
        with sd.InputStream(samplerate=self.config.sample_rate,
                           channels=self.config.channels,
                           callback=self.audio_callback,
                           blocksize=self.config.chunk_size // 4):  # Small blocks for lower latency
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopping...")
                self.stop()
    
    def stop(self):
        """Stop real-time processing"""
        self.running = False
        
        # Wait for threads to finish
        if hasattr(self, 'proc_thread'):
            self.proc_thread.join(timeout=2)
        if hasattr(self, 'disp_thread'):
            self.disp_thread.join(timeout=2)
        
        # Print final statistics
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print("SESSION SUMMARY")
        print(f"{'='*70}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Total inferences: {self.total_inferences}")
        print(f"Average rate: {self.total_inferences/elapsed:.2f} inferences/second")
        
        if self.config.enable_profiling:
            self.perf_monitor.print_report()


def main():
    """Main entry point"""
    # Check for help
    if '--help' in sys.argv:
        print("""
Real-time YAMNet Audio Classification

Usage: python3 realtime_yamnet.py [options]

Options:
  --use-npu      Use NPU acceleration (default: CPU)
  --profile      Enable performance profiling
  --trigger      Enable trigger mode (only process on certain sounds)
  --help         Show this help message

Examples:
  # Run on CPU with profiling
  python3 realtime_yamnet.py --profile
  
  # Run on NPU with trigger mode
  python3 realtime_yamnet.py --use-npu --trigger
  
  # Run on NPU with full profiling
  python3 realtime_yamnet.py --use-npu --profile
        """)
        return
    
    # Create configuration
    config = AudioConfig()
    
    # Create and start classifier
    classifier = RealtimeYAMNet(config)
    classifier.start()


if __name__ == "__main__":
    main()
