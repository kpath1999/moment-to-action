#!/usr/bin/env python3
"""
Real-time YAMNet Audio Classification - OPTIMIZED
- Efficient 480ms hop (YAMNet's native stride)
- Supports live microphone input and file streaming
- MP4 audio extraction built-in
- Clean inference output with neat formatting
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import sounddevice as sd
import soundfile as sf
import csv
import time
import sys
from collections import deque
import threading
import queue
import os
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Security-specific prompts
prompts = [
    "a person pointing a gun at someone",
    "a violent physical attack",
    "people walking peacefully",
    "a normal street scene",
    "an armed robbery at gunpoint",
    "a person who has been shot in the heart",
    "a person who is happy",
    "people appreciating each other"
]

# YAMNet's Native Configuration (OPTIMIZED)
SAMPLE_RATE = 16000
WINDOW_SIZE = 15600  # 0.975 seconds (YAMNet's analysis window)
HOP_SIZE = 7680      # 0.48 seconds (YAMNet's native hop) - EFFICIENT!
CHANNELS = 1

class AudioConfig:
    """Configuration for audio processing"""
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.window_size = WINDOW_SIZE
        self.hop_size = HOP_SIZE  # Using efficient 480ms hop!
        self.channels = CHANNELS
        self.use_npu = '--use-npu' in sys.argv
        self.model_path = "models/yamnet/yamnet_quantized.tflite"
        self.labels_path = "models/yamnet/yamnet_class_map.csv"
        
        # Performance monitoring
        self.enable_profiling = '--profile' in sys.argv
        
        # Security/violence detection settings
        self.detect_screams = '--detect-screams' in sys.argv

        # Per-class thresholds — lower for distress speech since speech prior dominates.
        # Impulsive classes (gunshot, glass) are acoustically distinct so can be higher.
        self.scream_classes = [
            'Scream', 'Screaming', 'Shout', 'Yell',
            'Battle cry', 'Children shouting',
            'Crying, sobbing', 'Wail, moan'
        ]
        self.scream_threshold = 0.10  # Absolute floor — we add energy gate on top

        # Security trigger classes with individual thresholds
        # (class_name_substring, absolute_score_threshold)
        self.security_classes = {
            # Distress speech — low threshold, gated by energy
            'Scream':              0.10,
            'Screaming':           0.10,
            'Shout':               0.10,
            'Yell':                0.10,
            'Battle cry':          0.10,
            'Children shouting':   0.10,
            'Crying, sobbing':     0.12,
            'Wail, moan':          0.12,
            # Impulsive/transient — acoustically distinct, higher threshold ok
            'Gunshot':             0.20,
            'Gunfire':             0.20,
            'Explosion':           0.18,
            'Glass':               0.18,  # matches "Glass" and "Breaking glass"
            # Ambient/contextual
            'Crowd':               0.25,
            'Riot':                0.15,
        }

        # Energy gate — speech is ~0.005-0.03 RMS, shouting is >0.05
        # Tune this by checking rms_energy printed during normal speech vs shouting
        self.energy_gate_rms = 0.05        # minimum RMS to trigger distress-speech classes
        self.energy_gate_classes = {       # which class substrings require energy gate
            'Scream', 'Screaming', 'Shout', 'Yell',
            'Battle cry', 'Children shouting',
            'Crying, sobbing', 'Wail, moan'
        }

        # Temporal smoothing — require N consecutive detections to fire alert
        # Reduces false positives from single-frame spikes
        self.confirmation_frames = 2
        
        # File streaming settings
        self.simulate_stream = '--simulate' in sys.argv
        self.audio_file = None
        self.stream_chunk_duration = 0.1  # Stream in 100ms chunks (like a real mic buffer)
        
        # Parse audio file path if provided
        for arg in sys.argv:
            if arg.endswith('.wav') or arg.endswith('.mp4'):
                self.audio_file = arg
                self.simulate_stream = True
                break

class AudioExtractor:
    """Extract audio from video files"""
    
    @staticmethod
    def extract_audio_from_mp4(mp4_file, output_wav=None):
        """Extract audio from MP4 using ffmpeg"""
        if output_wav is None:
            output_wav = mp4_file.replace('.mp4', '_audio.wav')
        
        cmd = [
            'ffmpeg', '-i', mp4_file,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNELS),
            '-y',  # Overwrite
            output_wav
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Extracted audio: {output_wav}")
            return output_wav
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to extract audio from {mp4_file}")
            return None

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.metrics = {
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

class AudioStreamSimulator:
    """Simulates real-time audio streaming from a file"""
    
    def __init__(self, audio_file, chunk_duration=0.1, sample_rate=16000, loop=True):
        self.audio_file = audio_file
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.loop = loop
        self.running = False
        
        # Load audio file
        print(f"Loading audio file: {audio_file}")
        self.audio_data, self.file_sample_rate = sf.read(audio_file, dtype='float32')
        
        # Resample if needed
        if self.file_sample_rate != sample_rate:
            print(f"Resampling from {self.file_sample_rate}Hz to {sample_rate}Hz")
            try:
                from scipy import signal
                num_samples = int(len(self.audio_data) * sample_rate / self.file_sample_rate)
                self.audio_data = signal.resample(self.audio_data, num_samples)
            except ImportError:
                print("⚠ scipy not available, using librosa for resampling")
                import librosa
                self.audio_data = librosa.resample(self.audio_data, 
                                                   orig_sr=self.file_sample_rate, 
                                                   target_sr=sample_rate)
        
        # Convert to mono if stereo
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
        
        self.chunk_size = int(chunk_duration * sample_rate)
        self.position = 0
        
        duration = len(self.audio_data) / sample_rate
        print(f"✓ Loaded {duration:.2f}s of audio")
        print(f"  Streaming in {chunk_duration*1000:.0f}ms chunks ({self.chunk_size} samples)")
        print(f"  Loop mode: {'Enabled' if loop else 'Disabled'}")
    
    def start(self, callback):
        """Start streaming audio chunks to callback"""
        self.running = True
        self.callback = callback
        self.thread = threading.Thread(target=self._stream_thread, daemon=True)
        self.thread.start()
    
    def _stream_thread(self):
        """Thread that reads and streams audio chunks"""
        while self.running:
            # Get next chunk
            end_pos = self.position + self.chunk_size
            
            if end_pos <= len(self.audio_data):
                chunk = self.audio_data[self.position:end_pos]
            else:
                # End of file
                if self.loop:
                    # Wrap around
                    remaining = end_pos - len(self.audio_data)
                    chunk = np.concatenate([
                        self.audio_data[self.position:],
                        self.audio_data[:remaining]
                    ])
                    self.position = remaining
                else:
                    # Pad and stop
                    chunk = np.pad(self.audio_data[self.position:], 
                                 (0, self.chunk_size - (len(self.audio_data) - self.position)))
                    self.running = False
            
            # Update position
            if self.running and not (end_pos > len(self.audio_data) and self.loop):
                self.position = end_pos
            
            # Send chunk to callback
            chunk_2d = chunk.reshape(-1, 1)
            self.callback(chunk_2d, len(chunk), None, None)
            
            # Sleep to simulate real-time streaming
            time.sleep(self.chunk_duration)
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)

class RealtimeYAMNet:
    """Real-time audio classification with YAMNet - OPTIMIZED"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Circular buffer for audio
        self.audio_buffer = deque(maxlen=config.window_size * 2)
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(config.enable_profiling)
        
        # Load labels
        self.labels = self.load_labels(config.labels_path)
        
        # Find scream/security indices if detecting screams
        if config.detect_screams:
            self.scream_indices = self.find_scream_indices()
            self.security_index_map = self.build_security_index_map()
            print(f"Security detection enabled ({len(self.security_index_map)} classes matched)")
            for label, (idx, thr) in self.security_index_map.items():
                print(f"  [{idx:3d}] {label:35s} threshold={thr:.2f}")

        # Temporal confirmation buffer
        self.detection_streak = 0
        
        # Load model
        self.load_model()
        
        # Statistics
        self.total_inferences = 0
        self.scream_count = 0
        self.start_time = None
        self.stream_simulator = None
        
        # Track time for timestamps
        self.processing_start_time = None
    
    def load_labels(self, csv_path):
        """Load YAMNet class labels"""
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row['index'])] = row['display_name']
        return labels
    
    def find_scream_indices(self):
        """Find indices of scream-related classes"""
        scream_indices = []
        for idx, label in self.labels.items():
            for scream_class in self.config.scream_classes:
                if scream_class.lower() in label.lower():
                    scream_indices.append(idx)
                    break
        return scream_indices
    
    def build_security_index_map(self):
        """
        Build a map of {display_name: (index, threshold)} for all security
        trigger classes, matched by substring against the label CSV.
        """
        index_map = {}
        for idx, label in self.labels.items():
            for class_substr, threshold in self.config.security_classes.items():
                if class_substr.lower() in label.lower():
                    # Keep the entry with the lowest threshold if multiple substrings match
                    if label not in index_map or threshold < index_map[label][1]:
                        index_map[label] = (idx, threshold)
                    break
        return index_map

    def load_model(self):
        """Load TFLite model with optional NPU acceleration"""
        print(f"\nLoading model (backend: {'NPU' if self.config.use_npu else 'CPU'})...")
        
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
        dummy = np.zeros(self.config.window_size, dtype=np.float32)
        self.run_inference(dummy)
        print("✓ Model ready\n")
    
    def preprocess_audio(self, audio_chunk):
        """
        Preprocess audio for YAMNet.

        KEY CHANGE: We no longer normalize by peak amplitude.
        Peak normalization throws away loudness, which is the primary
        acoustic feature separating calm speech from shouting/screaming.
        Instead we apply a soft global gain so very quiet environments
        don't lose dynamic range entirely, while preserving relative
        loudness differences within a session.
        """
        t_start = time.perf_counter()

        # Ensure correct shape
        if len(audio_chunk) != self.config.window_size:
            if len(audio_chunk) < self.config.window_size:
                audio_chunk = np.pad(audio_chunk,
                                   (0, self.config.window_size - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:self.config.window_size]

        audio_chunk = audio_chunk.astype(np.float32)

        # Compute RMS *before* any gain — this is what the energy gate uses
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

        # Soft gain: scale so that a "normal speech" level (~0.02 RMS) sits
        # around 0.3 in the normalized range.  Loud events will then exceed
        # 0.3 naturally.  Hard-clip to [-1, 1] to keep the model happy.
        TARGET_RMS = 0.02
        SOFT_GAIN  = 15.0   # 0.02 * 15 = 0.30 for average speech
        if rms > 1e-6:      # avoid gain on pure silence
            gain = min(SOFT_GAIN, TARGET_RMS / rms * SOFT_GAIN)
            audio_chunk = np.clip(audio_chunk * gain, -1.0, 1.0)

        t_end = time.perf_counter()
        self.perf_monitor.record('preprocessing', (t_end - t_start) * 1000)

        return audio_chunk, rms   # return rms so postprocess can use it
    
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
    
    def postprocess_output(self, output_data, rms, top_k=5):
        """
        Post-process model output.

        Detection logic:
        1. For each security class, check if its absolute score exceeds
           its individual threshold (lower than before for distress speech).
        2. For distress-speech classes (shouting, screaming etc.), additionally
           require RMS energy > energy_gate_rms.  This disambiguates calm
           speech (low energy, high YAMNet speech score) from shouting
           (high energy, elevated YAMNet speech + shout scores).
        3. Require confirmation_frames consecutive detections before firing.
        """
        t_start = time.perf_counter()

        scores = output_data
        if len(scores.shape) > 1:
            scores = scores[0] if scores.shape[0] == 1 else np.mean(scores, axis=0)

        print(f"Here")
        speech_score = float(scores[0])
        print(f"Speech score: {speech_score}")
        is_human_vocal = speech_score >= 0.3
        is_scream = False

        if is_human_vocal:
            if rms >= 0.05:
                is_scream = True
                print(f"\n Override: Loud shout detected\n")

                # Load standard OpenAI CLIP - larger and stronger than MobileCLIP
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model.eval()

                # Load your test image
                #image = Image.open("pedestrian.jpg").convert("RGB")
                image = Image.open("images_for_clip_test/weapon.jpg").convert("RGB")

                inputs = processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits_per_image

                    # Use model's own learned temperature
                    probs = logits.softmax(dim=1)

                print("OpenAI CLIP scores:")
                for prompt, score in zip(prompts, probs[0]):
                    print(f"  {score:.4f} - {prompt}")
                else:
                    print(f"No override")        

        # Top-k for display
        top_indices = np.argsort(scores)[-top_k:][::-1]
        predictions = [(self.labels[idx], scores[idx]) for idx in top_indices]

        # Security detection
        triggered_classes = []
        max_scream_score = 0.0
        is_scream = False

        if self.config.detect_screams:
            for label, (idx, threshold) in self.security_index_map.items():
                score = float(scores[idx])
                if score < threshold:
                    continue

                # Check if this class requires the energy gate
                needs_energy_gate = any(
                    ec.lower() in label.lower()
                    for ec in self.config.energy_gate_classes
                )
                if needs_energy_gate and rms < self.config.energy_gate_rms:
                    continue  # YAMNet says speech/shout but energy is too low → calm speech

                triggered_classes.append((label, score))
                if score > max_scream_score:
                    max_scream_score = score

            # Temporal confirmation
            if triggered_classes:
                self.detection_streak += 1
            else:
                self.detection_streak = 0

            is_scream = self.detection_streak >= self.config.confirmation_frames

        t_end = time.perf_counter()
        self.perf_monitor.record('postprocessing', (t_end - t_start) * 1000)

        return predictions, is_scream, max_scream_score, triggered_classes, rms
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Add to queue
        self.audio_queue.put(indata.copy())
    
    def processing_thread(self):
        """Thread for processing audio chunks"""
        print("Processing thread started")
        
        while self.running:
            try:
                # Get audio from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                t_total_start = time.perf_counter()
                
                # Add to buffer
                self.audio_buffer.extend(audio_chunk.flatten())
                
                # Process when we have enough data (using efficient 480ms hop!)
                if len(self.audio_buffer) >= self.config.window_size:
                    # Get window to process
                    window_to_process = np.array(list(self.audio_buffer)[:self.config.window_size])
                    
                    # Calculate timestamp
                    elapsed = time.time() - self.processing_start_time if self.processing_start_time else 0
                    timestamp = elapsed
                    
                    # Preprocess — returns (chunk, rms) now
                    processed_window, rms = self.preprocess_audio(window_to_process)
                    print(f"\n rms for this chunk = {20*np.log10(rms)}\n")
                    
                    # Run inference
                    output = self.run_inference(processed_window)
                    
                    """
                    speech_score = float(output[0])
                    is_human_vocal = speech_score >= 0.3
                    is_scream = False

                    if is_human_vocal:
                        if rms >= 0.05:
                            is_scream = True
                            print(f"\n Override: Loud shout detected\n")
                        else:
                            print(f"No override")

                    """

                    # Postprocess — returns triggered_classes and rms too
                    predictions, is_scream, scream_score, triggered_classes, rms = \
                        self.postprocess_output(output, rms)
                    
                    # Record total time
                    t_total_end = time.perf_counter()
                    total_time = (t_total_end - t_total_start) * 1000
                    self.perf_monitor.record('total', total_time)
                    
                    self.total_inferences += 1
                    if is_scream:
                        self.scream_count += 1
                    
                    # Put results in queue
                    self.result_queue.put({
                        'predictions': predictions,
                        'is_scream': is_scream,
                        'scream_score': scream_score,
                        'triggered_classes': triggered_classes,
                        'rms': rms,
                        'inference_num': self.total_inferences,
                        'total_time': total_time,
                        'timestamp': timestamp
                    })
                    
                    # Remove processed samples (advance by HOP_SIZE for efficiency!)
                    for _ in range(self.config.hop_size):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                import traceback
                traceback.print_exc()
    
    def display_thread(self):
        """Thread for displaying results - NEAT FORMATTING"""
        print("Display thread started")
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=0.1)
                
                # Clean, neat output for each inference
                print(f"\n{'='*70}")
                print(f"Inference #{result['inference_num']:4d} | "
                      f"Time: {result['timestamp']:7.2f}s | "
                      f"Proc: {result['total_time']:5.1f}ms")

                # Scream detection indicator
                if self.config.detect_screams:
                    rms_bar = "▓" * min(20, int(result['rms'] * 400))
                    print(f"RMS Energy : {result['rms']:.4f}  [{rms_bar:<20s}]  "
                          f"Gate: {'PASS' if result['rms'] >= self.config.energy_gate_rms else 'BLOCK'} "
                          f"(≥{self.config.energy_gate_rms:.3f})")
                    print(f"Streak     : {self.detection_streak}/{self.config.confirmation_frames} frames needed")
                    if result['is_scream']:
                        print(f"🔴 SECURITY ALERT | Score: {result['scream_score']:.3f}")
                        for label, score in result['triggered_classes']:
                            print(f"   ↳ {label}: {score:.3f}")
                    elif result['triggered_classes']:
                        print(f"🟡 BUILDING ({len(result['triggered_classes'])} class(es) triggered, "
                              f"waiting for confirmation)")
                        for label, score in result['triggered_classes']:
                            print(f"   ↳ {label}: {score:.3f}")
                    else:
                        print(f"⚪ No alert  | Max security score: {result['scream_score']:.3f}")
                
                print(f"{'='*70}")
                print("Top 5 Predictions:")
                
                for i, (label, score) in enumerate(result['predictions'], 1):
                    confidence = score * 100
                    bar_length = int(confidence / 2)
                    bar = "█" * bar_length
                    
                    # Highlight scream classes
                    if self.config.detect_screams and any(sc.lower() in label.lower() 
                                                          for sc in self.config.scream_classes):
                        prefix = "🎯 "
                    else:
                        prefix = "   "
                    
                    print(f"{prefix}{i}. {label:35s} {confidence:5.1f}% {bar}")
                
                # Performance stats every 20 inferences
                if result['inference_num'] % 20 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.total_inferences / elapsed
                    print(f"\nRate: {rate:.2f} inferences/sec | "
                          f"Expected: ~2.08/sec (480ms hop)")
                    
                    if self.config.detect_screams:
                        print(f"Screams detected: {self.scream_count}")
                    
                    if self.config.enable_profiling:
                        self.perf_monitor.print_report()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display thread: {e}")
    
    def start(self):
        """Start real-time processing"""
        self.running = True
        self.start_time = time.time()
        self.processing_start_time = time.time()
        
        # Start processing thread
        self.proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        self.proc_thread.start()
        
        # Start display thread
        self.disp_thread = threading.Thread(target=self.display_thread, daemon=True)
        self.disp_thread.start()
        
        print(f"\n{'='*70}")
        print("REAL-TIME AUDIO CLASSIFICATION - OPTIMIZED")
        print(f"{'='*70}")
        print(f"Sample rate: {self.config.sample_rate} Hz")
        print(f"Window size: {self.config.window_size} samples ({self.config.window_size/self.config.sample_rate:.3f}s)")
        print(f"Hop size: {self.config.hop_size} samples ({self.config.hop_size/self.config.sample_rate:.3f}s)")
        print(f"Expected rate: ~{1/(self.config.hop_size/self.config.sample_rate):.2f} inferences/second")
        print(f"Backend: {'NPU' if self.config.use_npu else 'CPU'}")
        print(f"Profiling: {'Enabled' if self.config.enable_profiling else 'Disabled'}")
        print(f"Scream detection: {'Enabled' if self.config.detect_screams else 'Disabled'}")
        
        # Choose audio source
        if self.config.simulate_stream and self.config.audio_file:
            print(f"Mode: SIMULATED STREAMING")
            print(f"Audio file: {self.config.audio_file}")
            print(f"{'='*70}")
            print("\nStreaming from file... (Press Ctrl+C to stop)\n")
            
            # Use simulated stream
            self.stream_simulator = AudioStreamSimulator(
                self.config.audio_file,
                chunk_duration=self.config.stream_chunk_duration,
                sample_rate=self.config.sample_rate,
                loop=False  # Don't loop by default for files
            )
            
            try:
                self.stream_simulator.start(self.audio_callback)
                while self.running and self.stream_simulator.running:
                    time.sleep(0.1)
                if self.stream_simulator.running:
                    self.stop()
                else:
                    # File finished
                    time.sleep(2)  # Let processing finish
                    self.stop()
            except KeyboardInterrupt:
                print("\n\nStopping...")
                self.stop()
        else:
            print(f"Mode: LIVE MICROPHONE")
            print(f"{'='*70}")
            print("\nListening... (Press Ctrl+C to stop)\n")
            
            # Use real microphone
            with sd.InputStream(samplerate=self.config.sample_rate,
                               channels=self.config.channels,
                               callback=self.audio_callback,
                               blocksize=self.config.window_size // 4):
                try:
                    while self.running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    print("\n\nStopping...")
                    self.stop()
    
    def stop(self):
        """Stop real-time processing"""
        self.running = False
        
        # Stop stream simulator if active
        if self.stream_simulator:
            self.stream_simulator.stop()
        
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
        print(f"Expected rate: ~2.08 inferences/second (480ms hop)")
        
        if self.config.detect_screams:
            print(f"Total screams detected: {self.scream_count}")
            print(f"Scream rate: {100*self.scream_count/max(1,self.total_inferences):.1f}%")
        
        if self.config.enable_profiling:
            self.perf_monitor.print_report()


def main():
    """Main entry point"""
    if '--help' in sys.argv:
        print("""
Real-time YAMNet Audio Classification - OPTIMIZED

Usage: python3 realtime_yamnet_stream.py [options] [audio_file]

Options:
  --use-npu          Use NPU acceleration (default: CPU)
  --profile          Enable performance profiling
  --detect-screams   Enable scream detection mode
  --simulate         Enable simulated streaming (auto if file provided)
  --help             Show this help message

Audio File:
  Provide a .wav or .mp4 file path to process
  (MP4 files will have audio extracted automatically)

Examples:
  # Live microphone with scream detection
  python3 realtime_yamnet_stream.py --detect-screams --use-npu
  
  # Process WAV file
  python3 realtime_yamnet_stream.py fighting_scene.wav --detect-screams
  
  # Process MP4 file (auto-extracts audio)
  python3 realtime_yamnet_stream.py fight.mp4 --use-npu --detect-screams
  
  # With profiling
  python3 realtime_yamnet_stream.py audio.wav --profile --detect-screams

Optimization:
  - Uses 480ms hop (YAMNet's native stride)
  - ~2.08 inferences per second
  - 5x more efficient than 100ms chunks!
  - Still catches all scream events with 50% overlap
        """)
        return
    
    # Create configuration
    config = AudioConfig()
    
    # Handle MP4 files - extract audio first
    if config.audio_file and config.audio_file.endswith('.mp4'):
        print("\nMP4 file detected - extracting audio...")
        wav_file = AudioExtractor.extract_audio_from_mp4(config.audio_file)
        if not wav_file:
            print("Failed to extract audio. Exiting.")
            return
        config.audio_file = wav_file
    
    # Validate audio file if provided
    if config.simulate_stream:
        if not config.audio_file:
            print("Error: --simulate flag used but no audio file provided")
            return
        if not os.path.exists(config.audio_file):
            print(f"Error: Audio file not found: {config.audio_file}")
            return
    
    # Create and start classifier
    classifier = RealtimeYAMNet(config)
    classifier.start()


if __name__ == "__main__":
    main()
