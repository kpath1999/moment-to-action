#!/usr/bin/env python3
"""
Security Monitoring System - YAMNet + Whisper Integration
- Rolling audio buffer with lookback
- YAMNet detects trigger sounds (shouts, screams, distress)
- Whisper transcribes speech around trigger events
- Logs events with timestamps and transcriptions
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
from datetime import datetime
import json
import whisper

# Audio Configuration
SAMPLE_RATE = 16000
YAMNET_WINDOW_SIZE = 15600  # 0.975 seconds
YAMNET_HOP_SIZE = 7680      # 0.48 seconds
CHANNELS = 1

# Security Monitoring Configuration
class SecurityConfig:
    """Configuration for security monitoring"""
    def __init__(self):
        # Audio settings
        self.sample_rate = SAMPLE_RATE
        self.yamnet_window_size = YAMNET_WINDOW_SIZE
        self.yamnet_hop_size = YAMNET_HOP_SIZE
        self.channels = CHANNELS
        
        # YAMNet model
        self.use_npu = '--use-npu' in sys.argv
        self.yamnet_model_path = "yamnet_quantized.tflite"
        self.labels_path = "yamnet_class_map.csv"
        
        # Whisper model
        self.whisper_model_name = "base"  # tiny, base, small, medium, large
        self.whisper_device = "cpu"  # Will use NPU via TFLite when you switch
        
        # Trigger detection settings
        self.trigger_labels = [
            'Shout', 'Yell', 'Screaming', 'Scream',
            'Battle cry', 'Children shouting',
            'Crying, sobbing', 'Wail, moan',
            'Male speech, man speaking', 'Female speech, woman speaking',
            'Speech', 'Conversation',
            'Gunshot, gunfire', 'Glass breaking', 'Door slam'
        ]
        self.trigger_threshold = 0.30  # Confidence threshold
        self.trigger_cooldown = 5.0    # Seconds before next trigger allowed
        
        # Buffer settings (Option 2: Rolling Buffer with Lookback)
        self.rolling_buffer_duration = 10.0   # Keep last 10 seconds in memory
        self.pre_trigger_duration = 2.0       # Capture 2s before trigger
        self.post_trigger_duration = 8.0      # Capture 8s after trigger
        
        # Logging
        self.log_dir = "security_logs"
        self.save_audio_clips = '--save-audio' in sys.argv
        self.verbose = '--verbose' in sys.argv
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)


class EventLogger:
    """Log security events with timestamps and transcriptions"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        
    def log_event(self, event_data):
        """Log an event to JSON Lines file"""
        with open(self.log_file, 'a') as f:
            json.dump(event_data, f)
            f.write('\n')
        
        # Also print to console
        print(f"\n{'='*80}")
        print(f"🚨 SECURITY EVENT DETECTED")
        print(f"{'='*80}")
        print(f"Time: {event_data['timestamp']}")
        print(f"Trigger: {event_data['trigger_label']} ({event_data['trigger_confidence']:.2%})")
        print(f"Transcription: {event_data['transcription']}")
        if event_data.get('audio_file'):
            print(f"Audio saved: {event_data['audio_file']}")
        print(f"{'='*80}\n")


class SecurityMonitor:
    """Main security monitoring system integrating YAMNet + Whisper"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # Audio queues and buffers
        self.audio_queue = queue.Queue()
        self.rolling_buffer = deque(maxlen=int(config.sample_rate * config.rolling_buffer_duration))
        
        # YAMNet processing
        self.yamnet_buffer = deque(maxlen=config.yamnet_window_size * 2)
        
        # Trigger state
        self.last_trigger_time = 0
        self.is_capturing_post_trigger = False
        self.post_trigger_buffer = []
        self.trigger_event_data = {}
        
        # Event logging
        self.logger = EventLogger(config.log_dir)
        
        # Load YAMNet
        self.labels = self.load_labels(config.labels_path)
        self.trigger_indices = self.find_trigger_indices()
        print(f"Monitoring {len(self.trigger_indices)} trigger classes")
        self.load_yamnet()
        
        # Load Whisper
        self.load_whisper()
        
        # Statistics
        self.total_yamnet_inferences = 0
        self.total_triggers = 0
        self.total_transcriptions = 0
        
    def load_labels(self, csv_path):
        """Load YAMNet class labels"""
        labels = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row['index'])] = row['display_name']
        return labels
    
    def find_trigger_indices(self):
        """Find indices of trigger classes"""
        trigger_indices = []
        print("\nTrigger classes:")
        for idx, label in self.labels.items():
            for trigger_class in self.config.trigger_labels:
                if trigger_class.lower() in label.lower():
                    trigger_indices.append(idx)
                    print(f"  - {label}")
                    break
        return trigger_indices
    
    def load_yamnet(self):
        """Load YAMNet TFLite model"""
        print(f"\nLoading YAMNet (backend: {'NPU' if self.config.use_npu else 'CPU'})...")
        
        experimental_delegates = []
        if self.config.use_npu:
            try:
                experimental_delegates = [
                    load_delegate("libQnnTFLiteDelegate.so", 
                                options={"backend_type": "htp"})
                ]
                print("✓ NPU delegate loaded")
            except Exception as e:
                print(f"✗ NPU delegate failed: {e}")
                print("  Falling back to CPU")
        
        self.yamnet_interpreter = Interpreter(
            model_path=self.config.yamnet_model_path,
            experimental_delegates=experimental_delegates
        )
        self.yamnet_interpreter.allocate_tensors()
        
        self.yamnet_input_details = self.yamnet_interpreter.get_input_details()
        self.yamnet_output_details = self.yamnet_interpreter.get_output_details()
        
        # Warmup
        dummy = np.zeros(self.config.yamnet_window_size, dtype=np.float32)
        self.run_yamnet_inference(dummy)
        print("✓ YAMNet ready")
    
    def load_whisper(self):
        """Load Whisper model"""
        print(f"\nLoading Whisper model: {self.config.whisper_model_name}...")
        self.whisper_model = whisper.load_model(
            self.config.whisper_model_name,
            device=self.config.whisper_device
        )
        print("✓ Whisper ready")
    
    def run_yamnet_inference(self, audio_chunk):
        """Run YAMNet inference"""
        # Ensure correct length
        if len(audio_chunk) != self.config.yamnet_window_size:
            if len(audio_chunk) < self.config.yamnet_window_size:
                audio_chunk = np.pad(audio_chunk, 
                                   (0, self.config.yamnet_window_size - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:self.config.yamnet_window_size]
        
        # Normalize
        if np.abs(audio_chunk).max() > 0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        audio_chunk = audio_chunk.astype(np.float32)
        
        # Run inference
        self.yamnet_interpreter.set_tensor(self.yamnet_input_details[0]['index'], audio_chunk)
        self.yamnet_interpreter.invoke()
        output_data = self.yamnet_interpreter.get_tensor(self.yamnet_output_details[0]['index'])
        
        return output_data
    
    def check_for_trigger(self, scores):
        """Check if any trigger class is detected"""
        # Get scores for trigger classes
        trigger_scores = scores[self.trigger_indices]
        max_trigger_idx = np.argmax(trigger_scores)
        max_trigger_score = trigger_scores[max_trigger_idx]
        
        if max_trigger_score >= self.config.trigger_threshold:
            # Map back to label
            trigger_class_idx = self.trigger_indices[max_trigger_idx]
            trigger_label = self.labels[trigger_class_idx]
            return True, trigger_label, max_trigger_score
        
        return False, None, 0.0
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        print(f"\n🎤 Transcribing {len(audio_data)/self.config.sample_rate:.1f}s of audio...")
        
        # Ensure audio is float32 and normalized
        audio_data = np.array(audio_data, dtype=np.float32)
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Transcribe
        result = self.whisper_model.transcribe(
            audio_data,
            language='en',  # Set to None for auto-detection
            fp16=False,     # Rubik Pi 3 doesn't have FP16 GPU
        )
        
        return result['text'].strip()
    
    def handle_trigger_event(self, trigger_label, trigger_confidence):
        """Handle a trigger event - start capturing post-trigger audio"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_trigger_time < self.config.trigger_cooldown:
            if self.config.verbose:
                print(f"⏳ Trigger in cooldown (last trigger {current_time - self.last_trigger_time:.1f}s ago)")
            return
        
        self.last_trigger_time = current_time
        self.total_triggers += 1
        
        # Get pre-trigger audio from rolling buffer
        pre_trigger_samples = int(self.config.sample_rate * self.config.pre_trigger_duration)
        pre_trigger_audio = list(self.rolling_buffer)[-pre_trigger_samples:]
        
        print(f"\n⚡ TRIGGER: {trigger_label} ({trigger_confidence:.2%})")
        print(f"📼 Capturing {self.config.pre_trigger_duration}s pre + {self.config.post_trigger_duration}s post trigger audio...")
        
        # Store trigger event data
        self.trigger_event_data = {
            'timestamp': datetime.now().isoformat(),
            'trigger_label': trigger_label,
            'trigger_confidence': float(trigger_confidence),
            'pre_trigger_audio': pre_trigger_audio,
        }
        
        # Start capturing post-trigger audio
        self.is_capturing_post_trigger = True
        self.post_trigger_buffer = []
    
    def finalize_trigger_event(self):
        """Finalize trigger event - transcribe and log"""
        if not self.trigger_event_data:
            return
        
        # Combine pre and post trigger audio
        full_audio = np.concatenate([
            self.trigger_event_data['pre_trigger_audio'],
            self.post_trigger_buffer
        ])
        
        # Transcribe
        transcription = self.transcribe_audio(full_audio)
        self.total_transcriptions += 1
        
        # Save audio if requested
        audio_file = None
        if self.config.save_audio_clips:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_file = os.path.join(
                self.config.log_dir,
                f"event_{timestamp_str}.wav"
            )
            sf.write(audio_file, full_audio, self.config.sample_rate)
        
        # Log event
        event_data = {
            'timestamp': self.trigger_event_data['timestamp'],
            'trigger_label': self.trigger_event_data['trigger_label'],
            'trigger_confidence': self.trigger_event_data['trigger_confidence'],
            'transcription': transcription,
            'audio_duration': len(full_audio) / self.config.sample_rate,
            'audio_file': audio_file,
        }
        
        self.logger.log_event(event_data)
        
        # Reset
        self.trigger_event_data = {}
        self.post_trigger_buffer = []
        self.is_capturing_post_trigger = False
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Add to queue
        self.audio_queue.put(indata.copy())
    
    def processing_thread(self):
        """Main processing thread"""
        print("\n🔍 Security monitoring active...")
        
        while self.running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_queue.get(timeout=0.1)
                audio_flat = audio_chunk.flatten()
                
                # Add to rolling buffer (always)
                self.rolling_buffer.extend(audio_flat)
                
                # If capturing post-trigger, add to post-trigger buffer
                if self.is_capturing_post_trigger:
                    self.post_trigger_buffer.extend(audio_flat)
                    
                    # Check if we have enough post-trigger audio
                    post_duration = len(self.post_trigger_buffer) / self.config.sample_rate
                    if post_duration >= self.config.post_trigger_duration:
                        self.finalize_trigger_event()
                
                # YAMNet processing (check for triggers)
                self.yamnet_buffer.extend(audio_flat)
                
                if len(self.yamnet_buffer) >= self.config.yamnet_window_size:
                    # Get window to process
                    window = np.array(list(self.yamnet_buffer)[:self.config.yamnet_window_size])
                    
                    # Run YAMNet inference
                    scores = self.run_yamnet_inference(window)
                    self.total_yamnet_inferences += 1
                    
                    # Handle output shape
                    if len(scores.shape) > 1:
                        if scores.shape[0] == 1:
                            scores = scores[0]
                        else:
                            scores = np.mean(scores, axis=0)
                    
                    # Check for trigger
                    is_trigger, trigger_label, trigger_confidence = self.check_for_trigger(scores)
                    
                    if is_trigger and not self.is_capturing_post_trigger:
                        self.handle_trigger_event(trigger_label, trigger_confidence)
                    
                    # Display top prediction if verbose
                    if self.config.verbose and self.total_yamnet_inferences % 10 == 0:
                        top_idx = np.argmax(scores)
                        top_label = self.labels[top_idx]
                        top_score = scores[top_idx]
                        print(f"[{self.total_yamnet_inferences}] Top: {top_label} ({top_score:.2%})")
                    
                    # Remove processed samples (hop forward)
                    for _ in range(min(self.config.yamnet_hop_size, len(self.yamnet_buffer))):
                        self.yamnet_buffer.popleft()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing: {e}")
                import traceback
                traceback.print_exc()
    
    def start(self):
        """Start security monitoring"""
        self.running = True
        
        # Start processing thread
        self.proc_thread = threading.Thread(target=self.processing_thread, daemon=True)
        self.proc_thread.start()
        
        print(f"\n{'='*80}")
        print("SECURITY MONITORING SYSTEM")
        print(f"{'='*80}")
        print(f"YAMNet model: {self.config.yamnet_model_path}")
        print(f"Whisper model: {self.config.whisper_model_name}")
        print(f"Backend: {'NPU' if self.config.use_npu else 'CPU'}")
        print(f"Trigger threshold: {self.config.trigger_threshold:.2%}")
        print(f"Rolling buffer: {self.config.rolling_buffer_duration}s")
        print(f"Capture window: {self.config.pre_trigger_duration}s pre + {self.config.post_trigger_duration}s post")
        print(f"Log directory: {self.config.log_dir}")
        print(f"Save audio clips: {self.config.save_audio_clips}")
        print(f"{'='*80}\n")
        
        # Start microphone
        with sd.InputStream(samplerate=self.config.sample_rate,
                           channels=self.config.channels,
                           callback=self.audio_callback,
                           blocksize=self.config.yamnet_window_size // 4):
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nStopping...")
                self.stop()
    
    def stop(self):
        """Stop security monitoring"""
        self.running = False
        
        # Wait for processing thread
        if hasattr(self, 'proc_thread'):
            self.proc_thread.join(timeout=2)
        
        # Print statistics
        print(f"\n{'='*80}")
        print("SESSION SUMMARY")
        print(f"{'='*80}")
        print(f"YAMNet inferences: {self.total_yamnet_inferences}")
        print(f"Triggers detected: {self.total_triggers}")
        print(f"Transcriptions completed: {self.total_transcriptions}")
        print(f"Log file: {self.logger.log_file}")
        print(f"{'='*80}\n")


def main():
    """Main entry point"""
    if '--help' in sys.argv:
        print("""
Security Monitoring System - YAMNet + Whisper Integration

Usage: python3 security_monitor.py [options]

Options:
  --use-npu       Use NPU acceleration for YAMNet (default: CPU)
  --save-audio    Save audio clips of triggered events
  --verbose       Show detailed YAMNet predictions
  --help          Show this help message

Features:
  • Rolling audio buffer with 10s lookback
  • YAMNet detects trigger sounds (shouts, screams, distress)
  • Captures 2s before + 8s after trigger
  • Whisper transcribes speech content
  • Logs events with timestamps and transcriptions
  • 5s cooldown between triggers to avoid spam

Examples:
  # Basic monitoring
  python3 security_monitor.py
  
  # With NPU acceleration and audio saving
  python3 security_monitor.py --use-npu --save-audio
  
  # Verbose mode to see all detections
  python3 security_monitor.py --verbose

Security Logs:
  Events are logged to security_logs/events_TIMESTAMP.jsonl
  Each event includes:
    - Timestamp
    - Trigger sound detected
    - Confidence score
    - Whisper transcription
    - Audio file path (if --save-audio used)
        """)
        return
    
    # Create configuration
    config = SecurityConfig()
    
    # Create and start monitor
    monitor = SecurityMonitor(config)
    monitor.start()


if __name__ == "__main__":
    main()
