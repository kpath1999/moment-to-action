#!/usr/bin/env python3
"""
Simple YAMNet + Whisper Integration
- Process audio file with YAMNet chunk by chunk
- Print classification for each chunk
- When "Shout" detected, trigger Whisper transcription
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import soundfile as sf
import csv
import sys
import os
import whisper
import time
from yolo_triggered_detector import run_yolo_on_trigger, SamplingConfig
# YAMNet Configuration
SAMPLE_RATE = 16000
WINDOW_SIZE = 15600  # 0.975 seconds
HOP_SIZE = 7680      # 0.48 seconds

class SimpleYAMNetWhisper:
    """Process audio file with YAMNet, trigger Whisper on detection"""
    
    def __init__(self, yamnet_model="yamnet_quantized.tflite", 
                 labels_path="yamnet_class_map.csv",
                 whisper_model="tiny",
                 trigger_label="Shout",
                 trigger_threshold=0.30,
                 use_npu=False):
        
        self.trigger_label = trigger_label
        self.trigger_threshold = trigger_threshold
        self.whisper_model_name = whisper_model
        
        # Load YAMNet labels
        print(f"Loading YAMNet labels...")
        self.labels = {}
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.labels[int(row['index'])] = row['display_name']
        print(f"âœ“ Loaded {len(self.labels)} class labels")
        
        # Find trigger index
        self.trigger_idx = None
        for idx, label in self.labels.items():
            if self.trigger_label.lower() in label.lower():
                self.trigger_idx = idx
                print(f"âœ“ Trigger class found: '{label}' (index {idx})")
                break
        
        if self.trigger_idx is None:
            print(f"âš  Warning: Trigger '{trigger_label}' not found in labels")
        
        # Load YAMNet model
        print(f"\nLoading YAMNet model (backend: {'NPU' if use_npu else 'CPU'})...")
        
        experimental_delegates = []
        if use_npu:
            try:
                experimental_delegates = [
                    load_delegate("libQnnTFLiteDelegate.so", 
                                options={"backend_type": "htp"})
                ]
                print("âœ“ NPU delegate loaded")
            except Exception as e:
                print(f"âš  NPU failed, using CPU: {e}")
        
        self.interpreter = Interpreter(
            model_path=yamnet_model,
            experimental_delegates=experimental_delegates
        )
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Warmup
        dummy = np.zeros(WINDOW_SIZE, dtype=np.float32)
        self.run_yamnet(dummy)
        print("âœ“ YAMNet ready")
        
        # Load Whisper model
        print(f"\nLoading Whisper {whisper_model} model...")
        self.whisper_model = whisper.load_model(whisper_model)
        print("âœ“ Whisper ready")

        self.yolo_config = SamplingConfig(
        video_path="fighting_audio_rt2.mp4",
        model_path="yolov8_det.onnx",
        )
        #self.sampler.start()

    def run_yamnet(self, audio_chunk):
        """Run YAMNet inference on audio chunk"""
        # Ensure correct length
        if len(audio_chunk) != WINDOW_SIZE:
            if len(audio_chunk) < WINDOW_SIZE:
                audio_chunk = np.pad(audio_chunk, (0, WINDOW_SIZE - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:WINDOW_SIZE]
        
        # Normalize
        if np.abs(audio_chunk).max() > 0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        audio_chunk = audio_chunk.astype(np.float32)
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], audio_chunk)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle output shape
        if len(scores.shape) > 1:
            scores = scores[0] if scores.shape[0] == 1 else np.mean(scores, axis=0)
        
        return scores
    
    def process_file(self, audio_file):
        """Process audio file with YAMNet, trigger Whisper on detection"""
        
        print(f"\n{'='*80}")
        print(f"Processing: {audio_file}")
        print(f"{'='*80}")
        
        # Load audio
        print(f"\nLoading audio file...")
        audio_data, sr = sf.read(audio_file, dtype='float32')
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
            from scipy import signal
            num_samples = int(len(audio_data) * SAMPLE_RATE / sr)
            audio_data = signal.resample(audio_data, num_samples)
        
        # Convert to mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        duration = len(audio_data) / SAMPLE_RATE
        print(f"âœ“ Audio loaded: {duration:.2f}s, {SAMPLE_RATE}Hz")
        
        # Process with YAMNet
        print(f"\n{'='*80}")
        print("YAMNet Analysis (chunk by chunk)")
        print(f"{'='*80}")
        print(f"Chunk size: {WINDOW_SIZE/SAMPLE_RATE:.3f}s")
        print(f"Hop size: {HOP_SIZE/SAMPLE_RATE:.3f}s")
        print(f"Trigger: '{self.trigger_label}' (threshold: {self.trigger_threshold:.0%})")
        print(f"{'='*80}\n")
        
        position = 0
        chunk_num = 0
        trigger_detected = False
        
        while position + WINDOW_SIZE <= len(audio_data):
            # Get window
            window = audio_data[position:position + WINDOW_SIZE]
            
            # Run YAMNet
            scores = self.run_yamnet(window)
            
            # Get top 5 predictions
            top_indices = np.argsort(scores)[-5:][::-1]
            top_predictions = [(self.labels[idx], scores[idx]) for idx in top_indices]
            
            # Calculate timestamp
            timestamp = position / SAMPLE_RATE
            
            # Print chunk analysis
            chunk_num += 1
            print(f"Chunk {chunk_num:3d} @ {timestamp:6.2f}s:")
            
            for i, (label, score) in enumerate(top_predictions, 1):
                # Highlight trigger if detected
                if self.trigger_idx and label.lower() == self.trigger_label.lower() and score >= self.trigger_threshold:
                    prefix = "ðŸŽ¯ "
                    trigger_detected = True
                else:
                    prefix = "   "
                
                print(f"{prefix}{i}. {label:40s} {score*100:5.1f}%")
            
            # Check for trigger
            if self.trigger_idx is not None:
                trigger_score = scores[self.trigger_idx]
                if trigger_score >= self.trigger_threshold:
                    print(f"\nâš¡ TRIGGER DETECTED! '{self.trigger_label}' ({trigger_score*100:.1f}%)")
                    trigger_detected = True
            
            print()  # Blank line between chunks
            
            # Move to next window
            position += HOP_SIZE
        
        print(f"{'='*80}")
        print(f"Processed {chunk_num} chunks")
        print(f"{'='*80}\n")
        
        # If trigger detected, run Whisper
        if trigger_detected:
            print(f"\n{'='*80}")
            print("ðŸŽ¤ TRIGGER DETECTED - Running Whisper Transcription")
            print(f"{'='*80}\n")
            
            print("Transcribing full audio file...")
            start_time = time.time()
            
            result = self.whisper_model.transcribe(audio_file)
            transcript = result["text"].strip()
            
            elapsed = time.time() - start_time
            
            print(f"\n{'='*80}")
            print("TRANSCRIPT")
            print(f"{'='*80}")
            print(transcript)
            print(f"{'='*80}")
            print(f"â±ï¸  Transcription time: {elapsed:.2f}s\n")
            
            # Save transcript
            output_file = audio_file.rsplit('.', 1)[0] + '_transcript.txt'
            with open(output_file, 'w') as f:
                f.write(f"Trigger: {self.trigger_label}\n")
                f.write(f"Timestamp: Multiple detections\n")
                f.write(f"\nTranscript:\n{transcript}\n")
            
            print(f"âœ“ Transcript saved: {output_file}\n")

            print(f"Running yolo for object detection\n")
            run_yolo_on_trigger(self.yolo_config)
            
        else:
            print(f"\n{'='*80}")
            print(f"â„¹ï¸  No '{self.trigger_label}' detected above {self.trigger_threshold:.0%} threshold")
            print("   Whisper transcription skipped")
            print(f"{'='*80}\n")


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print("""
Simple YAMNet + Whisper Integration

Usage: python3 simple_trigger.py <audio_file.wav> [options]

Options:
  --trigger LABEL    Trigger class to detect (default: "Shout")
  --threshold VALUE  Detection threshold 0.0-1.0 (default: 0.30)
  --whisper-model    Whisper model: tiny, base, small (default: tiny)
  --use-npu          Use NPU for YAMNet (default: CPU)
  --help             Show this help

Examples:
  # Basic usage (detect "Shout")
  python3 simple_trigger.py audio.wav
  
  # Detect screaming with higher threshold
  python3 simple_trigger.py audio.wav --trigger "Scream" --threshold 0.40
  
  # Use NPU and better Whisper model
  python3 simple_trigger.py audio.wav --use-npu --whisper-model base

How it works:
  1. YAMNet processes audio file chunk by chunk (every 480ms)
  2. Prints top 5 classifications for each chunk
  3. If trigger detected (e.g., "Shout" > 30%), runs Whisper on full file
  4. Saves transcript to _transcript.txt

Common trigger classes:
  - Shout
  - Scream, Screaming
  - Yell
  - Speech
  - Male speech, man speaking
  - Female speech, woman speaking
  - Crying, sobbing

Installation:
  pip install numpy soundfile scipy ai-edge-litert openai-whisper --break-system-packages
  
Required files:
  - yamnet_quantized.tflite
  - yamnet_class_map.csv
        """)
        return
    
    # Parse arguments
    audio_file = sys.argv[1]
    trigger_label = "Shout"
    trigger_threshold = 0.30
    whisper_model = "tiny"
    use_npu = False
    
    for i, arg in enumerate(sys.argv):
        if arg == '--trigger' and i + 1 < len(sys.argv):
            trigger_label = sys.argv[i + 1]
        elif arg == '--threshold' and i + 1 < len(sys.argv):
            trigger_threshold = float(sys.argv[i + 1])
        elif arg == '--whisper-model' and i + 1 < len(sys.argv):
            whisper_model = sys.argv[i + 1]
        elif arg == '--use-npu':
            use_npu = True
    
    # Validate audio file
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return
    
    # Create processor
    processor = SimpleYAMNetWhisper(
        trigger_label=trigger_label,
        trigger_threshold=trigger_threshold,
        whisper_model=whisper_model,
        use_npu=use_npu
    )
    
    # Process file
    processor.process_file(audio_file)


if __name__ == "__main__":
    main()
