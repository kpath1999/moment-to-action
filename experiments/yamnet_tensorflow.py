#!/usr/bin/env python3
"""
YamNet Audio Event Detection using TensorFlow
This uses the official YamNet model from TensorFlow Hub
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import resampy
import csv
from pathlib import Path

class YamNetTFDetector:
    """YamNet detector using TensorFlow"""
    
    SAMPLE_RATE = 16000
    
    TARGET_CLASSES = [
        'Screaming',
        'Shout',
        'Battle cry',
        'Yell',
        'Crying, sobbing',
        'Whimper',
        'Glass breaking',
        'Bang',
        'Thump, thud',
        'Crash',
        'Gunshot, gunfire',
        'Speech',
        'Laughter',
        'Clapping',
        'Fighting',
        'Crowd',
        'Shouting',
    ]
    
    def __init__(self, class_names_path=None, confidence_threshold=0.3):
        """
        Initialize YamNet detector
        
        Args:
            class_names_path: Path to class names CSV
            confidence_threshold: Minimum confidence to trigger
        """
        self.confidence_threshold = confidence_threshold
        
        # Load YamNet model from TensorFlow Hub
        print("Loading YamNet model from TensorFlow Hub...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("✓ Model loaded successfully")
        
        # Load class names
        self.class_names = self._load_class_names(class_names_path)
        
    def _load_class_names(self, class_names_path):
        """Load AudioSet class names"""
        if class_names_path and Path(class_names_path).exists():
            class_names = []
            with open(class_names_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        class_names.append(row[2])  # Display name
            print(f"Loaded {len(class_names)} class names from {class_names_path}")
            return class_names
        else:
            # Use default class names from the model
            print("Using default YamNet class names")
            return None
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz if needed
        if sr != self.SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {self.SAMPLE_RATE}Hz")
            audio = resampy.resample(audio, sr, self.SAMPLE_RATE)
        
        return audio
    
    def predict(self, audio):
        """
        Run YamNet inference
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            scores: Class scores per frame
            embeddings: Audio embeddings
            spectrogram: Log mel spectrogram
        """
        # YamNet expects a 1D waveform
        scores, embeddings, spectrogram = self.model(audio)
        
        return scores.numpy(), embeddings.numpy(), spectrogram.numpy()
    
    def detect_events(self, audio_path, verbose=True):
        """
        Detect audio events in file
        
        Args:
            audio_path: Path to audio file
            verbose: Print detailed results
            
        Returns:
            dict with detection results and trigger status
        """
        # Load audio
        audio = self.load_audio(audio_path)
        duration = len(audio) / self.SAMPLE_RATE
        
        # Run inference
        scores, embeddings, spectrogram = self.predict(audio)
        
        # scores shape: (num_frames, 521)
        # Average scores across all frames
        mean_scores = scores.mean(axis=0)
        max_scores = scores.max(axis=0)
        
        results = {
            'audio_path': audio_path,
            'duration': duration,
            'num_frames': len(scores),
            'trigger': False,
            'detections': [],
            'max_confidence': 0.0,
            'triggered_classes': []
        }
        
        # Get top predictions based on max scores
        top_indices = np.argsort(max_scores)[::-1][:15]
        
        for rank, class_idx in enumerate(top_indices):
            mean_conf = float(mean_scores[class_idx])
            max_conf = float(max_scores[class_idx])
            
            class_name = self.class_names[class_idx] if self.class_names else f"Class_{class_idx}"
            
            # Check if target class
            is_target = any(target.lower() in class_name.lower() for target in self.TARGET_CLASSES)
            
            detection = {
                'rank': rank + 1,
                'class': class_name,
                'mean_confidence': mean_conf,
                'max_confidence': max_conf,
                'is_target': is_target
            }
            
            results['detections'].append(detection)
            
            # Trigger if target class exceeds threshold
            if is_target and max_conf >= self.confidence_threshold:
                if max_conf > results['max_confidence']:
                    results['max_confidence'] = max_conf
                if class_name not in results['triggered_classes']:
                    results['triggered_classes'].append(class_name)
                results['trigger'] = True
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """Pretty print results"""
        print(f"\n{'='*80}")
        print(f"Audio: {results['audio_path']}")
        print(f"Duration: {results['duration']:.2f}s")
        print(f"Frames analyzed: {results['num_frames']}")
        print(f"{'='*80}")
        
        if results['trigger']:
            print(f"\n🚨 TRIGGER ACTIVATED 🚨")
            print(f"Max confidence: {results['max_confidence']:.3f}")
            print(f"Triggered classes: {', '.join(results['triggered_classes'])}")
            print(f"\n>>> ACTION: Wake video pipeline for detailed analysis <<<")
        else:
            print(f"\n✓ No trigger events detected")
            print(f"  (Max confidence: {results['max_confidence']:.3f}, threshold: {self.confidence_threshold})")
        
        # Show predictions
        print(f"\nTop Predictions:")
        print(f"{'Rank':<6} {'Class':<40} {'Mean':<10} {'Max':<10} {'Target'}")
        print("-" * 80)
        
        for det in results['detections']:
            target_mark = "🎯" if det['is_target'] else ""
            print(f"{det['rank']:<6} {det['class']:<40} {det['mean_confidence']:>8.3f}  {det['max_confidence']:>8.3f}  {target_mark}")
        
        print(f"{'='*80}\n")
    
    def monitor_audio_stream(self, audio_path, window_size=3.0, overlap=1.0):
        """
        Simulate real-time monitoring with sliding windows
        
        Args:
            audio_path: Path to audio file
            window_size: Window size in seconds
            overlap: Overlap in seconds
        """
        audio = self.load_audio(audio_path)
        
        window_samples = int(window_size * self.SAMPLE_RATE)
        hop_samples = int((window_size - overlap) * self.SAMPLE_RATE)
        
        print(f"Monitoring audio in {window_size}s windows with {overlap}s overlap...")
        print(f"Audio duration: {len(audio) / self.SAMPLE_RATE:.2f}s\n")
        
        for start_idx in range(0, len(audio) - window_samples, hop_samples):
            end_idx = start_idx + window_samples
            window_audio = audio[start_idx:end_idx]
            
            # Run inference
            scores, _, _ = self.predict(window_audio)
            
            # Get max score across frames
            max_scores = scores.max(axis=0)
            top_idx = np.argmax(max_scores)
            confidence = max_scores[top_idx]
            
            class_name = self.class_names[top_idx] if self.class_names else f"Class_{top_idx}"
            is_target = any(target.lower() in class_name.lower() for target in self.TARGET_CLASSES)
            
            timestamp = start_idx / self.SAMPLE_RATE
            
            if is_target and confidence >= self.confidence_threshold:
                print(f"[{timestamp:6.2f}s] 🚨 TRIGGER: {class_name} (conf: {confidence:.3f})")
            else:
                print(f"[{timestamp:6.2f}s] ✓ {class_name[:40]} (conf: {confidence:.3f})")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YamNet TensorFlow Audio Event Detection')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--classes', type=str, help='Path to class names CSV (optional)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--monitor', action='store_true', help='Simulate real-time monitoring')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YamNetTFDetector(
        class_names_path=args.classes,
        confidence_threshold=args.threshold
    )
    
    # Run detection
    if args.monitor:
        detector.monitor_audio_stream(args.audio)
    else:
        results = detector.detect_events(args.audio)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
