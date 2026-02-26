#!/usr/bin/env python3
"""
YamNet Audio Event Detection using TensorFlow Lite
Optimized for ARM devices like Rubix Pi 3
"""

import numpy as np
import soundfile as sf
import resampy
import csv
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class YamNetTFLiteDetector:
    """YamNet detector using TensorFlow Lite"""
    
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
    ]
    
    def __init__(self, model_path, class_names_path=None, confidence_threshold=0.3):
        """
        Initialize YamNet TFLite detector
        
        Args:
            model_path: Path to YamNet TFLite model (.tflite)
            class_names_path: Path to class names CSV
            confidence_threshold: Minimum confidence to trigger
        """
        self.confidence_threshold = confidence_threshold
        
        # Load TFLite model
        print(f"Loading YamNet TFLite model from {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded successfully")
        print(f"  Input: {self.input_details[0]['name']}, shape: {self.input_details[0]['shape']}, dtype: {self.input_details[0]['dtype']}")
        print(f"  Output: {self.output_details[0]['name']}, shape: {self.output_details[0]['shape']}, dtype: {self.output_details[0]['dtype']}")
        
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
            print(f"Loaded {len(class_names)} class names")
            return class_names
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
        Run YamNet TFLite inference
        
        Args:
            audio: Audio waveform (1D numpy array)
            
        Returns:
            scores: Class scores
        """
        # Prepare input
        input_data = audio.astype(np.float32)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return scores
    
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
        scores = self.predict(audio)
        
        # Handle different output shapes
        if len(scores.shape) == 3:
            # Shape: (1, num_frames, num_classes)
            scores = scores[0]  # Remove batch dimension
            mean_scores = scores.mean(axis=0)
            max_scores = scores.max(axis=0)
            num_frames = scores.shape[0]
        elif len(scores.shape) == 2:
            # Shape: (num_frames, num_classes)
            mean_scores = scores.mean(axis=0)
            max_scores = scores.max(axis=0)
            num_frames = scores.shape[0]
        else:
            # Shape: (num_classes,)
            mean_scores = scores
            max_scores = scores
            num_frames = 1
        
        results = {
            'audio_path': audio_path,
            'duration': duration,
            'num_frames': num_frames,
            'trigger': False,
            'detections': [],
            'max_confidence': 0.0,
            'triggered_classes': []
        }
        
        # Get top predictions
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


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YamNet TFLite Audio Event Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to YamNet TFLite model (.tflite)')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--classes', type=str, help='Path to class names CSV')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YamNetTFLiteDetector(
        model_path=args.model,
        class_names_path=args.classes,
        confidence_threshold=args.threshold
    )
    
    # Run detection
    results = detector.detect_events(args.audio)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
