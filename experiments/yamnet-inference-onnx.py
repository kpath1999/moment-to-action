#!/usr/bin/env python3
"""
YamNet Audio Event Detection for Moment-to-Action
Detects specific audio events (fights, altercations, distress) to trigger video capture
"""

import numpy as np
import onnxruntime as ort
import soundfile as sf
import resampy
import librosa
import csv
from pathlib import Path

class YamNetDetector:
    """YamNet-based audio event detector for low-power monitoring"""
    
    # YamNet expects 16kHz mono audio
    SAMPLE_RATE = 16000
    
    # Target classes for altercation/distress detection
    # These are example class names from AudioSet - adjust based on your needs
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
        'Speech',  # Can indicate verbal altercation
    ]
    
    def __init__(self, model_path, class_names_path=None, confidence_threshold=0.3):
        """
        Initialize YamNet detector
        
        Args:
            model_path: Path to YamNet ONNX model
            class_names_path: Path to class names CSV (optional)
            confidence_threshold: Minimum confidence to trigger alert
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load ONNX model
        print(f"Loading YamNet model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        
        # Get input/output details
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.input_dtype = input_meta.type
        
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model input: {self.input_name}")
        print(f"  Shape: {self.input_shape}")
        print(f"  Type: {self.input_dtype}")
        print(f"Model outputs: {self.output_names}")
        
        # Load class names if provided
        self.class_names = self._load_class_names(class_names_path)
        
    def _load_class_names(self, class_names_path):
        """Load AudioSet class names from CSV file"""
        if class_names_path is None or not Path(class_names_path).exists():
            print("No class names file provided, using indices")
            return None
            
        class_names = []
        with open(class_names_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    class_names.append(row[1])  # Class name is in second column
        
        print(f"Loaded {len(class_names)} class names")
        return class_names
    
    def load_audio(self, audio_path):
        """
        Load and preprocess audio file for YamNet
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        # Load audio file
        audio, sr = sf.read(audio_path, dtype='float32')
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample to 16kHz if needed
        if sr != self.SAMPLE_RATE:
            print(f"Resampling from {sr}Hz to {self.SAMPLE_RATE}Hz")
            audio = resampy.resample(audio, sr, self.SAMPLE_RATE)
        
        return audio
    
    def audio_to_melspectrogram(self, audio, target_shape=(96, 64)):
        """
        Convert audio waveform to mel-spectrogram
        
        Args:
            audio: Audio waveform (1D numpy array)
            target_shape: Target mel-spectrogram shape (n_mels, n_frames)
            
        Returns:
            mel_spec: Mel-spectrogram as uint8 (n_mels, n_frames)
        """
        n_mels, n_frames = target_shape
        
        # Calculate required audio length for target frames
        # YamNet uses hop_length=160 (10ms at 16kHz), frame_length=400 (25ms)
        hop_length = 160
        n_fft = 400
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=125.0,
            fmax=7500.0
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Handle different number of frames
        if mel_spec_norm.shape[1] < n_frames:
            # Pad with zeros if too short
            pad_width = n_frames - mel_spec_norm.shape[1]
            mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec_norm.shape[1] > n_frames:
            # Truncate if too long
            mel_spec_norm = mel_spec_norm[:, :n_frames]
        
        # Convert to uint8 [0, 255]
        mel_spec_uint8 = (mel_spec_norm * 255).astype(np.uint8)
        
        return mel_spec_uint8
    
    def softmax(self, x):
        """Apply softmax to convert logits to probabilities"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def predict(self, audio):
        """
        Run YamNet inference on audio
        
        Args:
            audio: Audio waveform (1D numpy array, float32 normalized to [-1, 1])
            
        Returns:
            scores: Class scores for the audio segment
        """
        # Model expects [1, 1, 96, 64] - mel-spectrogram input
        n_mels = self.input_shape[2]  # 96
        n_frames = self.input_shape[3]  # 64
        
        # Convert audio to mel-spectrogram
        mel_spec = self.audio_to_melspectrogram(audio, target_shape=(n_mels, n_frames))
        
        # Reshape to model input: [batch, channels, height, width]
        # [1, 1, 96, 64]
        mel_spec_4d = mel_spec.reshape(1, 1, n_mels, n_frames)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: mel_spec_4d})
        
        # Parse outputs - shape is typically [1, num_classes]
        scores = outputs[0]
        
        # Remove batch dimension if present
        if len(scores.shape) > 1:
            scores = scores[0]
        
        print(f"Raw output dtype: {scores.dtype}")
        print(f"Raw scores - min: {scores.min():.2f}, max: {scores.max():.2f}, mean: {scores.mean():.2f}")
        print(f"Output shape: {scores.shape}")
        
        # This appears to be a quantized model
        # The outputs are uint8 values that need to be dequantized
        # Typical dequantization: (uint8_value / 255.0) to get [0, 1] range
        if scores.dtype == np.uint8:
            print("Detected uint8 output - applying dequantization")
            scores = scores.astype(np.float32) / 255.0
        else:
            # If it's already float, normalize from the observed range
            # Values around 96-213 suggest a 0-255 scale
            print("Normalizing output scores")
            scores = scores.astype(np.float32) / 255.0
        
        print(f"After processing - min: {scores.min():.6f}, max: {scores.max():.6f}")
        
        return scores
    
    def detect_events(self, audio_path, verbose=True):
        """
        Detect audio events in file and check for trigger conditions
        
        Args:
            audio_path: Path to audio file
            verbose: Print detailed results
            
        Returns:
            dict with detection results and trigger status
        """
        # Load and preprocess audio
        audio = self.load_audio(audio_path)
        duration = len(audio) / self.SAMPLE_RATE
        
        # Run inference
        scores = self.predict(audio)
        
        # scores shape is [1, 521] or [521] - single prediction for the audio segment
        if len(scores.shape) > 1:
            scores = scores[0]  # Remove batch dimension
        
        # Get top predictions
        results = {
            'audio_path': audio_path,
            'duration': duration,
            'trigger': False,
            'detections': [],
            'max_confidence': 0.0,
            'triggered_classes': []
        }
        
        # Get top classes
        top_indices = np.argsort(scores)[::-1][:10]
        
        for class_idx in top_indices:
            confidence = scores[class_idx]
            
            if confidence < self.confidence_threshold:
                continue
                
            class_name = self.class_names[class_idx] if self.class_names else f"Class_{class_idx}"
            
            # Check if this is a target class
            is_target = any(target.lower() in class_name.lower() for target in self.TARGET_CLASSES)
            
            detection = {
                'class': class_name,
                'confidence': float(confidence),
                'is_target': is_target
            }
            
            results['detections'].append(detection)
            
            # Update trigger status
            if is_target and confidence > results['max_confidence']:
                results['max_confidence'] = float(confidence)
                if class_name not in results['triggered_classes']:
                    results['triggered_classes'].append(class_name)
                results['trigger'] = True
        
        # Print results if verbose
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """Pretty print detection results"""
        print(f"\n{'='*60}")
        print(f"Audio: {results['audio_path']}")
        print(f"Duration: {results['duration']:.2f}s")
        print(f"{'='*60}")
        
        if results['trigger']:
            print(f"\n🚨 TRIGGER ACTIVATED 🚨")
            print(f"Max confidence: {results['max_confidence']:.3f}")
            print(f"Triggered classes: {', '.join(results['triggered_classes'])}")
            print(f"\n>>> ACTION: Wake video pipeline for detailed analysis <<<")
        else:
            print(f"\n✓ No trigger events detected")
        
        # Show top detections
        if results['detections']:
            print(f"\nTop detections (>= {self.confidence_threshold}):")
            target_detections = [d for d in results['detections'] if d['is_target']]
            other_detections = [d for d in results['detections'] if not d['is_target']]
            
            if target_detections:
                print("\n  TARGET EVENTS:")
                for det in target_detections:
                    print(f"    {det['class']:<30} {det['confidence']:.3f}")
            
            if other_detections and len(other_detections) > 0:
                print("\n  OTHER EVENTS:")
                for det in other_detections[:5]:
                    print(f"    {det['class']:<30} {det['confidence']:.3f}")
        
        print(f"{'='*60}\n")
    
    def monitor_audio_stream(self, audio_path, window_size=3.0, overlap=1.0):
        """
        Simulate real-time monitoring by processing audio in sliding windows
        
        Args:
            audio_path: Path to audio file
            window_size: Window size in seconds
            overlap: Overlap between windows in seconds
        """
        audio = self.load_audio(audio_path)
        
        window_samples = int(window_size * self.SAMPLE_RATE)
        hop_samples = int((window_size - overlap) * self.SAMPLE_RATE)
        
        print(f"Monitoring audio in {window_size}s windows with {overlap}s overlap...")
        print(f"Audio duration: {len(audio) / self.SAMPLE_RATE:.2f}s\n")
        
        for start_idx in range(0, len(audio) - window_samples, hop_samples):
            end_idx = start_idx + window_samples
            window_audio = audio[start_idx:end_idx]
            
            # Run inference on window
            scores = self.predict(window_audio)
            
            # Remove batch dimension if present
            if len(scores.shape) > 1:
                scores = scores[0]
            
            # Check for triggers
            max_confidence = 0.0
            triggered_class = None
            
            top_idx = np.argmax(scores)
            confidence = scores[top_idx]
            
            if confidence > max_confidence:
                max_confidence = confidence
                class_name = self.class_names[top_idx] if self.class_names else f"Class_{top_idx}"
                
                is_target = any(target.lower() in class_name.lower() for target in self.TARGET_CLASSES)
                if is_target and confidence >= self.confidence_threshold:
                    triggered_class = class_name
            
            # Report status
            timestamp = start_idx / self.SAMPLE_RATE
            if triggered_class:
                print(f"[{timestamp:6.2f}s] 🚨 TRIGGER: {triggered_class} (conf: {max_confidence:.3f})")
            else:
                top_class = self.class_names[top_idx] if self.class_names else f"Class_{top_idx}"
                print(f"[{timestamp:6.2f}s] ✓ {top_class[:30]} (conf: {max_confidence:.3f})")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YamNet Audio Event Detection')
    parser.add_argument('--model', type=str, required=True, help='Path to YamNet ONNX model')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--classes', type=str, help='Path to class names CSV (optional)')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--monitor', action='store_true', help='Simulate real-time monitoring')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YamNetDetector(
        model_path=args.model,
        class_names_path=args.classes,
        confidence_threshold=args.threshold
    )
    
    # Run detection
    if args.monitor:
        detector.monitor_audio_stream(args.audio)
    else:
        results = detector.detect_events(args.audio)
        
        # Return exit code based on trigger status
        return 0 if not results['trigger'] else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
