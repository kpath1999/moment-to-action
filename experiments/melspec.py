#!/usr/bin/env python3
"""
Visualize what mel-spectrograms your audio files are actually producing
"""

import numpy as np
import soundfile as sf
import resampy
import librosa
import sys

def analyze_audio_melspec(audio_path):
    """Analyze audio and its mel-spectrogram"""
    
    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        print(f"Resampling {sr}Hz → 16kHz")
        audio = resampy.resample(audio, sr, 16000)
    
    duration = len(audio) / 16000
    
    print(f"\n{'='*70}")
    print(f"Audio: {audio_path}")
    print(f"Duration: {duration:.2f}s ({len(audio)} samples)")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}], RMS: {np.sqrt(np.mean(audio**2)):.3f}")
    
    # Compute mel-spectrogram
    hop_length = 160
    n_fft = 400
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=96,
        fmin=125.0,
        fmax=7500.0
    )
    
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    print(f"Mel-spec (dB): [{mel_db.min():.1f}, {mel_db.max():.1f}] dB, mean: {mel_db.mean():.1f}")
    print(f"Mel-spec shape: {mel_spec.shape} → will be resized to (96, 64)")
    
    # Normalize
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    # Resize to 64 frames
    if mel_norm.shape[1] < 64:
        pad_width = 64 - mel_norm.shape[1]
        mel_norm_64 = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode='constant')
        print(f"Padded from {mel_norm.shape[1]} to 64 frames (added {pad_width} zero frames)")
    elif mel_norm.shape[1] > 64:
        mel_norm_64 = mel_norm[:, :64]
        print(f"Truncated from {mel_norm.shape[1]} to 64 frames")
    else:
        mel_norm_64 = mel_norm
        print("Exactly 64 frames - no padding needed")
    
    # Convert to uint8
    mel_uint8 = (mel_norm_64 * 255).astype(np.uint8)
    
    print(f"Final uint8 mel-spec:")
    print(f"  Shape: {mel_uint8.shape}")
    print(f"  Range: [{mel_uint8.min()}, {mel_uint8.max()}], mean: {mel_uint8.mean():.1f}, std: {mel_uint8.std():.1f}")
    
    # Compute histogram to see distribution
    hist, bins = np.histogram(mel_uint8.flatten(), bins=10, range=(0, 255))
    print(f"  Value distribution:")
    for i in range(len(hist)):
        bar = '█' * int(hist[i] / hist.max() * 40)
        print(f"    {bins[i]:5.1f}-{bins[i+1]:5.1f}: {bar} ({hist[i]})")
    
    # Show some statistics about energy in different frequency bands
    low_freq = mel_uint8[:32, :].mean()   # Low frequencies
    mid_freq = mel_uint8[32:64, :].mean()  # Mid frequencies  
    high_freq = mel_uint8[64:, :].mean()   # High frequencies
    
    print(f"  Frequency band energy:")
    print(f"    Low  (0-32):  {low_freq:.1f}")
    print(f"    Mid  (32-64): {mid_freq:.1f}")
    print(f"    High (64-96): {high_freq:.1f}")
    
    # Show temporal variation
    frame_energies = mel_uint8.mean(axis=0)
    print(f"  Temporal variation: std={frame_energies.std():.1f}")
    print(f"    First 10 frames avg: {frame_energies[:10].mean():.1f}")
    print(f"    Last  10 frames avg: {frame_energies[-10:].mean():.1f}")
    
    return mel_uint8


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_melspec.py <audio1.wav> [audio2.wav] ...")
        sys.exit(1)
    
    mel_specs = []
    for audio_path in sys.argv[1:]:
        mel = analyze_audio_melspec(audio_path)
        mel_specs.append((audio_path, mel))
    
    # Compare mel-spectrograms if multiple files
    if len(mel_specs) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON:")
        print(f"{'='*70}")
        
        for i in range(len(mel_specs)):
            for j in range(i+1, len(mel_specs)):
                name1 = mel_specs[i][0].split('/')[-1]
                name2 = mel_specs[j][0].split('/')[-1]
                mel1 = mel_specs[i][1]
                mel2 = mel_specs[j][1]
                
                # Compute similarity
                diff = np.abs(mel1.astype(float) - mel2.astype(float))
                mae = diff.mean()
                max_diff = diff.max()
                
                # Correlation
                corr = np.corrcoef(mel1.flatten(), mel2.flatten())[0, 1]
                
                print(f"\n{name1} vs {name2}:")
                print(f"  Mean Absolute Error: {mae:.1f} (out of 255)")
                print(f"  Max Difference: {max_diff}")
                print(f"  Correlation: {corr:.3f}")
                
                if mae < 20:
                    print(f"  ⚠️  WARNING: Mel-spectrograms are VERY SIMILAR!")
                elif mae < 40:
                    print(f"  ⚠️  Mel-spectrograms are somewhat similar")
                else:
                    print(f"  ✓ Mel-spectrograms are sufficiently different")
    
    print(f"\n{'='*70}\n")
