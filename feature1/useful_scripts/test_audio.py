# Quick test: Generate a simple sine wave for testing
import numpy as np
import soundfile as sf

# Generate 1 second of 440 Hz tone at 16kHz
sample_rate = 16000
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)

# Save test audio
sf.write('test_audio.wav', audio, sample_rate)
print("Created test_audio.wav")
