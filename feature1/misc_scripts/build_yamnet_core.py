import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import os
import soundfile as sf
import librosa

# --- 1. Define YAMNet Core Architecture ---
def conv_bn_relu(inputs, filters, kernel_size, strides):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)(x)
    x = layers.ReLU()(x)
    return x

def depthwise_separable_conv(inputs, filters_dw, filters_pw, strides):
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters_pw, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4)(x)
    x = layers.ReLU()(x)
    return x

def build_yamnet_core():
    input_tensor = layers.Input(shape=(96, 64, 1))
    x = conv_bn_relu(input_tensor, filters=32, kernel_size=3, strides=2)
    blocks = [
        (32, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2),
        (256, 256, 1), (256, 512, 2), (512, 512, 1), (512, 512, 1),
        (512, 512, 1), (512, 512, 1), (512, 512, 1), (512, 1024, 2),
        (1024, 1024, 1)
    ]
    for (dw, pw, stride) in blocks:
        x = depthwise_separable_conv(x, dw, pw, stride)
    x = layers.GlobalAveragePooling2D()(x)
    predictions = layers.Dense(521, activation='sigmoid')(x)
    return Model(inputs=input_tensor, outputs=predictions, name="yamnet_core")

# --- 2. Load Weights ---
WEIGHTS_FILE = 'yamnet.h5'
WEIGHTS_URL = 'https://storage.googleapis.com/audioset/yamnet.h5'

if not os.path.exists(WEIGHTS_FILE):
    utils = tf.keras.utils.get_file(WEIGHTS_FILE, WEIGHTS_URL, cache_dir='.', cache_subdir='.')

model = build_yamnet_core()
model.load_weights(WEIGHTS_FILE, by_name=True, skip_mismatch=True)
print("Model built and weights loaded.")

# --- 3. PREPARE REAL DATA FOR QUANTIZATION ---
# We generate a real spectrogram from a file to teach the quantizer the correct ranges (-7 to +5)
AUDIO_FILE = "test_audio.wav"  # Ensure this file exists in your folder!

def generate_spectrogram(audio_path):
    # Standard YAMNet Preprocessing
    wav, sr = sf.read(audio_path, dtype='float32')
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
        
    # Spectrogram Parameters
    n_fft = int(16000 * 0.025)
    hop_length = int(16000 * 0.010)
    spectrogram = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, window='hann', center=False))**2
    
    # Mel Features
    mel_basis = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=64, fmin=125, fmax=7500)
    mel_spectrogram = np.dot(mel_basis, spectrogram)
    
    # Log Scaling (The critical part: output is ~ -7.0 to +3.0)
    log_mel = np.log(mel_spectrogram + 0.001).T
    
    # Pad/Crop to 96 frames
    if log_mel.shape[0] < 96:
        padding = np.zeros((96 - log_mel.shape[0], 64))
        # Important: Pad with "Silence" value (log(0.001) = -6.9), not 0
        padding[:] = np.log(0.001) 
        log_mel = np.vstack((log_mel, padding))
    
    # Create multiple 96-frame patches if the audio is long enough
    patches = []
    num_patches = log_mel.shape[0] // 96
    if num_patches == 0:
        patches.append(log_mel[:96])
    else:
        for i in range(num_patches):
            patches.append(log_mel[i*96:(i+1)*96])
            
    return np.array(patches)

# --- 4. Convert to TFLite (Int8) with Real Data ---
print(f"Generating calibration data from {AUDIO_FILE}...")
try:
    real_spectrograms = generate_spectrogram(AUDIO_FILE)
    print(f"Created {len(real_spectrograms)} patches for calibration.")
except Exception as e:
    print(f"WARNING: Could not load audio ({e}). Falling back to synthetic stats.")
    # Fallback to synthetic but ACCURATE ranges (-7 to +5)
    real_spectrograms = np.random.uniform(-7.0, 5.0, size=(10, 96, 64)).astype(np.float32)

def representative_dataset():
    for i in range(len(real_spectrograms)):
        # Reshape to [1, 96, 64, 1]
        data = real_spectrograms[i][np.newaxis, :, :, np.newaxis].astype(np.float32)
        yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

OUTPUT_FILE = "yamnet_core_int8.tflite"
with open(OUTPUT_FILE, "wb") as f:
    f.write(tflite_model)

print(f"SUCCESS! Saved calibrated model to {OUTPUT_FILE}")
