import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load YAMNet
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# Create a concrete function
@tf.function
def yamnet_inference(waveform):
    scores, embeddings, spectrogram = yamnet(waveform)
    return scores

# Get concrete function
concrete_func = yamnet_inference.get_concrete_function(
    waveform=tf.TensorSpec(shape=[None], dtype=tf.float32)
)

# Converter with FULL INT8 quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

# CRITICAL: Full integer quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8

# Representative dataset for calibration
def representative_dataset():
    for _ in range(100):
        # Generate random audio samples in correct range
        data = np.random.uniform(-1.0, 1.0, size=15600).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# Convert
tflite_model = converter.convert()

# Save
with open('yamnet_int8_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Fully quantized model created!")
