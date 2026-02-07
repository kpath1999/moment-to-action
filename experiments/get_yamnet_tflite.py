import urllib.request
import os

MODEL_PATH = "yamnet_quantized.tflite"
LABELS_PATH = "yamnet_class_map.csv"

# Download YAMNet model
if not os.path.exists(MODEL_PATH):
    print("Downloading YAMNet model...")
    model_url = 'https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite'
    urllib.request.urlretrieve(model_url, MODEL_PATH)

# Download class labels
if not os.path.exists(LABELS_PATH):
    print("Downloading class labels...")
    labels_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    urllib.request.urlretrieve(labels_url, LABELS_PATH)
