import argparse
import numpy as np
from PIL import Image
import urllib.request
import json

# Import LiteRT (Google's rebranded TFLite runtime)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter

def load_labels_and_shift():
    """
    Load ImageNet class labels.
    Returns a dictionary where keys are indices and values are label names.
    """
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
        
        # Standard ImageNet mapping: { "0": ["n01440764", "tench"], ... }
        # Create a list where index maps to the human-readable label
        labels_map = {}
        for k, v in data.items():
            labels_map[int(k)] = v[1]
        return labels_map
    except Exception as e:
        print(f"⚠️  Could not load labels: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to .tflite model')
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()
    
    labels_map = load_labels_and_shift()

    # 1. Load Model
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Preprocess Image
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    
    img = Image.open(args.image).convert('RGB')
    img = img.resize((width, height))
    input_data = np.expand_dims(img, axis=0)

    # --- CRITICAL FIX: Normalization ---
    # EfficientNet TFLite models from AI Hub typically expect [-1, 1]
    # Formula: (pixel - 127.5) / 127.5
    # If the results are still wrong, try changing this to [0, 1] (pixel / 255.0)
    
    input_dtype = input_details[0]['dtype']
    if input_dtype == np.float32:
        print("ℹ️  Model expects Float32. Applying Normalization: [-1, 1]")
        input_data = (np.float32(input_data) - 127.5) / 127.5
        # ALTERNATIVE IF THIS FAILS:
        # input_data = np.float32(input_data) / 255.0  # [0, 1] normalization
        # input_data = (input_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] # ImageNet Mean/Std
    elif input_dtype == np.uint8:
        print("ℹ️  Model expects Uint8. Using raw pixel values [0-255].")
        input_data = np.uint8(input_data)

    # 3. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # 4. Process Results
    predictions = np.squeeze(output_data)
    top_index = np.argmax(predictions)
    confidence = predictions[top_index]
    
    print(f"\n✅ Inference Complete")
    print(f"Predicted Class Index: {top_index}")
    print(f"Confidence Score: {confidence:.4f}")

    # --- CRITICAL FIX: Label Offset Handling ---
    # TFLite models often have a background class at index 0.
    # If the model output size is 1001, we shift the label index by -1.
    output_size = predictions.shape[0]
    
    mapped_label = "Unknown"
    if labels_map:
        if output_size == 1001:
            # Model has background class at 0. ImageNet labels start at 1.
            # So index 976 in model means 975 in standard ImageNet list.
            print(f"ℹ️  Detected 1001 classes (Background + ImageNet). Adjusting index by -1.")
            mapped_label = labels_map.get(top_index - 1, "Index out of range")
        else:
            # Standard 1000 classes
            mapped_label = labels_map.get(top_index, "Index out of range")

    print(f"🖼️  Image depicts: {mapped_label}")

if __name__ == '__main__':
    main()
