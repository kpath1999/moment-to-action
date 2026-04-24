import sys
import time
import os
import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

# -----------------------------
# Model path
# -----------------------------
model_path = (
    sys.argv[1] if len(sys.argv) > 1 else "src/moment_to_action/models/_vendored/yolo/model.onnx"
)

# -----------------------------
# Model Converter Helper
# -----------------------------
def get_or_create_fp16_model(source_path):
    fp16_path = source_path.replace(".onnx", "_fp16.onnx")
    if not os.path.exists(fp16_path):
        print(f"Converting {source_path} to Float16 for GPU compatibility...")
        model = onnx.load(source_path)
        # Convert the float32 ONNX model to float16
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, fp16_path)
    return fp16_path

# -----------------------------
# Dummy input helper
# -----------------------------
def get_dummy_input(session, dtype=np.float32):
    input_shape = session.get_inputs()[0].shape
    # Replace dynamic dims with 1
    shape = [
        1 if isinstance(dim, str) or dim is None else dim for dim in input_shape
    ]
    return np.random.rand(*shape).astype(dtype)

# -----------------------------
# Run inference on a backend
# -----------------------------
def run_backend(name, providers, target_model_path, dtype=np.float32):
    print(f"\n===== Running on {name} =====")
    try:
        so = ort.SessionOptions()
        # Prevent ONNX from attempting unsupported internal layout transformations at load-time
        # which trigger the "was inserted using the NHWC format... but was not selected" error
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(
            target_model_path,
            sess_options=so,
            providers=providers,
        )
        print("Active providers:", sess.get_providers())
        dummy = get_dummy_input(sess, dtype=dtype)
        input_name = sess.get_inputs()[0].name
        # Warmup
        sess.run(None, {input_name: dummy})
        # Latency check
        start = time.time()
        outputs = sess.run(None, {input_name: dummy})
        end = time.time()
        print(f"{name} inference OK")
        print(f"Latency: {(end - start) * 1000:.2f} ms")
        return True
    except Exception as e:
        print(f"{name} FAILED:")
        print(e)
        return False

# -----------------------------
# Provider configs
# -----------------------------
CPU_PROVIDERS = ["CPUExecutionProvider"]

NPU_PROVIDERS = [
    ("QNNExecutionProvider", {
        "backend_path": "libQnnHtp.so",
        "profiling_level": "detailed",
        "profiling_file_path": "qnn_profile_npu.csv",
    }),
    "CPUExecutionProvider",
]

GPU_PROVIDERS = [
    ("QNNExecutionProvider", {
        "backend_path": "libQnnGpu.so",
        "profiling_level": "detailed",
        "profiling_file_path": "qnn_profile_gpu.csv",
    }),
    "CPUExecutionProvider",
]

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Base Model:", model_path)
    
    # Generate an fp16 copy of the model for the GPU
    model_fp16_path = get_or_create_fp16_model(model_path)
    
    results = {}

    # CPU and NPU can run the base model
    results["CPU"] = run_backend("CPU", CPU_PROVIDERS, model_path, dtype=np.float32)
    
    # Assuming your NPU model is quantized, it needs the base model
    # (If you get QDQ errors here, you need to load your quantized NPU model instead)
    results["NPU (QNN HTP)"] = run_backend("NPU", NPU_PROVIDERS, model_path, dtype=np.float32)

    # GPU runs the fp16 version with optimizations disabled to bypass the NHWC rejection
    results["GPU (QNN)"] = run_backend("GPU", GPU_PROVIDERS, model_fp16_path, dtype=np.float16)

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k}: {'OK' if v else 'FAILED'}")
