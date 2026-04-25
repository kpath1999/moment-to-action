"""driver_discovery.py — ONNX backend discovery and latency test."""

import sys
import time

import numpy as np
import onnxruntime as ort

# -----------------------------
# Model path
# -----------------------------
model_path = (
    sys.argv[1] if len(sys.argv) > 1 else "src/moment_to_action/models/_vendored/yolo/model.onnx"
)

# -----------------------------
# Dummy input helper
# -----------------------------


def get_dummy_input(session: ort.InferenceSession) -> np.ndarray:
    """Return a dummy input tensor matching the session's first input shape."""
    input_shape = session.get_inputs()[0].shape
    shape = [1 if isinstance(dim, str) or dim is None else dim for dim in input_shape]
    rng = np.random.default_rng()
    return rng.random(shape, dtype=np.float32)


# -----------------------------
# Run inference on a backend
# -----------------------------


def run_backend(name: str, providers: list) -> bool | None:
    """Run inference on a backend and print latency."""
    print(f"\n===== Running on {name} =====")
    try:
        so = ort.SessionOptions()
        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=providers,
        )
        print("Active providers:", sess.get_providers())
        dummy = get_dummy_input(sess)
        input_name = sess.get_inputs()[0].name
        start = time.time()
        _ = sess.run(None, {input_name: dummy})
        end = time.time()
        print(f"{name} inference OK")
        print(f"Latency: {(end - start) * 1000:.2f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"{name} FAILED:")
        print(exc)
        return None
    else:
        return True


# -----------------------------
# Provider configs
# -----------------------------

CPU_PROVIDERS = ["CPUExecutionProvider"]

NPU_PROVIDERS = [
    (
        "QNNExecutionProvider",
        {
            "backend_type": "htp",
            "profiling_level": "detailed",
            "profiling_file_path": "qnn_profile_npu.csv",
        },
    ),
    "CPUExecutionProvider",
]

GPU_PROVIDERS = [
    (
        "QNNExecutionProvider",
        {
            "backend_type": "gpu",
            "profiling_level": "detailed",
            "profiling_file_path": "qnn_profile_gpu.csv",
        },
    ),
    "CPUExecutionProvider",
]


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Model:", model_path)

    results = {}

    results["CPU"] = run_backend("CPU", CPU_PROVIDERS)
    results["NPU (QNN HTP)"] = run_backend("NPU", NPU_PROVIDERS)
    results["GPU (QNN)"] = run_backend("GPU", GPU_PROVIDERS)

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k}: {'OK' if v else 'FAILED'}")
