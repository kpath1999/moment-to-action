"""Check TFLite delegate partitioning for a model."""

from __future__ import annotations

import argparse
import logging
import sys

from moment_to_action.hardware import ComputeBackend
from moment_to_action.hardware._types import ComputeUnit
from moment_to_action.models import ModelID, ModelManager

logging.basicConfig(level=logging.INFO, format="%(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model ID (e.g., mobileclip_s2)")
args = parser.parse_args()

# Map model names to ModelIDs
model_map = {
    "mobileclip": ModelID.MOBILECLIP_S2,
    "yolo_int8": ModelID.YOLO_V8_TFLITE_INT8,
    "yolo": ModelID.YOLO_V8_TFLITE,
}

if args.model not in model_map:
    print(f"Unknown model: {args.model}")
    print(f"Available: {list(model_map.keys())}")
    sys.exit(1)

model_id = model_map[args.model]
manager = ModelManager()
backend = ComputeBackend(preferred_unit=ComputeUnit.NPU)

print(f"\nLoading {model_id.value} on {backend.active_unit.name}...")
path = manager.get_path(model_id)
handle = backend.load_model(path)

# Access the underlying interpreter
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[import-not-found]

if not isinstance(handle, Interpreter):
    print("Not a TFLite model")
    sys.exit(1)

# Check what methods are available
print("\n=== Available diagnostic methods ===")
diagnostic_methods = [
    m
    for m in dir(handle)
    if "tensor" in m.lower() or "subgraph" in m.lower() or "node" in m.lower()
]
for method in sorted(diagnostic_methods):
    if not method.startswith("_"):
        print(f"  {method}")

# Try to get execution plan (this might be a private method)
print("\n=== Checking execution plan ===")
try:
    if hasattr(handle, "_get_execution_plan"):
        exec_plan = handle._get_execution_plan()  # noqa: SLF001
        delegate_nodes = sum(1 for node_id in exec_plan if node_id < 0)
        cpu_nodes = len(exec_plan) - delegate_nodes
        print(f"Total nodes: {len(exec_plan)}")
        print(f"Delegate nodes: {delegate_nodes} ({100 * delegate_nodes / len(exec_plan):.1f}%)")
        print(f"CPU nodes: {cpu_nodes} ({100 * cpu_nodes / len(exec_plan):.1f}%)")

        if cpu_nodes > 0:
            print(f"\n⚠️  HETEROGENEOUS EXECUTION: {cpu_nodes} nodes falling back to CPU")
        else:
            print(f"\n✓ FULL ACCELERATION: All {delegate_nodes} nodes on delegate")
    else:
        print("_get_execution_plan() not available")

        # Try alternative: check tensor allocations
        num_tensors = handle.get_tensor_details()
        print(f"Total tensors: {len(num_tensors) if num_tensors else 'unknown'}")

except Exception as e:  # noqa: BLE001
    print(f"Error checking execution plan: {e}")

print()
