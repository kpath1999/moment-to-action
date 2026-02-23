"""
inspect_model.py — dumps all input/output tensor details for a .tflite model
Usage: python3 inspect_model.py --model movinet_model.tflite
"""
import argparse
import numpy as np

try:
    import ai_edge_litert.interpreter as litert
    Interpreter = litert.Interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
args = parser.parse_args()

interp = Interpreter(model_path=args.model)
interp.allocate_tensors()

inputs  = interp.get_input_details()
outputs = interp.get_output_details()

print(f"\n{'='*70}")
print(f"  MODEL: {args.model}")
print(f"  {len(inputs)} input(s)   {len(outputs)} output(s)")
print(f"{'='*70}\n")

print("── INPUTS ──────────────────────────────────────────────────────────")
for t in inputs:
    print(f"  index : {t['index']}")
    print(f"  name  : {t['name']}")
    print(f"  shape : {t['shape']}")
    print(f"  dtype : {t['dtype']}")
    print(f"  quant : scale={t['quantization'][0]:.6f}  zero_point={t['quantization'][1]}")
    print()

print("── OUTPUTS ─────────────────────────────────────────────────────────")
for t in outputs:
    print(f"  index : {t['index']}")
    print(f"  name  : {t['name']}")
    print(f"  shape : {t['shape']}")
    print(f"  dtype : {t['dtype']}")
    print(f"  quant : scale={t['quantization'][0]:.6f}  zero_point={t['quantization'][1]}")
    print()

# Identify the image input (float32, 4D or 5D) and the logit output
print("── QUICK SUMMARY ───────────────────────────────────────────────────")
for t in inputs:
    if t['dtype'] == np.float32 and len(t['shape']) >= 4:
        print(f"  🎥 Video input  → index {t['index']}  shape {t['shape']}  ({t['name']})")
    elif t['dtype'] == np.int32:
        print(f"  🔢 State int32  → index {t['index']}  shape {t['shape']}  ({t['name']})")
    else:
        print(f"  📦 Other input  → index {t['index']}  shape {t['shape']}  dtype={t['dtype']}  ({t['name']})")

for t in outputs:
    if len(t['shape']) <= 2:
        print(f"  🏷️  Logit output → index {t['index']}  shape {t['shape']}  ({t['name']})")
    else:
        print(f"  📦 State output → index {t['index']}  shape {t['shape']}  ({t['name']})")
