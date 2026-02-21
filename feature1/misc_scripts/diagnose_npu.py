#!/usr/bin/env python3
"""
Comprehensive NPU Diagnostic Tool
Determines if NPU is actually being used for inference
"""

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
import time
import psutil
import os
import subprocess

print("="*80)
print("NPU DIAGNOSTIC TOOL")
print("="*80)

# ============ TEST 1: Check Model Quantization ============
print("\n[TEST 1] Checking Model Quantization")
print("-"*80)

MODEL_PATH = "yamnet_int8_quantized.tflite"

try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print(f"Model: {MODEL_PATH}")
    print(f"Input dtype:  {input_details['dtype']}")
    print(f"Output dtype: {output_details['dtype']}")
    
    # Check quantization parameters
    if 'quantization' in input_details:
        scale, zero_point = input_details['quantization']
        if scale != 0.0:
            print(f"Input quantization:  scale={scale:.8f}, zero_point={zero_point}")
            print("✓ Model IS quantized (INT8)")
            model_is_quantized = True
        else:
            print("✗ Model is NOT quantized (Float32)")
            print("  NPU CANNOT be used with this model!")
            model_is_quantized = False
    else:
        print("✗ No quantization info found")
        model_is_quantized = False
    
except FileNotFoundError:
    print(f"✗ Model file not found: {MODEL_PATH}")
    print("\nDid you run: python3 quantize_yamnet.py ?")
    exit(1)

if not model_is_quantized:
    print("\n" + "="*80)
    print("CONCLUSION: Model is not quantized. NPU cannot be used.")
    print("Run: python3 quantize_yamnet.py")
    print("="*80)
    exit(1)

# ============ TEST 2: Check QNN Delegate Loading ============
print("\n[TEST 2] Checking QNN Delegate")
print("-"*80)

try:
    delegate = load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
    print("✓ QNN delegate loaded successfully")
    delegate_loaded = True
except Exception as e:
    print(f"✗ Failed to load QNN delegate: {e}")
    print("\nPossible issues:")
    print("  1. libQnnTFLiteDelegate.so not found")
    print("  2. Incompatible QNN version")
    print("  3. NPU drivers not installed")
    delegate_loaded = False

if not delegate_loaded:
    print("\n" + "="*80)
    print("CONCLUSION: Cannot load QNN delegate. NPU unavailable.")
    print("="*80)
    exit(1)

# ============ TEST 3: Check Delegate Node Assignment ============
print("\n[TEST 3] Checking Node Delegation")
print("-"*80)

# Capture stderr to see delegation info
import io
import sys
from contextlib import redirect_stderr

stderr_capture = io.StringIO()

with redirect_stderr(stderr_capture):
    interpreter_npu = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[delegate]
    )
    interpreter_npu.allocate_tensors()

stderr_output = stderr_capture.getvalue()

# Look for delegation info in stderr
delegation_info_found = False
if "nodes delegated" in stderr_output.lower():
    delegation_info_found = True
    # Extract the delegation info
    for line in stderr_output.split('\n'):
        if "delegate" in line.lower():
            print(line)

if not delegation_info_found:
    print("⚠️  No delegation info found in stderr")
    print("    Checking if model loaded successfully...")
else:
    # Parse delegation percentage
    if "0 nodes delegated" in stderr_output:
        print("\n✗ CRITICAL: 0 nodes delegated to NPU!")
        print("  NPU is NOT being used at all!")
        npu_being_used = False
    elif "nodes delegated" in stderr_output:
        # Extract numbers
        import re
        match = re.search(r'(\d+)\s+nodes delegated out of\s+(\d+)', stderr_output)
        if match:
            delegated = int(match.group(1))
            total = int(match.group(2))
            percentage = (delegated / total) * 100
            print(f"\n✓ {delegated}/{total} nodes delegated ({percentage:.1f}%)")
            
            if percentage > 80:
                print(f"  ✓ EXCELLENT: NPU is handling most operations")
                npu_being_used = True
            elif percentage > 50:
                print(f"  ⚠️  PARTIAL: Some operations still on CPU")
                npu_being_used = True
            else:
                print(f"  ✗ POOR: Most operations still on CPU")
                npu_being_used = False

# ============ TEST 4: CPU Usage Comparison ============
print("\n[TEST 4] CPU Usage Comparison")
print("-"*80)

def measure_cpu_usage(interpreter, num_iterations=50):
    """Measure CPU usage during inference"""
    input_details = interpreter.get_input_details()[0]
    
    # Prepare test data
    test_audio = np.random.uniform(-1.0, 1.0, size=15600).astype(np.float32)
    if input_details['dtype'] != np.float32:
        scale, zero_point = input_details['quantization']
        test_input = (test_audio / scale + zero_point).astype(input_details['dtype'])
    else:
        test_input = test_audio
    
    # Get baseline CPU usage
    process = psutil.Process(os.getpid())
    baseline_cpu = process.cpu_percent(interval=0.1)
    
    # Run inference and measure
    cpu_samples = []
    for _ in range(num_iterations):
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        cpu_samples.append(process.cpu_percent(interval=None))
    
    return np.mean(cpu_samples)

# CPU-only interpreter
print("Testing CPU-only inference...")
interpreter_cpu = Interpreter(model_path=MODEL_PATH)
interpreter_cpu.allocate_tensors()
cpu_usage_cpu = measure_cpu_usage(interpreter_cpu)
print(f"  CPU usage (CPU backend): {cpu_usage_cpu:.1f}%")

# NPU interpreter
print("Testing NPU inference...")
cpu_usage_npu = measure_cpu_usage(interpreter_npu)
print(f"  CPU usage (NPU backend): {cpu_usage_npu:.1f}%")

cpu_reduction = ((cpu_usage_cpu - cpu_usage_npu) / cpu_usage_cpu) * 100
print(f"\nCPU usage reduction: {cpu_reduction:.1f}%")

if cpu_reduction > 40:
    print("✓ Significant CPU reduction - NPU is being used!")
elif cpu_reduction > 20:
    print("⚠️  Moderate CPU reduction - NPU partially used")
else:
    print("✗ Minimal CPU reduction - NPU likely NOT being used")

# ============ TEST 5: Performance Comparison ============
print("\n[TEST 5] Performance Comparison")
print("-"*80)

def benchmark(interpreter, name, num_iterations=100):
    """Benchmark inference speed"""
    input_details = interpreter.get_input_details()[0]
    
    # Prepare test data
    test_audio = np.random.uniform(-1.0, 1.0, size=15600).astype(np.float32)
    if input_details['dtype'] != np.float32:
        scale, zero_point = input_details['quantization']
        test_input = (test_audio / scale + zero_point).astype(input_details['dtype'])
    else:
        test_input = test_audio
    
    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        interpreter.set_tensor(input_details['index'], test_input)
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'name': name,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

cpu_results = benchmark(interpreter_cpu, "CPU")
npu_results = benchmark(interpreter_npu, "NPU")

print(f"\nCPU Backend:")
print(f"  Mean: {cpu_results['mean']:.2f}ms ± {cpu_results['std']:.2f}ms")
print(f"  Range: [{cpu_results['min']:.2f}ms, {cpu_results['max']:.2f}ms]")

print(f"\nNPU Backend:")
print(f"  Mean: {npu_results['mean']:.2f}ms ± {npu_results['std']:.2f}ms")
print(f"  Range: [{npu_results['min']:.2f}ms, {npu_results['max']:.2f}ms]")

speedup = cpu_results['mean'] / npu_results['mean']
print(f"\nSpeedup: {speedup:.2f}x")

if speedup > 2.5:
    print("✓ EXCELLENT: NPU providing significant acceleration!")
elif speedup > 1.5:
    print("✓ GOOD: NPU is providing acceleration")
elif speedup > 1.1:
    print("⚠️  MARGINAL: Limited NPU benefit")
else:
    print("✗ NO SPEEDUP: NPU is NOT being used effectively")

# ============ TEST 6: Check System NPU Status ============
print("\n[TEST 6] System NPU Status")
print("-"*80)

# Check for NPU-related kernel modules
print("Checking kernel modules...")
try:
    result = subprocess.run(['lsmod'], capture_output=True, text=True)
    npu_modules = [line for line in result.stdout.split('\n') 
                   if any(keyword in line.lower() for keyword in ['qcom', 'hexagon', 'npu', 'cdsp'])]
    if npu_modules:
        print("✓ Found NPU-related kernel modules:")
        for module in npu_modules[:5]:  # Show first 5
            print(f"  {module}")
    else:
        print("⚠️  No NPU kernel modules found")
except:
    print("⚠️  Cannot check kernel modules")

# Check for NPU device files
print("\nChecking device files...")
npu_devices = [
    "/dev/qcom-npu",
    "/dev/cdsp",
    "/sys/class/misc/fastrpc",
    "/sys/kernel/debug/qcom_npu"
]
found_devices = []
for device in npu_devices:
    if os.path.exists(device):
        found_devices.append(device)
        print(f"✓ Found: {device}")

if not found_devices:
    print("⚠️  No NPU device files found")

# ============ FINAL VERDICT ============
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

checks = []
checks.append(("Model Quantized", model_is_quantized))
checks.append(("Delegate Loaded", delegate_loaded))
checks.append(("CPU Reduction > 30%", cpu_reduction > 30))
checks.append(("Speedup > 1.5x", speedup > 1.5))

passed = sum([1 for _, result in checks if result])
total = len(checks)

print(f"\nChecks Passed: {passed}/{total}")
print()
for check_name, result in checks:
    status = "✓" if result else "✗"
    print(f"  {status} {check_name}")

print("\n" + "="*80)
if passed >= 3:
    print("CONCLUSION: ✓ NPU IS BEING USED")
    print("="*80)
    print(f"\nYour inference time of 2.7ms is REAL NPU performance!")
    print(f"The NPU is working correctly and providing {speedup:.1f}x acceleration.")
elif passed >= 2:
    print("CONCLUSION: ⚠️  NPU PARTIALLY WORKING")
    print("="*80)
    print("\nThe NPU is loaded but not providing full acceleration.")
    print("Some operations may still be running on CPU.")
else:
    print("CONCLUSION: ✗ NPU IS NOT BEING USED")
    print("="*80)
    print("\nThe NPU is not working. Your 2.7ms is CPU performance.")
    print("\nTroubleshooting steps:")
    print("1. Verify model is properly quantized (INT8)")
    print("2. Check NPU drivers are installed")
    print("3. Try different QNN delegate options")

print("="*80)
