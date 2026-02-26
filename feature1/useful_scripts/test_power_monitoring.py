#!/usr/bin/env python3
"""
Quick test to verify power monitoring is working
"""

import sys
import os

# Add current directory to path to import power_profiler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from power_profiler import PowerProfiler
import time

print("="*70)
print("  Power Monitoring Test - Qualcomm QCS6490")
print("="*70)
print()

# Create profiler
profiler = PowerProfiler()

if not profiler.power_paths:
    print("❌ No power monitoring paths detected!")
    print("   This shouldn't happen - check if you need sudo access")
    sys.exit(1)

print("✓ Power monitoring initialized")
print()

# Take a few quick samples
print("Taking 5 power samples (1 second apart)...")
print()

for i in range(5):
    sample = profiler._read_power_sample()
    
    print(f"Sample {i+1}:")
    if "battery_power_uw" in sample:
        print(f"  Power (raw):      {sample['battery_power_uw']:>10.0f} μW")
    if "battery_power_mw" in sample:
        print(f"  Power (mW):       {sample['battery_power_mw']:>10.2f} mW")
    if "battery_current_ua" in sample:
        current_ma = abs(sample['battery_current_ua']) / 1000
        print(f"  Current:          {current_ma:>10.2f} mA")
    if "battery_voltage_uv" in sample:
        voltage_v = sample['battery_voltage_uv'] / 1_000_000
        print(f"  Voltage:          {voltage_v:>10.3f} V")
    print()
    
    time.sleep(1)

print("="*70)
print("✓ Power monitoring is working correctly!")
print()
print("Expected values:")
print("  - Idle power: ~500-2000 mW (depends on what's running)")
print("  - Under load: ~2000-5000+ mW")
print("  - Voltage: ~3.7-4.2V (typical Li-ion)")
print()
print("You can now run the full benchmarks:")
print("  sudo /path/to/venv/bin/python3 yamnet_inference_profiled.py")
print("  sudo /path/to/venv/bin/python3 yamnet_inference_profiled.py --use-npu")
print("="*70)
