#!/usr/bin/env python3
"""
Power profiling utility for Qualcomm SoCs (QCS6490)
Measures CPU, GPU, NPU power consumption during inference
FIXED VERSION for RubiX Pi 3 with correct Qualcomm battery manager paths
"""

import time
import subprocess
import threading
import statistics
from pathlib import Path

class PowerProfiler:
    def __init__(self):
        self.power_samples = []
        self.sampling = False
        self.sample_thread = None
        
        # Common Qualcomm power monitoring paths
        # These vary by platform, we'll detect which ones exist
        self.power_paths = self._detect_power_monitors()
        
    def _detect_power_monitors(self):
        """Detect available power monitoring interfaces"""
        paths = {}
        
        # IMPORTANT: Use EXACT paths found on RubiX Pi 3
        potential_paths = [
            # Qualcomm Battery Manager (QCS6490 / RubiX Pi specific) - VERIFIED WORKING
            ("/sys/devices/platform/pmic-glink/pmic_glink.power-supply.0/power_supply/qcom-battmgr-bat/power_now", "battery_power_uw"),
            ("/sys/devices/platform/pmic-glink/pmic_glink.power-supply.0/power_supply/qcom-battmgr-bat/current_now", "battery_current_ua"),
            ("/sys/devices/platform/pmic-glink/pmic_glink.power-supply.0/power_supply/qcom-battmgr-bat/voltage_now", "battery_voltage_uv"),
            
            # USB power (when connected)
            ("/sys/devices/platform/pmic-glink/pmic_glink.power-supply.0/power_supply/qcom-battmgr-usb/current_now", "usb_current_ua"),
            
            # GPU busy percentage
            ("/sys/class/kgsl/kgsl-3d0/gpubusy", "gpu_busy_percent"),
            
            # GPU frequency
            ("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq", "gpu_freq_hz"),
        ]
        
        for path, name in potential_paths:
            if Path(path).exists():
                paths[name] = path
                print(f"✓ Found power monitor: {name} at {path}")
            else:
                print(f"✗ Not found: {name} at {path}")
        
        if not paths:
            print("\n⚠️  WARNING: No power monitoring interfaces detected!")
            print("   Falling back to CPU time measurements only.\n")
        
        return paths
    
    def _read_power_sample(self):
        """Read current power consumption from all available monitors"""
        sample = {"timestamp": time.time()}
        
        for name, path in self.power_paths.items():
            try:
                with open(path, 'r') as f:
                    value = int(f.read().strip())
                    sample[name] = value
            except (IOError, ValueError, PermissionError) as e:
                # Some monitors may not be readable at all times
                pass
        
        # Calculate battery power in mW if we have current and voltage
        if "battery_current_ua" in sample and "battery_voltage_uv" in sample:
            # Power (W) = Current (A) * Voltage (V)
            # Convert: uA * uV = pW, then to mW
            current_a = abs(sample["battery_current_ua"]) / 1_000_000  # abs() because discharge is negative
            voltage_v = sample["battery_voltage_uv"] / 1_000_000
            sample["battery_power_calculated_mw"] = current_a * voltage_v * 1000
        
        # Convert battery_power_uw to mW if available (direct measurement)
        if "battery_power_uw" in sample:
            sample["battery_power_mw"] = sample["battery_power_uw"] / 1000
        
        # Prefer calculated over direct measurement for consistency
        # (Some systems report power_now incorrectly)
        if "battery_power_calculated_mw" in sample and "battery_power_mw" not in sample:
            sample["battery_power_mw"] = sample["battery_power_calculated_mw"]
        
        return sample
    
    def _sampling_loop(self, interval_ms=50):
        """Continuously sample power at specified interval"""
        interval_sec = interval_ms / 1000.0
        
        while self.sampling:
            sample = self._read_power_sample()
            if sample:
                self.power_samples.append(sample)
            time.sleep(interval_sec)
    
    def start_profiling(self, sample_interval_ms=50):
        """Start collecting power samples in background thread"""
        self.power_samples = []
        self.sampling = True
        self.sample_thread = threading.Thread(
            target=self._sampling_loop, 
            args=(sample_interval_ms,),
            daemon=True
        )
        self.sample_thread.start()
        print(f"⚡ Power profiling started (sampling every {sample_interval_ms}ms)")
    
    def stop_profiling(self):
        """Stop collecting power samples"""
        self.sampling = False
        if self.sample_thread:
            self.sample_thread.join(timeout=1.0)
        print(f"⚡ Power profiling stopped ({len(self.power_samples)} samples collected)")
    
    def get_statistics(self):
        """Calculate power consumption statistics"""
        if not self.power_samples:
            return {"error": "No samples collected"}
        
        stats = {}
        
        # Get all metric names from first sample (excluding timestamp)
        metric_names = [k for k in self.power_samples[0].keys() if k != "timestamp"]
        
        for metric in metric_names:
            values = [s[metric] for s in self.power_samples if metric in s]
            
            if not values:
                continue
            
            # Determine unit based on metric name
            unit = "unknown"
            if "uw" in metric.lower() or "_ua" in metric.lower():
                if "_ua" in metric.lower():
                    unit = "μA"
                else:
                    unit = "μW"
            elif "mw" in metric.lower() or "mW" in metric:
                unit = "mW"
            elif "uv" in metric.lower():
                unit = "μV"
            elif "percent" in metric:
                unit = "%"
            elif "freq" in metric.lower() or "hz" in metric.lower():
                unit = "Hz"
            
            stats[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "unit": unit,
                "samples": len(values)
            }
        
        return stats
    
    def print_report(self, label=""):
        """Print a formatted power consumption report"""
        stats = self.get_statistics()
        
        if "error" in stats:
            print(f"\n❌ {stats['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"  Power Profiling Report: {label}")
        print(f"{'='*70}")
        
        for metric, data in stats.items():
            print(f"\n{metric}:")
            print(f"  Mean:   {data['mean']:>12.2f} {data['unit']}")
            print(f"  Median: {data['median']:>12.2f} {data['unit']}")
            print(f"  StdDev: {data['stdev']:>12.2f} {data['unit']}")
            print(f"  Min:    {data['min']:>12.2f} {data['unit']}")
            print(f"  Max:    {data['max']:>12.2f} {data['unit']}")
            print(f"  Samples: {data['samples']}")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def export_csv(self, filename="power_profile.csv"):
        """Export raw power samples to CSV"""
        if not self.power_samples:
            print("No samples to export")
            return
        
        import csv
        
        # Get all unique keys across all samples
        fieldnames = set()
        for sample in self.power_samples:
            fieldnames.update(sample.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.power_samples)
        
        print(f"✓ Exported {len(self.power_samples)} samples to {filename}")


# Alternative: Use Qualcomm's profiling tools if available
def check_qualcomm_tools():
    """Check if Qualcomm profiling tools are available"""
    tools = {
        "snpe-throughput-net-run": "SNPE Benchmarking Tool",
        "snpe-platform-validator": "SNPE Platform Validator",
        "adb": "Android Debug Bridge (for remote profiling)",
    }
    
    available = {}
    for cmd, name in tools.items():
        result = subprocess.run(
            ["which", cmd], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            available[cmd] = result.stdout.strip()
            print(f"✓ Found: {name} at {available[cmd]}")
        else:
            print(f"✗ Not found: {name}")
    
    return available


if __name__ == "__main__":
    print("Power Profiling Utility for Qualcomm QCS6490\n")
    
    # Check for Qualcomm tools
    print("Checking for Qualcomm profiling tools:")
    check_qualcomm_tools()
    print()
    
    # Test power profiler
    profiler = PowerProfiler()
    
    if not profiler.power_paths:
        print("\n⚠️  No power monitoring available on this system.")
        print("    You may need to:")
        print("    1. Run as root/sudo for sysfs access")
        print("    2. Enable power monitoring in kernel config")
        print("    3. Use external power measurement hardware")
    else:
        print("\nTesting power sampling for 3 seconds...")
        profiler.start_profiling(sample_interval_ms=100)
        time.sleep(3)
        profiler.stop_profiling()
        profiler.print_report("Test Run")
