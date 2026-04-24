"""Test QNN device discovery."""

from __future__ import annotations

import onnxruntime as ort

from moment_to_action.hardware._platforms.qcs6490._onnx import (
    _QNN_EP_NAME,
    _ensure_qnn_ep_registered,
)

# Register the plugin
_ensure_qnn_ep_registered()

# Check what devices are discovered
devices = [d for d in ort.get_ep_devices() if d.ep_name == _QNN_EP_NAME]
print(f"Found {len(devices)} QNN device(s)")
for d in devices:
    print(f"  Device: {d.device}, EP: {d.ep_name}")
    if hasattr(d, "ep_metadata"):
        print(f"    Metadata: {d.ep_metadata}")
