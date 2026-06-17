#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["torch", "onnx", "onnxruntime", "opencv-python-headless", "numpy"]
#
# [tool.uv.sources]
# ///
# ruff: noqa: PLC0415
"""Export a float single-graph Detectron2 (Faster R-CNN R50-C4) ONNX at 800x800.

Mirrors detectron2's tools/deploy/export_model.py tracing+onnx path, but pulls
the config + weights from ``detectron2.model_zoo`` and bakes a fixed 800x800
input (so the CPU/GPU benchmark can reuse the letterbox prepare).  The traced
GeneralizedRCNN runs RPN + ROI head + NMS internally and outputs final
detections in the 800x800 resized space.

Run inside the project venv (has torch + detectron2 0.6):
    uv run python scripts/export_detectron2_onnx.py <sample.jpg> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_CFG = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
_SIZE = 800


def _letterbox_bgr(frame: np.ndarray) -> np.ndarray:
    """Aspect-preserve resize + center pad a BGR frame into an 800x800 canvas."""
    h, w = frame.shape[:2]
    scale = min(_SIZE / h, _SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((_SIZE, _SIZE, 3), dtype=frame.dtype)
    top, left = (_SIZE - nh) // 2, (_SIZE - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def main() -> None:
    """Export + inspect the single-graph Detectron2 ONNX from CLI args."""
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.export import STABLE_ONNX_OPSET_VERSION, TracingAdapter
    from detectron2.modeling import GeneralizedRCNN, build_model

    sample_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(_CFG))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(_CFG)
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    print(
        f"INPUT.FORMAT={cfg.INPUT.FORMAT} PIXEL_MEAN={cfg.MODEL.PIXEL_MEAN} "
        f"PIXEL_STD={cfg.MODEL.PIXEL_STD} MIN_SIZE_TEST={cfg.INPUT.MIN_SIZE_TEST}"
    )

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    assert isinstance(model, GeneralizedRCNN)  # noqa: S101

    bgr = cv2.imread(str(sample_path))
    canvas = _letterbox_bgr(bgr)  # 800x800 BGR uint8
    # detectron2 input contract: raw BGR (cfg.INPUT.FORMAT), 0-255 float32, CHW;
    # the model normalizes with PIXEL_MEAN/STD internally.
    image = torch.as_tensor(canvas.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image}]

    def inference(m: object, batched: object) -> list[dict]:
        """Run the detector and wrap the instances for TracingAdapter."""
        inst = m.inference(batched, do_postprocess=False)[0]  # type: ignore[attr-defined]
        return [{"instances": inst}]

    adapter = TracingAdapter(model, inputs, inference)
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(adapter, (image,), str(onnx_path), opset_version=STABLE_ONNX_OPSET_VERSION)

    print("inputs_schema:", adapter.inputs_schema)
    print("outputs_schema:", adapter.outputs_schema)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")

    # Inspect with onnxruntime so we know the output names/order for the model code.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("ONNX inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
    feed = {sess.get_inputs()[0].name: image.numpy()}
    outs = sess.run(None, feed)
    for o, arr in zip(sess.get_outputs(), outs, strict=False):
        a = np.asarray(arr)
        print(f"  out {o.name}: shape={a.shape} dtype={a.dtype} sample={a.ravel()[:6]}")


if __name__ == "__main__":
    main()
