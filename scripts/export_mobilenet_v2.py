"""Export MobileNet V2 from torchvision to ONNX.

Usage:
    uv run python scripts/export_mobilenet_v2.py -o ../m2a-models/mobilenet_v2_onnx/model.onnx
"""

import argparse
from pathlib import Path

import torch
import torchvision.models as tv_models


def main() -> None:
    """Export pretrained MobileNet V2 to ONNX opset 12."""
    parser = argparse.ArgumentParser(description="Export MobileNet V2 to ONNX")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output path for the ONNX file (e.g. ../m2a-models/mobilenet_v2_onnx/model.onnx)",
    )
    args = parser.parse_args()

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = tv_models.mobilenet_v2(weights=weights)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Exported: {output_path}")
    print(f"Labels: {len(weights.meta['categories'])} classes")
    print(f"Top-5 labels: {weights.meta['categories'][:5]}")


if __name__ == "__main__":
    main()
