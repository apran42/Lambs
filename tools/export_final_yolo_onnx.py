"""Export the selected YOLOv8n checkpoint as a static, raw-output ONNX graph.

Uses PyTorch's legacy exporter because the installed Torch ONNX front-end
requires onnxscript, which is not present in the local environment.
"""

import argparse
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = YOLO(str(args.model)).model.float().eval().fuse()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, Detect):
            module.dynamic = False
            module.export = True
            module.format = "onnx"
        elif isinstance(module, C2f):
            module.forward = module.forward_split
    example = torch.zeros(1, 3, args.image_size, args.image_size)
    with torch.no_grad():
        expected = model(example).cpu().numpy()
        torch.onnx.utils.export(
            model,
            example,
            str(args.output),
            opset_version=12,
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
        )
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    print({"output": str(args.output), "expected_shape": expected.shape,
           "input_shape": [1, 3, args.image_size, args.image_size]})


if __name__ == "__main__":
    main()
