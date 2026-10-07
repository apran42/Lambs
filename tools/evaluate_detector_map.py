"""Evaluate one or more Ultralytics detectors on a fixed YOLO split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/evaluation")
    parser.add_argument("--output", type=Path, help="Write the numeric comparison to JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_root = Path(args.project).resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(evaluation_root.parent / ".ultralytics"))

    # Ultralytics 8.3.50 still calls np.trapz, removed by newer NumPy releases.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]

    from ultralytics import YOLO

    payload = []
    for index, model_path in enumerate(args.model, start=1):
        metrics = YOLO(model_path).val(
            data=args.data,
            split=args.split,
            imgsz=args.image_size,
            device=args.device,
            workers=0,
            plots=False,
            project=args.project,
            name=f"model_{index}_{args.split}",
            exist_ok=True,
        )
        payload.append(
            {
                "model": model_path,
                "split": args.split,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        )

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
