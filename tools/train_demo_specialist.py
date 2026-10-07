"""Optional scene-specific fine-tuning; never replace the general model in-place.

The two demo videos are also the training videos. Accuracy on those same frames
is an in-sample demonstration result, not an independent generalization claim.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from train_yolo_detector import read_results_without_pandas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--name", default="lambs_demo35_11_specialist_v1")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("data/training/.ultralytics").resolve()))
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer

    BaseTrainer.read_results_csv = read_results_without_pandas
    metrics = YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device="cpu",
        workers=0,
        optimizer="AdamW",
        lr0=args.learning_rate,
        lrf=0.1,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        patience=args.epochs,
        save_period=1,
        seed=42,
        deterministic=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,
        close_mosaic=0,
        project="runs/training",
        name=args.name,
        exist_ok=False,
        plots=False,
    )
    print({"run": str(metrics.save_dir), "best": str(metrics.save_dir / "weights/best.pt")})


if __name__ == "__main__":
    main()
