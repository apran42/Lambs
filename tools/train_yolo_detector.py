from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lambs YOLO person detector")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="lambs_yolov8n_pilot")
    parser.add_argument("--project", default="runs/training")
    return parser.parse_args()


def read_results_without_pandas(trainer) -> dict[str, list[float]]:
    with Path(trainer.csv).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        values: dict[str, list[float]] = {field.strip(): [] for field in (reader.fieldnames or [])}
        for row in reader:
            for field, value in row.items():
                values[field.strip()].append(float(value))
    return values


def main() -> None:
    args = parse_args()
    training_root = Path(args.project).resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(training_root.parent / ".ultralytics"))

    # Ultralytics 8.3.50 expects APIs removed by newer NumPy and imports a local
    # pandas build that is binary-incompatible with that NumPy. These compatibility
    # shims keep the project environment unchanged.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]

    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer

    BaseTrainer.read_results_csv = read_results_without_pandas
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        workers=0,
        patience=5,
        seed=42,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=False,
        plots=False,
        close_mosaic=3,
    )
    best = results.save_dir / "weights" / "best.pt"
    last = results.save_dir / "weights" / "last.pt"
    print({"best": str(best), "last": str(last)})


if __name__ == "__main__":
    main()
