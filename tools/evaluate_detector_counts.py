"""Evaluate per-frame person-count accuracy for an Ultralytics detector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Ultralytics 8.3.50 still calls np.trapz, removed by newer NumPy releases.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--sweep-confidence",
        action="store_true",
        help="Evaluate confidence thresholds from 0.05 through 0.90.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def ground_truth_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    args = parse_args()
    images = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    model = YOLO(args.model)
    prediction_confidence = 0.001 if args.sweep_confidence else args.confidence
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.image_size,
        conf=prediction_confidence,
        device=args.device,
        stream=True,
        verbose=False,
    )

    actual_counts: list[int] = []
    confidence_scores: list[list[float]] = []
    for image_path, result in zip(images, results, strict=True):
        actual_counts.append(ground_truth_count(args.labels / f"{image_path.stem}.txt"))
        confidence_scores.append([float(score) for score in result.boxes.conf.cpu().tolist()])

    def metrics_at(threshold: float) -> dict[str, float | int]:
        errors = [
            sum(score >= threshold for score in scores) - actual
            for actual, scores in zip(actual_counts, confidence_scores, strict=True)
        ]
        absolute = [abs(error) for error in errors]
        return {
            "confidence": threshold,
            "exact_count_frames": sum(error == 0 for error in errors),
            "exact_count_rate": sum(error == 0 for error in errors) / len(errors),
            "mae": sum(absolute) / len(errors),
            "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
            "bias": sum(errors) / len(errors),
            "under_count_frames": sum(error < 0 for error in errors),
            "over_count_frames": sum(error > 0 for error in errors),
            "max_absolute_error": max(absolute),
        }

    payload = {
        "model": args.model,
        "frames": len(actual_counts),
    }
    if args.sweep_confidence:
        candidates = [round(step / 100, 2) for step in range(5, 91, 5)]
        evaluations = [metrics_at(threshold) for threshold in candidates]
        payload["best"] = min(
            evaluations,
            key=lambda item: (item["mae"], -item["exact_count_rate"], abs(item["bias"])),
        )
        payload["evaluations"] = evaluations
    else:
        payload.update(metrics_at(args.confidence))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
