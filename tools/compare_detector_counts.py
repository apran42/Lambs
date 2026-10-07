"""Compare person-count errors at a fixed and validation-tuned threshold.

Video groups, not individual frames, must be assigned to train/val/test before use.
The sweep selects a threshold on validation only; test and train are then frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True, help="name=weights.pt")
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--fixed-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-train", action="store_true", help="Evaluate only validation and test videos")
    parser.add_argument("--train-only", action="store_true", help="Evaluate the training videos only at --fixed-threshold")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(records: list[tuple[int, list[float]]], threshold: float) -> dict:
    actual = [record[0] for record in records]
    predicted = [sum(score >= threshold for score in record[1]) for record in records]
    errors = [estimate - truth for truth, estimate in zip(actual, predicted, strict=True)]
    absolute = [abs(error) for error in errors]
    return {
        "frames": len(records), "ground_truth_people": sum(actual),
        "predicted_people": sum(predicted), "absolute_error_sum": sum(absolute),
        "wape": sum(absolute) / sum(actual) if sum(actual) else None,
        "mae": sum(absolute) / len(records) if records else None,
        "rmse": math.sqrt(sum(error * error for error in errors) / len(records)) if records else None,
        "bias": sum(errors) / len(records) if records else None,
        "exact_frame_rate": sum(error == 0 for error in errors) / len(records) if records else None,
        "frames_within_10pct": sum(abs(error) < 0.1 * truth if truth else error == 0 for truth, error in zip(actual, errors, strict=True)),
    }


def collect(model_path: str, data: Path, image_size: int, device: str, skip_train: bool, train_only: bool, iou: float) -> dict:
    model = YOLO(model_path)
    records: dict[str, list[tuple[str, int, list[float]]]] = {}
    for split in ("train", "val", "test"):
        if split == "train" and skip_train:
            continue
        if split != "train" and train_only:
            continue
        images = sorted((data / "images" / split).glob("*.jpg"))
        if not images:
            raise ValueError(f"No {split} images found")
        records[split] = []
        for start in range(0, len(images), 16):
            batch = images[start : start + 16]
            results = model.predict(
                source=[str(image) for image in batch], imgsz=image_size,
                conf=0.001, iou=iou, device=device, stream=True, verbose=False,
            )
            for image, result in zip(batch, results, strict=True):
                label = data / "labels" / split / f"{image.stem}.txt"
                ground_truth = sum(bool(line.strip()) for line in label.read_text(encoding="utf-8").splitlines())
                scores = [
                    float(score)
                    for score, klass in zip(
                        result.boxes.conf.cpu().tolist(),
                        result.boxes.cls.cpu().tolist(),
                        strict=True,
                    )
                    if int(klass) == 0
                ]
                records[split].append((image.stem.split("__", 1)[0], ground_truth, scores))
    return records


def evaluate(records: dict, fixed_threshold: float = 0.25) -> dict:
    thresholds = [round(value / 100, 2) for value in range(5, 91, 5)]
    validation = [(truth, scores) for _, truth, scores in records.get("val", [])]
    selected = min(thresholds, key=lambda threshold: (summarize(validation, threshold)["wape"], threshold)) if validation else None
    output = {"validation_selected_threshold": selected, "fixed_threshold": fixed_threshold}
    fixed_label = f"fixed_{fixed_threshold:.2f}".replace(".", "_")
    configurations = [(fixed_label, fixed_threshold)]
    if selected is not None:
        configurations.append(("validation_tuned", selected))
    for label, threshold in configurations:
        groups = {}
        for split, entries in records.items():
            values = [(truth, scores) for _, truth, scores in entries]
            per_video: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
            for video, truth, scores in entries:
                per_video[video].append((truth, scores))
            groups[split] = {
                "overall": summarize(values, threshold),
                "by_video": {video: summarize(video_values, threshold) for video, video_values in sorted(per_video.items())},
            }
        output[label] = groups
    return output


def main() -> None:
    args = parse_args()
    if args.skip_train and args.train_only:
        raise ValueError("--skip-train and --train-only cannot be combined")
    models = {}
    for entry in args.model:
        name, path = entry.split("=", 1)
        if name in models:
            raise ValueError(f"Duplicate model name: {name}")
        print(f"Evaluating {name}: {path}", flush=True)
        models[name] = {"path": path, **evaluate(collect(path, args.data, args.image_size, args.device, args.skip_train, args.train_only, args.iou), args.fixed_threshold)}
    payload = {"dataset": str(args.data.resolve()), "image_size": args.image_size, "nms_iou": args.iou, "models": models}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
