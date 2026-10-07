"""Measure demo-video count errors without using held-out test videos for tuning.

The full-video optimum is diagnostic only: selecting it on the same frames used
for reporting gives an in-sample result, not an independent accuracy estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", action="append", required=True)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--report-threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def score(rows: list[dict], threshold: float, offset: int = 0) -> dict:
    truth = sum(row["truth"] for row in rows)
    predictions = [max(0, sum(value >= threshold for value in row["scores"]) + offset) for row in rows]
    absolute_error = sum(abs(prediction - row["truth"]) for row, prediction in zip(rows, predictions))
    return {
        "frames": len(rows),
        "ground_truth_people": truth,
        "predicted_people": sum(predictions),
        "absolute_error_sum": absolute_error,
        "wape": absolute_error / truth if truth else None,
        "exact_frame_rate": sum(prediction == row["truth"] for row, prediction in zip(rows, predictions)) / len(rows),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("--calibration-fraction must be between 0 and 1")
    model = YOLO(args.model)
    thresholds = [round(index / 100, 2) for index in range(5, 91)]
    output = {
        "model": args.model,
        "image_size": args.image_size,
        "nms_iou": args.iou,
        "calibration_fraction": args.calibration_fraction,
        "report_threshold": args.report_threshold,
        "videos": {},
    }
    for video in args.video:
        images = sorted((args.data / "images" / "train").glob(f"video{video}__*.jpg"))
        if not images:
            raise ValueError(f"No training frames found for video {video}")
        rows = []
        for start in range(0, len(images), 16):
            batch = images[start:start + 16]
            results = model.predict(
                source=[str(image) for image in batch],
                imgsz=args.image_size,
                iou=args.iou,
                conf=0.001,
                device="cpu",
                stream=True,
                verbose=False,
            )
            for image, result in zip(batch, results):
                label = args.data / "labels" / "train" / f"{image.stem}.txt"
                truth = sum(bool(line.strip()) for line in label.read_text(encoding="utf-8").splitlines())
                scores = [
                    float(confidence)
                    for confidence, klass in zip(result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist())
                    if int(klass) == 0
                ]
                rows.append({"image": image.name, "truth": truth, "scores": scores})
        split = max(1, min(len(rows) - 1, round(len(rows) * args.calibration_fraction)))
        calibration, later = rows[:split], rows[split:]
        all_best = min(thresholds, key=lambda value: (score(rows, value)["wape"], value))
        calibration_best = min(thresholds, key=lambda value: (score(calibration, value)["wape"], value))
        candidates = [(threshold, offset) for threshold in thresholds for offset in range(-3, 4)]
        joint_best = min(candidates, key=lambda pair: (score(rows, *pair)["wape"], abs(pair[1]), pair[0]))
        early_joint_best = min(
            candidates,
            key=lambda pair: (score(calibration, *pair)["wape"], abs(pair[1]), pair[0]),
        )
        worst = sorted(
            (
                {
                    "image": row["image"],
                    "truth": row["truth"],
                    "predicted": sum(value >= 0.25 for value in row["scores"]),
                    "absolute_error": abs(sum(value >= 0.25 for value in row["scores"]) - row["truth"]),
                }
                for row in rows
            ),
            key=lambda row: row["absolute_error"],
            reverse=True,
        )[:10]
        output["videos"][video] = {
            "fixed_0_25": score(rows, 0.25),
            "reported_threshold_result": score(rows, args.report_threshold),
            "best_threshold_on_all_frames_diagnostic_only": all_best,
            "best_on_all_frames_diagnostic_only": score(rows, all_best),
            "best_threshold_and_offset_on_all_frames_diagnostic_only": {
                "threshold": joint_best[0], "offset": joint_best[1],
                **score(rows, *joint_best),
            },
            "calibration_threshold_from_early_frames": calibration_best,
            "early_calibration": score(calibration, calibration_best),
            "later_frames_at_calibration_threshold": score(later, calibration_best),
            "threshold_and_offset_from_early_frames": {
                "threshold": early_joint_best[0], "offset": early_joint_best[1],
            },
            "later_frames_at_early_threshold_and_offset": score(later, *early_joint_best),
            "later_frames_at_fixed_0_25": score(later, 0.25),
            "worst_frames_at_fixed_0_25": worst,
        }
        print(f"video{video}: {output['videos'][video]['fixed_0_25']['wape']:.2%} -> "
              f"best in-sample {output['videos'][video]['best_on_all_frames_diagnostic_only']['wape']:.2%}", flush=True)
    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
