"""Compare one-minute forecasts on the latest reviewed export of each video.

The detector's original video-level train/val/test split is reused. This is an
offline experiment, not a deployable confidence or accuracy guarantee. Only
observations at or before each forecast origin are used as model inputs.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/training"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/training/lambs_yolov8n_21videos_v1/weights/best.pt"),
    )
    parser.add_argument("--stride-seconds", type=int, default=10)
    parser.add_argument("--horizon-seconds", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--export-reviewed-demo",
        type=Path,
        help="Export same-video timeline reference (no detector inference)",
    )
    parser.add_argument(
        "--demo-video-ids",
        help="Comma-separated reviewed video IDs for demo export (default: all)",
    )
    return parser.parse_args()


def reviewed_videos(root: Path, stride_seconds: int) -> dict[int, dict]:
    videos: dict[int, dict] = {}
    for metadata_path in root.rglob("metadata.json"):
        folder = metadata_path.parent
        if "exports" in folder.parts:
            continue
        exports = sorted((folder / "exports").glob("export_*/reviewed_timeseries.csv"))
        if not exports:
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        match = re.search(r"\((\d+)\)", metadata.get("source_video", ""))
        if not match:
            continue
        video_id = int(match.group(1))
        if video_id in videos:
            raise ValueError(f"Duplicate reviewed video: {video_id}")
        export = exports[-1].parent
        image_by_id = {
            path.stem.split("__", 1)[-1]: path
            for path in (export / "images").rglob("*.jpg")
        }
        with exports[-1].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        samples = {}
        for row in rows:
            if row["video_time_seconds"] == "" or row["ground_truth_count"] == "":
                continue
            second = float(row["video_time_seconds"])
            if second != int(second) or int(second) % stride_seconds:
                continue
            image = image_by_id.get(row["sample_id"])
            if image is not None:
                samples[second] = {
                    "image": image,
                    "truth": int(row["ground_truth_count"]),
                }
        videos[video_id] = {
            "samples": samples,
            "export": str(export),
            "duration_seconds": float(metadata["duration_seconds"]),
        }
    return videos


def split_by_video(root: Path) -> dict[int, str]:
    manifest = json.loads(
        (root / "combined_21videos_grouped_20261001" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        int(re.search(r"video(\d+)", item["name"]).group(1)): item["split"]
        for item in manifest["sources"]
    }


def detect_counts(videos: dict[int, dict], args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    keys = []
    for video_id, video in sorted(videos.items()):
        samples = video["samples"]
        times = sorted(samples)
        origins = {
            second
            for second in samples
            if second >= 30 and second + args.horizon_seconds in samples
        }
        history_seconds = int(getattr(args, "history_seconds", 20))
        history_step = int(getattr(args, "stride_seconds", 10))
        needed = set(origins)
        for origin in origins:
            for query in range(int(origin - history_seconds), int(origin) + 1, history_step):
                index = bisect.bisect_right(times, query) - 1
                if index >= 0 and query - times[index] <= 10:
                    needed.add(times[index])
        keys.extend((video_id, second) for second in sorted(needed))
    if not keys:
        raise ValueError("No reviewed frames with a matching image")
    images = [
        str(videos[video_id]["samples"][second]["image"])
        for video_id, second in keys
    ]
    detector = YOLO(str(args.model))
    results = detector.predict(
        source=images,
        imgsz=args.image_size,
        conf=args.confidence,
        iou=args.iou,
        classes=[0],
        device=args.device,
        batch=8,
        stream=True,
        verbose=False,
    )
    processed = 0
    for (video_id, second), result in zip(keys, results):
        videos[video_id]["samples"][second]["detected"] = len(result.boxes)
        processed += 1
        if processed % 100 == 0:
            print(f"Inferred {processed}/{len(keys)} frames", file=sys.stderr, flush=True)
    if processed != len(keys):
        raise RuntimeError(f"Detector returned {processed} results for {len(keys)} images")


def forecast_rows(videos: dict[int, dict], splits: dict[int, str], horizon: int):
    grouped = defaultdict(list)
    for video_id, video in sorted(videos.items()):
        # Newly reviewed videos absent from the original detector-training
        # manifest remain independent test videos by default.
        split = splits.get(video_id, "test")
        samples = video["samples"]
        for second, sample in sorted(samples.items()):
            if second < 30 or second + horizon not in samples:
                continue
            current = sample["detected"]
            previous = samples.get(second - 10, {}).get("detected", current)
            previous2 = samples.get(second - 20, {}).get("detected", previous)
            features = np.array(
                [
                    1.0,
                    current,
                    (current + previous) / 2,
                    (current + previous + previous2) / 3,
                    current - previous,
                ],
                dtype=float,
            )
            grouped[split].append(
                (video_id, second, features, samples[second + horizon]["truth"])
            )
    return grouped


def ridge(rows: list, alpha: float) -> np.ndarray:
    x = np.stack([row[2] for row in rows])
    y = np.array([row[3] for row in rows], dtype=float)
    prior = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    penalty = np.diag([0.01, 1.0, 1.0, 1.0, 1.0]) * alpha
    return np.linalg.solve(x.T @ x + penalty, x.T @ y + penalty @ prior)


def metrics(rows: list, coefficients: np.ndarray) -> dict:
    errors = [
        abs(max(0, round(float(row[2] @ coefficients))) - row[3]) for row in rows
    ]
    people = sum(row[3] for row in rows)
    return {
        "pairs": len(rows),
        "mae": round(sum(errors) / len(errors), 4) if errors else None,
        "wape_percent": round(100 * sum(errors) / people, 4) if people else None,
    }


def prediction_interval(rows: list, coefficients: np.ndarray, radius: int) -> dict:
    covered = 0
    for _, _, features, truth in rows:
        estimate = max(0, round(float(features @ coefficients)))
        covered += max(0, estimate - radius) <= truth <= estimate + radius
    return {
        "radius_people": radius,
        "coverage_percent": round(100 * covered / len(rows), 4) if rows else None,
        "covered_pairs": covered,
        "pairs": len(rows),
    }


def calibrated_radius(rows: list, coefficients: np.ndarray, coverage: float = 0.9) -> int:
    errors = sorted(
        abs(max(0, round(float(features @ coefficients))) - truth)
        for _, _, features, truth in rows
    )
    index = min(len(errors) - 1, math.ceil((len(errors) + 1) * coverage) - 1)
    return int(errors[index])


def main() -> None:
    args = parse_args()
    if args.stride_seconds <= 0 or args.horizon_seconds <= 0:
        raise ValueError("Stride and horizon must be positive")
    videos = reviewed_videos(args.root, args.stride_seconds)
    if args.export_reviewed_demo:
        if args.demo_video_ids:
            selected_ids = {int(value.strip()) for value in args.demo_video_ids.split(",")}
            missing = selected_ids - set(videos)
            if missing:
                raise ValueError("Missing reviewed videos: {}".format(sorted(missing)))
            videos = {video_id: video for video_id, video in videos.items()
                      if video_id in selected_ids}
        payload = {
            "schema_version": 1,
            "horizon_seconds": args.horizon_seconds,
            "scope": "same-reviewed-video-timeline-only",
            "videos": {
                str(video_id): {
                    "duration_seconds": video["duration_seconds"],
                    "samples": [
                        [second, sample["truth"]]
                        for second, sample in sorted(video["samples"].items())
                    ],
                }
                for video_id, video in sorted(videos.items())
            },
        }
        args.export_reviewed_demo.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(json.dumps({"export": str(args.export_reviewed_demo), "videos": len(videos)}))
        return
    splits = split_by_video(args.root)
    detect_counts(videos, args)
    rows = forecast_rows(videos, splits, args.horizon_seconds)
    for split in ("train", "val", "test"):
        if not rows[split]:
            raise ValueError(f"No one-minute pairs in {split}")
    persistence = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    candidates = {}
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        coefficients = ridge(rows["train"], alpha)
        candidates[str(alpha)] = {
            "coefficients": coefficients.tolist(),
            "train": metrics(rows["train"], coefficients),
            "val": metrics(rows["val"], coefficients),
        }
    selected_alpha = min(
        candidates,
        key=lambda alpha: (candidates[alpha]["val"]["mae"], float(alpha)),
    )
    train_fitted = np.array(candidates[selected_alpha]["coefficients"])
    raw_radius = calibrated_radius(rows["val"], persistence)
    fitted_radius = calibrated_radius(rows["val"], train_fitted)
    selected = ridge(rows["train"] + rows["val"], float(selected_alpha))
    test_by_video = {
        str(video_id): {
            "persistence": metrics([row for row in rows["test"] if row[0] == video_id], persistence),
            "calibrated": metrics([row for row in rows["test"] if row[0] == video_id], selected),
        }
        for video_id in sorted({row[0] for row in rows["test"]})
    }
    print(
        json.dumps(
            {
                "reviewed_videos": len(videos),
                "stride_seconds": args.stride_seconds,
                "horizon_seconds": args.horizon_seconds,
                "note": "Sequence continuity flags in these exports remain unconfirmed.",
                "split_videos": {
                    split: sorted({row[0] for row in rows[split]}) for split in ("train", "val", "test")
                },
                "new_test_videos_not_in_manifest": sorted(set(videos) - set(splits)),
                "persistence": {split: metrics(rows[split], persistence) for split in ("train", "val", "test")},
                "train_fitted_candidates": candidates,
                "selected_alpha_on_val": float(selected_alpha),
                "refitted_train_val_coefficients": selected.tolist(),
                "refitted_test": metrics(rows["test"], selected),
                "provisional_90_percent_interval": {
                    "method": "validation-video absolute residual quantile; train-only model",
                    "persistence": {
                        "val": prediction_interval(rows["val"], persistence, raw_radius),
                        "test": prediction_interval(rows["test"], persistence, raw_radius),
                    },
                    "calibrated": {
                        "val": prediction_interval(rows["val"], train_fitted, fitted_radius),
                        "test": prediction_interval(rows["test"], train_fitted, fitted_radius),
                    },
                },
                "test_by_video": test_by_video,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
