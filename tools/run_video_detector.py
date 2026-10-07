"""Run a trained person detector on videos and save annotated MP4 results."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

# Compatibility for Ultralytics 8.3.50 with recent NumPy versions.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/inference"))
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Infer every Nth source frame; output FPS is divided by the same value.",
    )
    return parser.parse_args()


def process_video(model: YOLO, source: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source.stem}_trained_detection_stride{args.stride}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.stride,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")

    counts: list[int] = []
    started = time.perf_counter()
    try:
        results = model.predict(
            source=str(source),
            stream=True,
            imgsz=args.image_size,
            conf=args.confidence,
            device=args.device,
            vid_stride=args.stride,
            verbose=False,
        )
        for result in results:
            count = int(len(result.boxes))
            counts.append(count)
            annotated = result.plot()
            cv2.rectangle(annotated, (12, 12), (270, 62), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                f"Person count: {count}",
                (24, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            if len(counts) % 300 == 0:
                print(f"{source.name}: {len(counts)} sampled frames processed", flush=True)
    finally:
        writer.release()

    elapsed = time.perf_counter() - started
    return {
        "source": str(source.resolve()),
        "output": str(output_path.resolve()),
        "frames": len(counts),
        "expected_frames": expected_frames,
        "source_fps": fps,
        "frame_stride": args.stride,
        "output_fps": fps / args.stride,
        "processing_fps": len(counts) / elapsed if elapsed else 0.0,
        "elapsed_seconds": elapsed,
        "mean_person_count": sum(counts) / len(counts) if counts else 0.0,
        "max_person_count": max(counts, default=0),
    }


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    summaries = [process_video(model, source, args.output_dir, args) for source in args.input]
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
