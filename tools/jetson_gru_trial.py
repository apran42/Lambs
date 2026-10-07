"""Package reviewed frames, then benchmark a trained count GRU on Jetson.

The evaluate command runs on Jetson's Python 3.6 with NumPy, OpenCV, and the
existing TensorRT worker modules. It does not modify or restart live servers.
"""

from __future__ import print_function

import argparse
import bisect
import json
import os
import shutil
import sys

import numpy as np


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


class NumpyCountGRU(object):
    def __init__(self, weights_path):
        with np.load(weights_path, allow_pickle=False) as values:
            self.ih = values["gru.weight_ih_l0"]
            self.hh = values["gru.weight_hh_l0"]
            self.bi = values["gru.bias_ih_l0"]
            self.bh = values["gru.bias_hh_l0"]
            self.head_weight = values["head.weight"]
            self.head_bias = values["head.bias"]
        self.hidden_size = self.hh.shape[1]

    def predict(self, history, current):
        hidden = np.zeros(self.hidden_size, dtype=np.float32)
        size = self.hidden_size
        for entry in history:
            input_value = np.asarray(entry, dtype=np.float32)
            from_input = np.dot(self.ih, input_value) + self.bi
            from_hidden = np.dot(self.hh, hidden) + self.bh
            reset = _sigmoid(from_input[:size] + from_hidden[:size])
            update = _sigmoid(from_input[size:2 * size] + from_hidden[size:2 * size])
            candidate = np.tanh(from_input[2 * size:] + reset * from_hidden[2 * size:])
            hidden = (1.0 - update) * candidate + update * hidden
        return float(current + (np.dot(self.head_weight, hidden)[0]
                                + self.head_bias[0]) * 10.0)


def package(args):
    import pathlib
    import torch

    from evaluate_one_minute_forecast import reviewed_videos, split_by_video

    destination = pathlib.Path(args.package_dir)
    if destination.exists():
        raise ValueError("Package path already exists: {}".format(destination))
    destination.mkdir(parents=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = {key: value.detach().numpy().astype(np.float32)
               for key, value in checkpoint["state_dict"].items()}
    np.savez_compressed(str(destination / "weights.npz"), **weights)
    shutil.copy2(__file__, str(destination / "evaluate.py"))

    reviewed = reviewed_videos(pathlib.Path(args.review_root), 10)
    splits = split_by_video(pathlib.Path(args.review_root))
    payload = {"horizon_seconds": args.horizon_seconds,
               "history_seconds": checkpoint["history_seconds"],
               "stride_seconds": checkpoint["stride_seconds"],
               "videos": []}
    for video_id in args.video_ids:
        if video_id not in reviewed:
            raise ValueError("No reviewed video {}".format(video_id))
        entries = []
        for second, sample in sorted(reviewed[video_id]["samples"].items()):
            name = "video{}/second{:04d}.jpg".format(video_id, int(second))
            target = destination / name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(str(sample["image"]), str(target))
            entries.append({"second": int(second), "truth": sample["truth"],
                            "image": name})
        payload["videos"].append({"id": video_id,
                                  "split": splits.get(video_id, "test"),
                                  "samples": entries})
    (destination / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"package": str(destination),
                      "videos": args.video_ids,
                      "images": sum(len(video["samples"]) for video in payload["videos"])}))


def _score(rows, forecasts):
    if not rows:
        return {"pairs": 0, "mae": None, "wape_percent": None}
    absolute = [abs(max(0, round(float(value))) - row[2])
                for row, value in zip(rows, forecasts)]
    total = sum(row[2] for row in rows)
    return {"pairs": len(rows),
            "mae": round(sum(absolute) / float(len(rows)), 4),
            "wape_percent": round(100.0 * sum(absolute) / total, 4)
                            if total else None}


def evaluate(args):
    import cv2

    sys.path.insert(0, args.backend_root)
    from jetson_worker.postprocess import decode_yolo_output, prepare_input
    from jetson_worker.tensorrt_engine import TensorRTEngine

    package_dir = os.path.abspath(args.package_dir)
    with open(os.path.join(package_dir, "manifest.json"), "r") as stream:
        manifest = json.load(stream)
    model = NumpyCountGRU(os.path.join(package_dir, "weights.npz"))
    engine = TensorRTEngine(args.engine)
    all_rows = {"train": [], "val": [], "test": []}
    all_predictions = {"train": [], "val": [], "test": []}
    all_baselines = {"train": [], "val": [], "test": []}
    all_detector_rows = {"train": [], "val": [], "test": []}
    all_detector_counts = {"train": [], "val": [], "test": []}
    per_video = {}
    try:
        engine.warmup(5)
        for video in manifest["videos"]:
            samples = {}
            for entry in video["samples"]:
                image = cv2.imread(os.path.join(package_dir, entry["image"]))
                if image is None:
                    raise RuntimeError("Cannot read {}".format(entry["image"]))
                tensor, transform = prepare_input(image, engine.input_shape)
                outputs, _inference_ms = engine.infer(tensor)
                detections = decode_yolo_output(
                    outputs[0], transform, confidence_threshold=args.confidence,
                    iou_threshold=args.iou)
                samples[entry["second"]] = {"truth": entry["truth"],
                                             "detected": len(detections)}
            times = sorted(samples)
            rows = []
            forecasts = []
            baselines = []
            horizon = manifest["horizon_seconds"]
            history_seconds = manifest["history_seconds"]
            stride = manifest["stride_seconds"]
            for origin in times:
                if origin < history_seconds or origin + horizon not in samples:
                    continue
                history = []
                for query in range(origin - history_seconds, origin + 1, stride):
                    index = bisect.bisect_right(times, query) - 1
                    if index < 0 or query - times[index] > 10:
                        break
                    age = query - times[index]
                    history.append([samples[times[index]]["detected"] / 10.0,
                                    age / 10.0])
                if len(history) != history_seconds // stride + 1:
                    continue
                current = samples[origin]["detected"]
                truth = samples[origin + horizon]["truth"]
                rows.append((video["id"], origin, truth, current))
                forecasts.append(model.predict(history, current))
                baselines.append(current)
            split = video["split"]
            detector_rows = [(video["id"], second, samples[second]["truth"])
                             for second in times]
            detector_counts = [samples[second]["detected"] for second in times]
            all_detector_rows[split].extend(detector_rows)
            all_detector_counts[split].extend(detector_counts)
            all_rows[split].extend(rows)
            all_predictions[split].extend(forecasts)
            all_baselines[split].extend(baselines)
            per_video[str(video["id"])] = {
                "split": split,
                "detector": _score(detector_rows, detector_counts),
                "persistence": _score(rows, baselines),
                "gru": _score(rows, forecasts),
            }
            print("evaluated video {}: {} forecast pairs".format(video["id"], len(rows)),
                  file=sys.stderr)
    finally:
        engine.close()
    result = {"input": "jetson_tensorrt_detection_count",
              "horizon_video_seconds": manifest["horizon_seconds"],
              "confidence": args.confidence,
              "iou": args.iou,
              "scores": {split: {"detector": _score(all_detector_rows[split], all_detector_counts[split]),
                                  "persistence": _score(all_rows[split], all_baselines[split]),
                                  "gru": _score(all_rows[split], all_predictions[split])}
                         for split in ("train", "val", "test")},
              "by_video": per_video}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w") as stream:
            stream.write(encoded)
    print(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")
    local = sub.add_parser("package")
    local.add_argument("--checkpoint", required=True)
    local.add_argument("--review-root", default="data/training")
    local.add_argument("--package-dir", required=True)
    local.add_argument("--video-ids", type=int, nargs="+", default=[11, 35, 28, 36, 40, 44])
    local.add_argument("--horizon-seconds", type=int, default=40)
    remote = sub.add_parser("evaluate")
    remote.add_argument("--package-dir", required=True)
    remote.add_argument("--backend-root", required=True)
    remote.add_argument("--engine", required=True)
    remote.add_argument("--confidence", type=float, default=0.35)
    remote.add_argument("--iou", type=float, default=0.45)
    remote.add_argument("--output")
    args = parser.parse_args()
    if args.action == "package":
        package(args)
    elif args.action == "evaluate":
        evaluate(args)
    else:
        parser.error("choose package or evaluate")


if __name__ == "__main__":
    main()
