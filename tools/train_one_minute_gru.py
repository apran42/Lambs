"""Train a small GRU count forecaster on reviewed timelines, split by video.

This is an offline comparison against persistence. Ground-truth mode uses
perfect current counts to test whether historical counts contain predictive
signal at all; it is not a deployed detector-to-forecast accuracy estimate.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_one_minute_forecast import detect_counts, reviewed_videos, split_by_video


class CountGRU(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history)
        return current + self.head(hidden[-1]).squeeze(-1) * 10.0


def make_rows(videos: dict, splits: dict, history_seconds: int, stride: int,
              count_key: str, horizon_seconds: int):
    grouped = defaultdict(list)
    for video_id, video in sorted(videos.items()):
        samples = video["samples"]
        times = sorted(samples)
        for origin in times:
            if origin < history_seconds or origin + horizon_seconds not in samples:
                continue
            history = []
            for second in range(int(origin - history_seconds), int(origin) + 1, stride):
                index = bisect.bisect_right(times, second) - 1
                if index < 0 or second - times[index] > 10:
                    break
                age = second - times[index]
                value = samples[times[index]].get(count_key)
                if value is None:
                    break
                history.append([value / 10.0, age / 10.0])
            if len(history) != history_seconds // stride + 1:
                continue
            grouped[splits.get(video_id, "test")].append(
                (video_id, origin, history, samples[origin][count_key],
                 samples[origin + horizon_seconds]["truth"])
            )
    return grouped


def score(rows: list, predictions: list[float]) -> dict:
    if not rows:
        return {"pairs": 0, "mae": None, "wape_percent": None}
    errors = [
        abs(max(0, round(float(prediction))) - row[4])
        for row, prediction in zip(rows, predictions)
    ]
    total = sum(row[4] for row in rows)
    return {
        "pairs": len(rows),
        "mae": round(sum(errors) / len(rows), 4),
        "wape_percent": round(100 * sum(errors) / total, 4) if total else None,
    }


def tensors(rows: list):
    history = torch.tensor([row[2] for row in rows], dtype=torch.float32)
    current = torch.tensor([row[3] for row in rows], dtype=torch.float32)
    target = torch.tensor([row[4] for row in rows], dtype=torch.float32)
    return history, current, target


def predict(model: CountGRU, rows: list) -> list[float]:
    model.eval()
    with torch.no_grad():
        history, current, _ = tensors(rows)
        return model(history, current).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/training"))
    parser.add_argument("--stride-seconds", type=int, default=2)
    parser.add_argument("--history-seconds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--report", type=Path,
                        help="Write detailed JSON metrics to a local file")
    parser.add_argument("--input", choices=("ground-truth", "detector"),
                        default="ground-truth")
    parser.add_argument("--model", type=Path,
                        default=Path("runs/training/lambs_yolov8n_21videos_v1/weights/best.pt"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.stride_seconds <= 0 or args.history_seconds <= 0 or args.horizon_seconds <= 0:
        raise ValueError("Stride, history, and forecast horizon must be positive")
    if args.history_seconds % args.stride_seconds:
        raise ValueError("History must be divisible by stride")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    videos = reviewed_videos(args.root, args.stride_seconds)
    if args.input == "detector":
        detect_counts(videos, args)
    rows = make_rows(videos, split_by_video(args.root), args.history_seconds,
                     args.stride_seconds,
                     "detected" if args.input == "detector" else "truth",
                     args.horizon_seconds)
    train, val, test = (rows[name] for name in ("train", "val", "test"))
    if not train or not val or not test:
        raise ValueError("Train, validation, and test each need forecast pairs")
    model = CountGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    x_train, c_train, y_train = tensors(train)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(len(train))
        for batch in indices.split(64):
            optimizer.zero_grad()
            prediction = model(x_train[batch], c_train[batch])
            loss = loss_fn(prediction, y_train[batch])
            loss.backward()
            optimizer.step()
        val_score = score(val, predict(model, val))["mae"]
        if val_score < best_val - 1e-8:
            best_state = copy.deepcopy(model.state_dict())
            best_val = val_score
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if epoch % 10 == 0:
            print(f"epoch={epoch} val_mae={val_score} best={best_val}",
                  file=sys.stderr, flush=True)
        if patience >= 20:
            break
    model.load_state_dict(best_state)
    result = {
        "input": "yolo_detector_count" if args.input == "detector"
                 else "reviewed_ground_truth_count_oracle",
        "reviewed_videos": len(videos),
        "horizon_seconds": args.horizon_seconds,
        "history_seconds": args.history_seconds,
        "stride_seconds": args.stride_seconds,
        "best_epoch_selected_on_val": best_epoch,
        "split_videos": {name: sorted({row[0] for row in rows[name]})
                         for name in ("train", "val", "test")},
        "scores": {
            name: {
                "persistence": score(rows[name], [row[3] for row in rows[name]]),
                "gru": score(rows[name], predict(model, rows[name])),
            }
            for name in ("train", "val", "test")
        },
        "test_by_video": {
            str(video_id): {
                "persistence": score(subset, [row[3] for row in subset]),
                "gru": score(subset, predict(model, subset)),
            }
            for video_id in sorted({row[0] for row in test})
            for subset in [[row for row in test if row[0] == video_id]]
        },
        "demo_train_videos": {
            str(video_id): {
                "persistence": score(subset, [row[3] for row in subset]),
                "gru": score(subset, predict(model, subset)),
            }
            for video_id in (11, 35)
            for subset in [[row for row in train if row[0] == video_id]]
        },
        "caveat": "All continuity flags are unconfirmed. Evaluations are video-level held out.",
    }
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "history_seconds": args.history_seconds,
                    "stride_seconds": args.stride_seconds, "best_epoch": best_epoch},
                   args.save_model)
        result["saved_model"] = str(args.save_model)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(json.dumps({
            "report": str(args.report),
            "best_epoch": best_epoch,
            "val": result["scores"]["val"],
            "test": result["scores"]["test"],
            "demo_train_videos": result["demo_train_videos"],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
