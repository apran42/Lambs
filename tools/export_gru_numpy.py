"""Export a trained count GRU for NumPy-only inference on Jetson."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError("Output directory already exists: {}".format(args.output_dir))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    history = int(checkpoint["history_seconds"])
    stride = int(checkpoint["stride_seconds"])
    if (history != int(report["history_seconds"])
            or stride != int(report["stride_seconds"])):
        raise ValueError("Checkpoint and evaluation report do not match")

    args.output_dir.mkdir(parents=True)
    weights = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in checkpoint["state_dict"].items()
    }
    np.savez_compressed(str(args.output_dir / "weights.npz"), **weights)
    metadata = {
        "format": "lambs-count-gru-v1",
        "input": report["input"],
        "horizon_video_seconds": int(report["horizon_seconds"]),
        "history_seconds": history,
        "stride_seconds": stride,
        "best_epoch": int(checkpoint["best_epoch"]),
        "test_scores": report["scores"]["test"],
        "time_base": "source-video-seconds; live cameras use wall-clock seconds",
    }
    (args.output_dir / "model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
