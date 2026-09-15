"""Command-line smoke test for the Jetson TensorRT engine."""

import argparse
import json
import os
import sys
import time

import cv2

from jetson_worker.postprocess import decode_yolo_output, draw_detections, prepare_input
from jetson_worker.tensorrt_engine import TensorRTEngine


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, help="Path to a TensorRT .engine file")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Run one image and emit one JSON result")
    source.add_argument("--video", help="Run frames from a video file")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--output-image", help="Optional annotated image path")
    return parser.parse_args()


def _infer_frame(engine, image, confidence, iou):
    tensor, transform = prepare_input(image, engine.input_shape)
    outputs, inference_ms = engine.infer(tensor)
    detections = decode_yolo_output(
        outputs[0],
        transform,
        confidence_threshold=confidence,
        iou_threshold=iou,
    )
    return detections, inference_ms


def _image_result(args, engine):
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError("Could not read image: {}".format(args.image))
    detections, inference_ms = _infer_frame(
        engine,
        image,
        args.confidence,
        args.iou,
    )
    if args.output_image:
        output_directory = os.path.dirname(os.path.abspath(args.output_image))
        if output_directory and not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        if not cv2.imwrite(args.output_image, draw_detections(image, detections)):
            raise RuntimeError("Could not write image: {}".format(args.output_image))
    return {
        "status": "ok",
        "source": args.image,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "count": len(detections),
        "detections": detections,
        "inference_ms": inference_ms,
        "engine_input_shape": list(engine.input_shape),
    }


def _video_results(args, engine):
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError("Could not open video: {}".format(args.video))
    frame_id = 0
    started = time.time()
    inference_total_ms = 0.0
    try:
        while frame_id < max(1, args.max_frames):
            ok, image = capture.read()
            if not ok:
                break
            frame_id += 1
            detections, inference_ms = _infer_frame(
                engine,
                image,
                args.confidence,
                args.iou,
            )
            inference_total_ms += inference_ms
            yield {
                "status": "ok",
                "frame_id": frame_id,
                "count": len(detections),
                "detections": detections,
                "inference_ms": inference_ms,
            }
    finally:
        capture.release()
        elapsed = max(time.time() - started, 1e-9)
        summary = {
            "status": "summary",
            "frames": frame_id,
            "wall_fps": frame_id / elapsed,
            "mean_inference_ms": (
                inference_total_ms / frame_id if frame_id else None
            ),
        }
        print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)


def main():
    args = _arguments()
    with TensorRTEngine(args.engine) as engine:
        if args.image:
            print(
                json.dumps(
                    _image_result(args, engine),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            return
        for result in _video_results(args, engine):
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()

