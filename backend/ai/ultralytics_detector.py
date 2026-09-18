"""Ultralytics detector used by the development-PC runtime."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from ultralytics import YOLO

from config import settings
from utils.geometry import calculate_positions


class UltralyticsDetector:
    backend_name = "ultralytics"
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        requested_device = settings.YOLO_DEVICE.strip().lower()
        use_cpu = requested_device == "cpu" or (
            not requested_device and not torch.cuda.is_available()
        )
        if use_cpu:
            torch.set_num_threads(max(1, settings.CPU_INFERENCE_THREADS))
            with contextlib.suppress(RuntimeError):
                torch.set_num_interop_threads(max(1, settings.CPU_INTEROP_THREADS))
        self.model = YOLO(settings.YOLO_MODEL_PATH)

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "verbose": False,
            "classes": [0],
            "conf": settings.YOLO_CONFIDENCE,
            "iou": settings.YOLO_IOU,
            "imgsz": settings.YOLO_IMAGE_SIZE,
        }
        if settings.YOLO_DEVICE:
            options["device"] = settings.YOLO_DEVICE
        return options

    def track_objects(self, frame: Any) -> list[dict[str, Any]]:
        options = self._options()
        options.update(
            {
                "source": frame,
                "persist": True,
                "tracker": settings.YOLO_TRACKER,
            }
        )
        return calculate_positions(self.model.track(**options))

    def detect_batch(self, frames: list[Any]) -> list[list[dict[str, Any]]]:
        options = self._options()
        options["source"] = frames
        results = self.model.predict(**options)
        return [calculate_positions([result]) for result in results]

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "available": self.available,
            "reason": None,
            "model_path": settings.YOLO_MODEL_PATH,
            "device": settings.YOLO_DEVICE or "auto",
        }
