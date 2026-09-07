import contextlib

import torch
from ultralytics import YOLO

from config import settings


class PersonDetector:
    """Local Ultralytics implementation; a TensorRT implementation can replace it."""

    def __init__(self) -> None:
        requested_device = settings.YOLO_DEVICE.strip().lower()
        use_cpu = requested_device == "cpu" or (
            not requested_device and not torch.cuda.is_available()
        )
        if use_cpu:
            torch.set_num_threads(max(1, settings.CPU_INFERENCE_THREADS))
            # PyTorch only permits changing this before inter-op work starts.
            with contextlib.suppress(RuntimeError):
                torch.set_num_interop_threads(max(1, settings.CPU_INTEROP_THREADS))
        self.model = YOLO(settings.YOLO_MODEL_PATH)

    def track_objects(self, frame):
        options = {
            "source": frame,
            "persist": True,
            "verbose": False,
            "classes": [0],
            "conf": settings.YOLO_CONFIDENCE,
            "iou": settings.YOLO_IOU,
            "imgsz": settings.YOLO_IMAGE_SIZE,
            "tracker": settings.YOLO_TRACKER,
        }
        if settings.YOLO_DEVICE:
            options["device"] = settings.YOLO_DEVICE
        return self.model.track(**options)

    def detect_batch(self, frames):
        """Run one shared model over frames from multiple cameras."""
        options = {
            "source": frames,
            "verbose": False,
            "classes": [0],
            "conf": settings.YOLO_CONFIDENCE,
            "iou": settings.YOLO_IOU,
            "imgsz": settings.YOLO_IMAGE_SIZE,
        }
        if settings.YOLO_DEVICE:
            options["device"] = settings.YOLO_DEVICE
        return self.model.predict(**options)
