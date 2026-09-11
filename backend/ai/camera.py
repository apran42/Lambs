from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from config import settings


logger = logging.getLogger(__name__)
VideoSource = Union[str, int]


def normalize_video_source(source: VideoSource) -> VideoSource:
    """Convert a numeric source such as VIDEO_PATH=0 into a USB camera index."""
    if isinstance(source, str) and source.strip().isdigit():
        return int(source.strip())
    return source


class VideoStream:
    """Own one OpenCV capture. Only the capture worker may call get_frame()."""

    def __init__(
        self,
        source: VideoSource | None = None,
        calibration_path: str | None = None,
    ):
        configured_source = settings.VIDEO_PATH if source is None else source
        self.source = normalize_video_source(configured_source)
        self.cap = cv2.VideoCapture(self.source)
        self._owner_thread_id: int | None = None
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._calibration_version: str | None = None

        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, settings.CAPTURE_FPS)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        if calibration_path and Path(calibration_path).is_file():
            self.load_calibration(calibration_path)

    def load_calibration(self, calibration_path: str) -> None:
        with open(calibration_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self._camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        self._dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
        self._calibration_version = data.get("version", "unknown")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Video source returned an invalid frame size")

        calibration_size = tuple(data.get("image_size", (width, height)))
        if calibration_size != (width, height):
            raise ValueError(
                "Calibration image size does not match the video source: "
                f"{calibration_size} != {(width, height)}"
            )

        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (width, height),
            alpha=0,
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix,
            self._dist_coeffs,
            None,
            new_matrix,
            (width, height),
            cv2.CV_32FC1,
        )
        logger.info("Loaded calibration version %s", self._calibration_version)

    @property
    def is_calibrated(self) -> bool:
        return self._map1 is not None and self._map2 is not None

    def get_frame(self):
        thread_id = threading.get_ident()
        if self._owner_thread_id is None:
            self._owner_thread_id = thread_id
        elif self._owner_thread_id != thread_id:
            raise RuntimeError("VideoStream must be read by exactly one worker thread")

        ok, frame = self.cap.read()
        if not ok and isinstance(self.source, str) and settings.VIDEO_LOOP:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        if self.is_calibrated:
            frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
        return cv2.resize(frame, (settings.FRAME_WIDTH, settings.FRAME_HEIGHT))

    def release(self) -> None:
        self.cap.release()
