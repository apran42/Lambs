"""Persistent capture/inference service compatible with Jetson Python 3.6."""

import datetime
import json
import os
import threading
import time
from collections import deque

import cv2

from jetson_worker.postprocess import decode_yolo_output, prepare_input


def _utc_now():
    return datetime.datetime.utcnow().isoformat() + "Z"


def _rate(samples):
    if len(samples) < 2:
        return 0.0
    elapsed = samples[-1] - samples[0]
    return float(len(samples) - 1) / elapsed if elapsed > 0 else 0.0


class CameraState(object):
    def __init__(self, definition, width, height, jpeg_quality):
        self.definition = definition
        self.width = int(width)
        self.height = int(height)
        self.jpeg_quality = int(jpeg_quality)
        self.condition = threading.Condition()
        self.frame = None
        self.image_bytes = None
        self.frame_id = 0
        self.captured_at = None
        self.analysis = None
        self.last_analyzed_frame_id = 0
        self.capture_times = deque(maxlen=120)
        self.analysis_times = deque(maxlen=120)
        self.last_error = None
        self.capture = None

    def packet(self, after_frame_id, timeout_seconds):
        deadline = time.time() + max(0.0, float(timeout_seconds))
        with self.condition:
            while self.image_bytes is None or self.frame_id <= after_frame_id:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)
            analysis = self.analysis or {
                "count": 0,
                "detections": [],
                "analysis_frame_id": None,
                "analysis_captured_at": None,
                "analysis_processed_at": None,
                "inference_ms": None,
            }
            analysis_id = analysis.get("analysis_frame_id")
            completed = analysis.get("analysis_completed_monotonic")
            metadata = {
                "protocol_version": 2,
                "camera_id": self.definition["id"],
                "camera_name": self.definition.get("name", self.definition["id"]),
                "frame_id": int(self.frame_id),
                "captured_at": self.captured_at,
                "width": self.width,
                "height": self.height,
                "target_stream_fps": float(self.definition.get("target_fps", 20.0)),
                "capture_fps": _rate(self.capture_times),
                "analysis_fps": _rate(self.analysis_times),
                "analysis_frame_id": analysis_id,
                "analysis_lag_frames": (
                    int(self.frame_id - analysis_id) if analysis_id is not None else None
                ),
                "analysis_age_ms": (
                    (time.monotonic() - completed) * 1000.0
                    if completed is not None
                    else None
                ),
                "stream_view": "camera-perspective",
                "density_coordinate_system": "bird-eye-metres",
            }
            for key, value in analysis.items():
                if key != "analysis_completed_monotonic":
                    metadata[key] = value
            return metadata, self.image_bytes


class TensorRTWorkerService(object):
    def __init__(
        self,
        definitions,
        engine,
        confidence=0.35,
        iou=0.45,
        width=640,
        height=480,
        jpeg_quality=80,
    ):
        self.engine = engine
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.states = {
            item["id"]: CameraState(item, width, height, jpeg_quality)
            for item in definitions
            if item.get("enabled", True)
        }
        if not self.states:
            raise ValueError("At least one enabled camera is required")
        self.stopping = threading.Event()
        self.new_frame = threading.Event()
        self.threads = []
        self.started_at = None

    @property
    def camera_ids(self):
        return list(self.states)

    def start(self, warmup_iterations=5):
        if self.threads:
            return
        self.engine.warmup(warmup_iterations)
        self.started_at = time.time()
        for state in self.states.values():
            thread = threading.Thread(
                target=self._capture_loop,
                args=(state,),
                name="capture-{}".format(state.definition["id"]),
            )
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        inference = threading.Thread(target=self._inference_loop, name="inference")
        inference.daemon = True
        inference.start()
        self.threads.append(inference)

    def stop(self):
        self.stopping.set()
        self.new_frame.set()
        for state in self.states.values():
            if state.capture is not None:
                state.capture.release()
            with state.condition:
                state.condition.notify_all()
        for thread in self.threads:
            thread.join(3.0)
        self.threads = []

    @staticmethod
    def _source(value):
        if isinstance(value, int):
            return value
        text = str(value)
        return int(text) if text.isdigit() else text

    def _capture_loop(self, state):
        source = self._source(state.definition["source"])
        is_file = isinstance(source, str) and os.path.isfile(source)
        interval = 1.0 / max(1.0, float(state.definition.get("target_fps", 20.0)))
        while not self.stopping.is_set():
            if state.capture is None or not state.capture.isOpened():
                state.capture = cv2.VideoCapture(source)
                if not state.capture.isOpened():
                    state.last_error = "Could not open source: {}".format(source)
                    self.stopping.wait(1.0)
                    continue
            started = time.monotonic()
            ok, frame = state.capture.read()
            if not ok:
                if is_file and state.definition.get("loop", True):
                    state.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                state.last_error = "Could not read source: {}".format(source)
                state.capture.release()
                state.capture = None
                self.stopping.wait(0.5)
                continue
            if frame.shape[1] != state.width or frame.shape[0] != state.height:
                frame = cv2.resize(frame, (state.width, state.height))
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), state.jpeg_quality]
            )
            if not ok:
                state.last_error = "JPEG encoding failed"
                continue
            with state.condition:
                state.frame = frame
                state.image_bytes = encoded.tobytes()
                state.frame_id += 1
                state.captured_at = _utc_now()
                state.capture_times.append(time.monotonic())
                state.last_error = None
                state.condition.notify_all()
            self.new_frame.set()
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                self.stopping.wait(remaining)

    def _inference_loop(self):
        camera_ids = self.camera_ids
        cursor = 0
        while not self.stopping.is_set():
            self.new_frame.wait(0.5)
            self.new_frame.clear()
            found = False
            for offset in range(len(camera_ids)):
                index = (cursor + offset) % len(camera_ids)
                state = self.states[camera_ids[index]]
                with state.condition:
                    if state.frame is None or state.frame_id <= state.last_analyzed_frame_id:
                        continue
                    frame = state.frame.copy()
                    frame_id = state.frame_id
                    captured_at = state.captured_at
                found = True
                cursor = (index + 1) % len(camera_ids)
                try:
                    tensor, transform = prepare_input(frame, self.engine.input_shape)
                    outputs, inference_ms = self.engine.infer(tensor)
                    detections = decode_yolo_output(
                        outputs[0], transform, self.confidence, self.iou
                    )
                    completed = time.monotonic()
                    with state.condition:
                        state.analysis = {
                            "count": int(len(detections)),
                            "detections": detections,
                            "analysis_frame_id": int(frame_id),
                            "analysis_captured_at": captured_at,
                            "analysis_processed_at": _utc_now(),
                            "analysis_completed_monotonic": completed,
                            "inference_ms": float(inference_ms),
                        }
                        state.last_analyzed_frame_id = frame_id
                        state.analysis_times.append(completed)
                        state.last_error = None
                        state.condition.notify_all()
                except Exception as exc:
                    state.last_error = "inference: {}".format(exc)
                break
            if found and any(
                state.frame_id > state.last_analyzed_frame_id
                for state in self.states.values()
            ):
                self.new_frame.set()

    def packet(self, camera_id, after_frame_id=0, timeout_seconds=2.0):
        state = self.states.get(camera_id)
        if state is None:
            raise KeyError(camera_id)
        return state.packet(int(after_frame_id), float(timeout_seconds))

    def health(self):
        return {
            "status": "ok",
            "running": bool(self.threads) and not self.stopping.is_set(),
            "uptime_seconds": time.time() - self.started_at if self.started_at else 0.0,
            "inference": {"backend": "tensorrt", "available": True},
            "cameras": [
                {
                    "camera_id": camera_id,
                    "target_fps": float(state.definition.get("target_fps", 20.0)),
                    "capture_fps": _rate(state.capture_times),
                    "analysis_fps": _rate(state.analysis_times),
                    "last_frame_id": int(state.frame_id) if state.frame_id else None,
                    "last_analyzed_frame_id": (
                        int(state.last_analyzed_frame_id)
                        if state.last_analyzed_frame_id
                        else None
                    ),
                    "last_error": state.last_error,
                }
                for camera_id, state in self.states.items()
            ],
        }


def load_camera_definitions(path):
    with open(path, "r") as config_file:
        data = json.load(config_file)
    base = os.path.dirname(os.path.abspath(path))
    definitions = []
    for item in data.get("cameras", []):
        item = dict(item)
        source = item.get("source")
        if isinstance(source, str) and not source.isdigit() and not os.path.isabs(source):
            item["source"] = os.path.abspath(os.path.join(base, source))
        if float(item.get("target_fps", 20.0)) < 20.0:
            raise ValueError("{} target_fps must be at least 20".format(item.get("id")))
        definitions.append(item)
    return definitions
