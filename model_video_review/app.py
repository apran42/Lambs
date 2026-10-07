"""Minimal single-video viewer for the trained general-purpose person detector."""

from __future__ import annotations

import asyncio
import json
import os
import re
import struct
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "data"
STATIC_DIR = Path(__file__).resolve().parent / "static"
VIDEO_PATTERN = re.compile(r"^sample_data \((\d+)\)\.mp4$", re.IGNORECASE)
DEFAULT_MODEL = PROJECT_ROOT / "runs/training/lambs_yolov8n_21videos_v1/weights/best.pt"
MODEL_PATH = Path(os.getenv("SHEPHERD_REVIEW_MODEL", str(DEFAULT_MODEL))).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (PROJECT_ROOT / MODEL_PATH).resolve()

# Ultralytics 8.3.50 needs np.trapz on newer NumPy versions.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "data/training/.ultralytics"))

from ultralytics import YOLO  # noqa: E402


CONFIDENCE = 0.25
NMS_IOU = 0.45
MODEL_IMAGE_SIZE = 640
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def available_videos() -> list[dict]:
    """Only expose the numbered sample videos, never arbitrary client paths."""
    videos = []
    if not VIDEO_DIR.is_dir():
        return videos
    for path in VIDEO_DIR.iterdir():
        match = VIDEO_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            number = int(match.group(1))
            videos.append({"id": number, "name": f"{number}번 영상", "path": path})
    return sorted(videos, key=lambda item: item["id"])


def analyze_and_encode(frame: np.ndarray, model: YOLO, model_lock: threading.Lock) -> tuple[bytes, list[dict], float, int, int]:
    started = time.perf_counter()
    with model_lock:
        result = model.predict(
            frame,
            imgsz=MODEL_IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            classes=[0],
            device="cpu",
            verbose=False,
        )[0]
    inference_ms = (time.perf_counter() - started) * 1000
    detections = [
        {"box": [float(value) for value in box], "confidence": float(confidence)}
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
    ]

    height, width = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height)
    if scale < 1.0:
        display_width = max(1, round(width * scale))
        display_height = max(1, round(height * scale))
        display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    else:
        display_width, display_height = width, height
        display_frame = frame
    ok, encoded = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError("Could not encode video frame")
    return encoded.tobytes(), detections, inference_ms, display_width, display_height


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"General-purpose model not found: {MODEL_PATH}")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    app.state.model = YOLO(str(MODEL_PATH))
    app.state.model_lock = threading.Lock()
    yield


app = FastAPI(title="Shepherd-AI 영상별 객체 인식 확인", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/videos")
def videos():
    return {
        "videos": [{"id": item["id"], "name": item["name"]} for item in available_videos()],
        "model": "21영상 범용 모델",
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.is_file(), "video_count": len(available_videos())}


@app.websocket("/ws/video/{video_id}")
async def stream_video(websocket: WebSocket, video_id: int):
    selected = next((item for item in available_videos() if item["id"] == video_id), None)
    if selected is None:
        await websocket.close(code=1008, reason="Unknown video")
        return
    await websocket.accept()
    capture = cv2.VideoCapture(str(selected["path"]))
    if not capture.isOpened():
        await websocket.send_json({"type": "error", "message": "영상을 열 수 없습니다."})
        await websocket.close()
        return

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not 1 <= fps <= 120:
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    started = time.monotonic()
    next_frame = 0
    try:
        while True:
            target_frame = int((time.monotonic() - started) * fps)
            if total_frames > 0 and target_frame >= total_frames:
                await websocket.send_json({"type": "ended"})
                break
            if target_frame < next_frame:
                await asyncio.sleep(min(0.03, max(0.005, (next_frame / fps) - (time.monotonic() - started))))
                continue

            grabbed = True
            while next_frame <= target_frame:
                grabbed = capture.grab()
                if not grabbed:
                    break
                next_frame += 1
            if not grabbed:
                await websocket.send_json({"type": "ended"})
                break
            ok, frame = capture.retrieve()
            if not ok:
                await websocket.send_json({"type": "ended"})
                break

            source_height, source_width = frame.shape[:2]
            jpeg, detections, inference_ms, display_width, display_height = await asyncio.to_thread(
                analyze_and_encode, frame, websocket.app.state.model, websocket.app.state.model_lock
            )
            metadata = {
                "type": "frame",
                "video_id": video_id,
                "frame_index": next_frame - 1,
                "time_seconds": (next_frame - 1) / fps,
                "duration_seconds": total_frames / fps if total_frames > 0 else None,
                "source_width": source_width,
                "source_height": source_height,
                "display_width": display_width,
                "display_height": display_height,
                "count": len(detections),
                "detections": detections,
                "inference_ms": round(inference_ms, 1),
            }
            payload = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
            await websocket.send_bytes(struct.pack(">I", len(payload)) + payload + jpeg)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        # A client switching to another video can close the socket during send.
        if "websocket" not in str(exc).lower():
            raise
    finally:
        capture.release()
