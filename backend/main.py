import json
import logging
import struct
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ai.detector import PersonDetector
from config import settings
from database.influx_client import db_manager
from routers import calibration, metrics, roi, stats
from services.camera_registry import load_camera_definitions
from services.multi_camera_service import MultiCameraService
from utils.geometry import calculate_positions


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    definitions = load_camera_definitions()
    detector = PersonDetector()
    service = MultiCameraService(
        definitions,
        detector,
        calculate_positions,
        db_manager,
    )
    app.state.multi_camera_service = service
    app.state.default_camera_id = definitions[0].camera_id
    await service.start()
    try:
        yield
    finally:
        await service.stop()
        db_manager.close()


app = FastAPI(
    title="Shepherd-AI 관제 서버",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(stats.router)
app.include_router(calibration.router)
app.include_router(metrics.router)
app.include_router(roi.router)


@app.get("/")
async def root():
    return {
        "message": "Shepherd-AI 관제 서버가 정상 구동 중입니다.",
        "camera_ids": app.state.multi_camera_service.camera_ids,
    }


@app.get("/health")
async def health():
    return app.state.multi_camera_service.health()


@app.get("/api/cameras")
async def cameras():
    return app.state.multi_camera_service.health()


@app.get("/api/forecast/{camera_id}")
async def camera_forecast(camera_id: str):
    try:
        forecast = app.state.multi_camera_service.get_forecast(camera_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown camera id") from exc
    if forecast is None:
        return {"camera_id": camera_id, "status": "warming_up", "data": None}
    return {"camera_id": camera_id, "status": "ok", "data": forecast}


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await stream_camera(websocket, websocket.app.state.default_camera_id)


@app.websocket("/ws/stream/{camera_id}")
async def camera_websocket_endpoint(websocket: WebSocket, camera_id: str):
    await stream_camera(websocket, camera_id)


async def stream_camera(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    last_frame_id = 0
    service: MultiCameraService = websocket.app.state.multi_camera_service
    if camera_id not in service.camera_ids:
        await websocket.close(code=1008, reason="Unknown camera id")
        return
    try:
        while True:
            packet = await service.wait_for_packet(camera_id, last_frame_id)
            if packet is None:
                continue
            last_frame_id = packet.frame_id
            json_bytes = json.dumps(packet.metadata).encode("utf-8")
            payload = struct.pack("!I", len(json_bytes)) + json_bytes + packet.image_bytes
            await websocket.send_bytes(payload)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket stream ended unexpectedly")
