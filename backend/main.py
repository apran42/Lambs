import asyncio
import json
import struct
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ai.camera import VideoStream
from ai.detector import PersonDetector
from utils.geometry import calculate_positions
from services.stream_service import StreamService  # 모듈화 서비스 임포트

# 라우터 임포트
from routers import stats
from routers import calibration
from routers import metrics

app = FastAPI(title="Shepherd-AI 관제 서버", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_stream = VideoStream(calibration_path="./calibration_data.json")
detector = PersonDetector()

# 비즈니스 로직을 처리할 서비스 인스턴스 생성
stream_service = StreamService(video_stream, detector, calculate_positions)

# 라우터 등록
app.include_router(stats.router)
app.include_router(calibration.router)
app.include_router(metrics.router)

@app.on_event("startup")
async def startup_event():
    # 백그라운드 AI 워커 구동 체계를 서비스 내부 함수로 깔끔하게 위임하여 실행
    asyncio.create_task(stream_service.start_background_ai_worker())

@app.get("/")
async def root():
    return {"message": "Shepherd-AI 관제 서버가 정상 구동 중입니다."}

# ==========================================
# 💡 메인 루프: AI 연산에 절대 간섭받지 않는 초고속 송출 채널
# ==========================================

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 서비스 모듈에서 기존 로직 그대로 처리된 결과를 받아옴
            image_bytes, payload = stream_service.get_optimized_streaming_frame()
            
            if payload is None:
                await asyncio.sleep(0.1)
                continue
            
            # 구조체 바이너리 패킹 작업 (Base64 성능 저하 방지)
            json_bytes = json.dumps(payload).encode('utf-8')
            json_length = len(json_bytes)

            packet = struct.pack(f"!I", json_length) + json_bytes + image_bytes
            await websocket.send_bytes(packet)
            
            await asyncio.sleep(0.05) # 약 20fps 주기 싱크 조정
            
    except Exception as e:
        print(f"⚠️ 스트리밍 Websocket 종료: {e}")
