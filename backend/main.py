import asyncio
import json
import struct
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ai.camera import VideoStream
from ai.detector import PersonDetector
from utils.geometry import calculate_positions
from services.stream_service import StreamService  # 모듈화 서비스 임포트

app = FastAPI(title="Shepherd-AI 관제 서버", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 객체 생성 및 컴포넌트 초기화
video_stream = VideoStream()
detector = PersonDetector()

# 비즈니스 로직을 처리할 서비스 인스턴스 생성
stream_service = StreamService(video_stream, detector, calculate_positions)

@app.on_event("startup")
async def startup_event():
    # 백그라운드 AI 워커 구동 체계를 서비스 내부 함수로 깔끔하게 위임하여 실행
    asyncio.create_task(stream_service.start_background_ai_worker())

@app.get("/")
async def root():
    return {"message": "Shepherd-AI 관제 서버가 정상 구동 중입니다."}

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
        print(f"⚠️ Websocket 연결 종료 또는 에러 발생: {e}")
    finally:
        # 클라이언트가 이탈하거나 에러가 나더라도 스트림이 깨지지 않게 안전 유지
        pass