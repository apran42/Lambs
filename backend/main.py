import asyncio
from fastapi import FastAPI, WebSocket
from database.influx_client import db_manager
from ai.camera import VideoStream
from ai.detector import PersonDetector
from utils.geometry import calculate_positions
from services.stream_service import StreamService  # 모듈화한 서비스 임포트

app = FastAPI()

# 객체 생성 및 컴포넌트 초기화
video_stream = VideoStream()
detector = PersonDetector()

# 비즈니스 로직을 처리할 서비스 인스턴스 생성
stream_service = StreamService(video_stream, detector, calculate_positions)

@app.get("/")
async def root():
    return {"message": "Shepherd-AI 관제 서버가 정상 구동 중입니다."}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 서비스 모듈에서 기존 로직 그대로 처리된 결과를 받아옴
            payload = await stream_service.process_next_frame()
            
            if payload is None:
                await asyncio.sleep(0.1)
                continue

            # 웹소켓 JSON 전송 데이터 패키징 이후 전송 파트
            await websocket.send_json(payload)
            await asyncio.sleep(0.05) # 약 20fps 주기 싱크 조정
            
    except Exception as e:
        print(f"⚠️ Websocket 연결 종료 또는 에러 발생: {e}")
    finally:
        # 클라이언트가 이탈하거나 에러가 나더라도 스트림이 깨지지 않게 안전 유지
        pass