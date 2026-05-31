import asyncio
import json
import struct
import cv2
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware

from database.influx_client import db_manager
from ai.camera import VideoStream
from ai.detector import PersonDetector
from utils.geometry import calculate_positions
from routers.stats import router as stats_router

app = FastAPI(title="Shepherd-AI 관제 서버", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(stats_router)

video_stream = VideoStream()
detector = PersonDetector()

# 💡 [핵심 1] 무거운 AI 연산을 메인 스레드 방해 없이 처리할 독립 스레드 풀 생성
executor = ThreadPoolExecutor(max_workers=1)

# 실시간 AI 스냅샷 공유 변수
shared_ai_data = {
    "count": 0,
    "boxes": [],
    "status": "Normal"
}

# 💡 [핵심 2] CPU/GPU를 많이 쓰는 AI 추론은 일반 동기식(def) 함수로 작성합니다.
def heavy_ai_inference(frame):
    if frame is None:
        return None
    results = detector.track_objects(frame)
    processed_boxes = calculate_positions(results)
    return {
        "count": len(processed_boxes),
        "boxes": processed_boxes,
        "status": "Crowded" if len(processed_boxes) > 10 else "Normal"
    }

# 백그라운드에서 주기적으로 AI를 돌려주는 태스크
async def background_ai_worker():
    global shared_ai_data
    loop = asyncio.get_running_loop()
    frame_count = 0
    
    print("🚀 스레드 풀 격리형 AI 워커 가동")
    while True:
        try:
            frame = video_stream.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue
            
            # 🔥 무거운 AI 함수를 별도의 스레드 풀(Executor)로 보내어 메인 루프 결손 방지
            ai_result = await loop.run_in_executor(executor, heavy_ai_inference, frame)
            
            if ai_result:
                shared_ai_data = ai_result
                
                # 💡 [본인 파트] 시계열 데이터 가치 보존: InfluxDB 적재 처리 가동
                frame_count += 1
                if frame_count % 15 == 0:
                    # 🛠️ [버그 수정] 변수 count를 ai_result["count"]로 올바르게 수정 및 주석 해제
                    db_manager.save_crowd_stats(
                        facility_name="Test_Campus", 
                        location="Main_Entrance", 
                        camera_id="CAM_01", 
                        count=ai_result["count"] 
                    )
                    frame_count = 0
            
            await asyncio.sleep(0.01)
        except Exception as e:
            print(f"⚠️ AI 워커 에러: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_ai_worker())

@app.get("/")
async def root():
    return {"message": "Shepherd-AI 관제 서버가 정상 구동 중입니다."}

# ==========================================
# 💡 메인 루프: AI 연산에 절대 간섭받지 않는 초고속 송출 채널
# ==========================================
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global shared_ai_data

    try:
        while True:
            frame = video_stream.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # 전처리 및 압축 (이 연산은 매우 가벼움)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            image_bytes = buffer.tobytes()

            # 별도 스레드에서 실시간 업데이트 중인 최신 AI 스냅샷을 얹어서 전송
            meta_payload = {
                "count": shared_ai_data["count"],
                "boxes": shared_ai_data["boxes"],
                "status": shared_ai_data["status"]
            }
            json_bytes = json.dumps(meta_payload).encode('utf-8')
            json_length = len(json_bytes)

            packet = struct.pack(f"!I", json_length) + json_bytes + image_bytes
            await websocket.send_bytes(packet)
            
            # 카메라 원본 규격에 맞게 30fps(0.033초) 속도로 칼같이 밀어줌
            await asyncio.sleep(0.033)
            
    except Exception as e:
        print(f"⚠️ 스트리밍 Websocket 종료: {e}")
