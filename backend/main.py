import asyncio
import base64
import cv2
from fastapi import FastAPI, WebSocket
from database.influx_client import db_manager
from ai.camera import VideoStream
from ai.detector import PersonDetector
from utils.geometry import calculate_positions

app = FastAPI()

# 객체 생성 및 컴포넌트 초기화
video_stream = VideoStream()
detector = PersonDetector()

@app.get("/")
async def root():
    return {"message": "Shepherd-AI 관제 서버가 정상 구동 중입니다."}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0

    try:
        while True:
            # 1. [팀원 1 파트] 영상 프레임 캡처
            frame = video_stream.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # 2. [팀원 1 파트] AI 객체 인식 및 트래킹 호출
            results = detector.track_objects(frame)
            
            # 3. [팀원 2 파트] 바운딩 박스 좌표 추출 및 위치 계산 연산 호출
            processed_boxes = calculate_positions(results)
            count = len(processed_boxes)

            # 4. [본인 파트] 시계열 데이터 가치 보존: 20프레임(약 1초)에 1번씩만 DB 적재
            frame_count += 1
            if frame_count % 20 == 0:
                db_manager.save_crowd_stats(
                    camera_id="CAM_01", 
                    location="Main_Entrance", 
                    count=count
                )
                frame_count = 0

            # 5. 프론트엔드 실시간 전송용 경량화 전처리 (Jetson Nano 최적화)
            frame_resized = cv2.resize(frame, (640, 480))
            _, buffer = cv2.imencode('.jpg', frame_resized)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # 웹소켓 JSON 전송 데이터 패키징
            payload = {
                "image": img_base64,
                "count": count,
                "boxes": processed_boxes,
                "status": "Crowded" if count > 10 else "Normal"
            }
            
            await websocket.send_json(payload)
            await asyncio.sleep(0.05) # 약 20fps 주기 싱크 조정
            
    except Exception as e:
        print(f"⚠️ Websocket 연결 종료 또는 에러 발생: {e}")
    finally:
        # 클라이언트가 이탈하거나 에러가 나더라도 스트림이 깨지지 않게 안전 유지
        pass