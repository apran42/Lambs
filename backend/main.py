from fastapi import FastAPI, WebSocket
from ultralytics import YOLO
import cv2
import asyncio
import json

app = FastAPI()
model = YOLO("../yolov8n.pt")  # root 위치

@app.get("/")
async def root():
    return {"message": "CCTV Analysis Server API is running"}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # 1분 샘플 영상 읽기 (경로는 본인 환경에 맞게 수정)
    video_path = "../data/sample_data (1).mp4" 
    cap = cv2.VideoCapture(video_path)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 영상 무한 루프
                continue

            # YOLOv8 추론 (성능을 위해 5프레임당 1번 추천)
            results = model.track(frame, persist=True, verbose=False)
            
            # 탐지된 객체의 좌표 및 데이터 추출
            boxes = results[0].boxes.xyxy.tolist() if results[0].boxes else []
            count = len(boxes)

            # 프론트엔드로 전송할 데이터 포맷
            data = {
                "count": count,
                "boxes": boxes, # [[x1, y1, x2, y2], ...]
                "status": "Crowded" if count > 10 else "Normal"
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(0.05) # 약 20fps 송신
            
    except Exception as e:
        print(f"Websocket Error: {e}")
    finally:
        cap.release()