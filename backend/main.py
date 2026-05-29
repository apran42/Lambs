import asyncio
import json
import struct
import cv2
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware

# 우리가 구축한 패키지 모듈들 import
from database.influx_client import db_manager
from ai.camera import VideoStream
from ai.detector import PersonDetector
from utils.geometry import calculate_positions

app = FastAPI(title="Shepherd-AI 관제 서버", version="1.0.0")

# 웹소켓 및 API가 프론트엔드(React 등)와 원활하게 통신할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 컴포넌트 객체 초기화
video_stream = VideoStream()
detector = PersonDetector()

# 서버 상태 체크용 엔드포인트
@app.get("/")
async def root():
    return {"message": "Shepherd-AI 관제 서버가 정상 구동 중입니다."}

# ==========================================
# [데이터 조회 및 통계 API 파트]
# ==========================================
@app.get('/api/v1/stats/recent')
async def get_recent_stats(
    minutes: int = Query(default=60, description="조회할 최근 시간 (분 단위)")   
):
    """
    최근 {minutes} 분의 시계열 데이터를 list 형태로 반환합니다.
    """
    stats_data = db_manager.get_recent_crowd_stats(minutes=minutes)
    return {
        "status": "SUCCESS",
        "count": len(stats_data),
        "data": stats_data
    }

# ==========================================
# [실시간 초고속 하이브리드 웹소켓 스트리밍 파트]
# ==========================================
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    frame_count = 0

    try:
        while True:
            # 1. [팀원 1 담당] 영상 프레임 캡처
            frame = video_stream.get_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            # 2. [팀원 1 담당] AI 객체 인식 및 트래킹 호출
            results = detector.track_objects(frame)
            
            # 3. [팀원 2 담당] 바운딩 박스 좌표 추출 및 위치 계산 연산 호출
            processed_boxes = calculate_positions(results)
            count = len(processed_boxes)
            
            # 4. [본인 담당] 시계열 데이터 가치 보존을 위한 필터링 처리
            #    Jetson Nano 자원 절약을 위해 20프레임(약 1초)에 1번씩만 InfluxDB에 Write
            frame_count += 1
            if frame_count % 20 == 0:
                # [실전 배포 시 주석 해제하여 사용]
                # db_manager.save_crowd_stats(
                #     facility_name="Test_Campus", # 장소명
                #     location="Main_Entrance", # 구역명
                #     camera_id="CAM_01", # 카메라 번호
                #     count=count # 감지된 인원 수
                # )
                frame_count = 0

            # 5. 프론트엔드 실시간 전송용 경량화 전처리 (Jetson Nano 최적화)
            #    [개선] 중복 인코딩 절차를 하나로 통일하고 가벼운 압축 품질(80) 적용
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            image_bytes = buffer.tobytes()

            # 6. 메타데이터 패키징 및 고속 바이너리 패킹 (더블 버퍼링/0.75배속 밀림 완벽 해소)
            meta_payload = {
                "count": count,
                "boxes": processed_boxes,
                "status": "Crowded" if count > 10 else "Normal"
            }
            json_bytes = json.dumps(meta_payload).encode('utf-8')
            json_length = len(json_bytes)

            # 구조화 패킹: [4바이트: JSON 길이 정보] + [JSON 바이트 데이터] + [순수 이미지 JPEG 바이트 데이터]
            packet = struct.pack(f"!I", json_length) + json_bytes + image_bytes
            
            # 텍스트 오버헤드가 없는 순수 바이너리 패킷 송신 (초당 25fps 준수)
            await websocket.send_bytes(packet)
            await asyncio.sleep(0.04)
            
    except Exception as e:
        print(f"⚠️ Websocket 연결 종료 또는 에러 발생: {e}")
    finally:
        # 클라이언트가 브라우저 창을 닫아 세션이 끊겨도 백엔드 스트림이 파괴되지 않도록 안전 유지
        pass