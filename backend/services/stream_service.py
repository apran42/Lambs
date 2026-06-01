import asyncio
import base64
import cv2
# 기존 임포트 그대로 유지
# from database.influx_client import db_manager 

class StreamService:
    def __init__(self, video_stream, detector, calculate_positions):
        self.video_stream = video_stream
        self.detector = detector
        self.calculate_positions = calculate_positions
        self.frame_count = 0

    async def process_next_frame(self):
        # 1. [팀원 1 파트] 영상 프레임 캡처
        frame = self.video_stream.get_frame()
        if frame is None:
            return None

        # 2. [팀원 1 파트] AI 객체 인식 및 트래킹 호출
        results = self.detector.track_objects(frame)
        
        # 3. [팀원 2 파트] 바운딩 박스 좌표 추출 및 위치 계산 연산 호출
        processed_boxes = self.calculate_positions(results)
        count = len(processed_boxes)

        # 4. [본인 파트] 시계열 데이터 가치 보존: 20프레임(약 1초)에 1번씩만 DB 적재
        """
        self.frame_count += 1
        if self.frame_count % 20 == 0:
            db_manager.save_crowd_stats(
                facility_name="Test_Campus", 
                camera_id="CAM_01", 
                location="Main_Entrance", 
                count=count
            )
            self.frame_count = 0
        """

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
        
        return payload