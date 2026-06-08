import asyncio
import base64
import cv2
from concurrent.futures import ThreadPoolExecutor # 스레드 풀 생성
# 기존 임포트 그대로 유지
from database.influx_client import db_manager 

class StreamService:
    def __init__(self, video_stream, detector, calculate_positions):
        self.video_stream = video_stream
        self.detector = detector
        self.calculate_positions = calculate_positions

        # AI 연산을 위한 독립 스레드 풀
        self.executor = ThreadPoolExecutor(max_workers=1)

        # 실시간 AI 스냅샷 공유 변수
        self.shared_ai_data = {
            "count": 0,
            "boxes": [],
            "status": "Normal"
        }

    # AI 연산은 동기 메서드로
    def _heavy_ai_inference(self, frame):
            if frame is None:
                return None
            results = self.detector.track_objects(frame)
            processed_boxes = self.calculate_positions(results)
            return {
                "count": len(processed_boxes),
                "boxes": processed_boxes,
                "status": "Crowded" if len(processed_boxes) > 10 else "Normal",
            }

    # 백그라운드에서 메인 루프 간섭 없이 AI만 무한히
    async def start_background_ai_worker(self):
        loop = asyncio.get_running_loop()
        print("🚀 StreamService 내부 격리형 AI 워커 가동")
        while True:
            try:
                frame = self.video_stream.get_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # 🔥 스레드 풀로 무거운 추론 연산 격리 유도
                ai_result = await loop.run_in_executor(self.executor, self._heavy_ai_inference, frame)
                if ai_result:
                    self.shared_ai_data = ai_result
                    db_manager.save_crowd_stats(
                        facility_name="MainFacility",
                        location="Zone_A",
                        camera_id="Cam_01",
                        count=ai_result["count"]
                    )

                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"⚠️ 서비스 백그라운드 AI 에러: {e}")
                await asyncio.sleep(1)

    # 💡 웹소켓 송출용 초고속 프레임 패키징 함수 (바이너리 원본 보존 + 즉석 드로잉)
    def get_optimized_streaming_frame(self):
        frame = self.video_stream.get_frame()
        if frame is None:
            return None, None

        # 백그라운드 AI가 가공해 둔 가장 최신의 깨끗한 좌표 데이터 가져오기
        current_boxes = self.shared_ai_data.get("boxes", [])
        
        # 30fps 원본 스트림 위에 즉석에서 가볍게 박스 드로잉 (메모리 카피 제로)
        for box in current_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 압축 연산 진행
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        meta_payload = {
            "count": self.shared_ai_data["count"],
            "boxes": current_boxes,
            "status": self.shared_ai_data["status"]
        }
        
        return buffer.tobytes(), meta_payload