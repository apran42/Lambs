from ultralytics import YOLO
from config import settings

class PersonDetector:
    def __init__(self):
        # 중앙 세팅에서 지정한 모델 경로 로드 (yolov8n.pt 등)
        self.model = YOLO(settings.YOLO_MODEL_PATH)
        
    def track_objects(self, frame):
        """
        입력된 프레임에서 고유 ID를 부여하며 객체를 추적(track)합니다.
        """
        # 자원 최적화를 위해 내부 로그(verbose=False)는 끕니다.
        results = self.model.track(frame, persist=True, verbose=False)
        return results