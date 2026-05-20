import cv2
from config import settings

class VideoStream:
    def __init__(self):
        # 환경변수에 등록된 영상 경로 로드
        self.cap = cv2.VideoCapture(settings.VIDEO_PATH)
        
    def get_frame(self):
        """
        매 프레임을 읽어 반환합니다. 영상이 끝나면 처음으로 되돌려 무한 루프를 돕니다.
        """
        if not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            # 영상이 끝나면 프레임 인덱스를 0으로 초기화 (무한 루프)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
        return frame

    def release(self):
        self.cap.release()