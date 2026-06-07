import cv2
import json
import numpy as np
from pathlib import Path
from config import settings

class VideoStream:
    def __init__(self, calibration_path: str = None):
        # 환경변수에 등록된 영상 경로 로드
        self.cap = cv2.VideoCapture(settings.VIDEO_PATH)

        # 왜곡 보정 파라미터 초기화
        self._camera_matrix = None
        self._dist_coeffs = None
        self._map1 = None  # undistort remap 테이블 X
        self._map2 = None  # undistort remap 테이블 Y
        self._calibration_version = None

        # 캘리브레이션 파일이 있으면 로드
        if calibration_path and Path(calibration_path).exists():
            self.load_calibration(calibration_path)

    def load_calibration(self, calibration_path: str):
        """
        JSON 캘리브레이션 파일을 로드하고 undistort remap 테이블을 미리 계산합니다.
        (initUndistortRectifyMap으로 사전 계산 → get_frame 호출 시 remap만 수행)
        """
        with open(calibration_path, "r") as f:
            data = json.load(f)

        self._camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        self._dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        self._calibration_version = data.get("version", "unknown")

        # 영상 크기 기준으로 remap 테이블 사전 계산
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix, self._dist_coeffs, (w, h), alpha=0
        )
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix, self._dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
        )

        print(f"✅ 캘리브레이션 로드 완료 (버전: {self._calibration_version})")

    @property
    def is_calibrated(self):
        return self._map1 is not None

    def get_frame(self):
        """
        매 프레임을 읽어 반환합니다.
        캘리브레이션이 로드된 경우 undistort를 적용합니다.
        영상이 끝나면 처음으로 되돌려 무한 루프를 돕니다.
        """
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            # 영상이 끝나면 프레임 인덱스를 0으로 초기화 (무한 루프)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if frame is None:
            return None

        # 캘리브레이션이 있으면 왜곡 보정 적용 (remap이 가장 빠른 방식)
        if self.is_calibrated:
            frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

        return frame

    def release(self):
        self.cap.release()
