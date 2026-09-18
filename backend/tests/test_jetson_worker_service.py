import json
import threading
import time

import numpy as np

import jetson_worker.service as worker_module
from jetson_worker.service import TensorRTWorkerService
from jetson_worker.server import ThreadingHTTPServer, WorkerHandler
from services.jetson_camera_service import JetsonCameraService


class FakeCapture:
    def __init__(self, _source):
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.opened = False

    def set(self, _key, _value):
        return True


class FakeEngine:
    input_shape = (1, 3, 64, 64)

    def warmup(self, _iterations):
        return []

    def infer(self, _tensor):
        return [np.zeros((1, 5, 6), dtype=np.float32)], 5.0


def test_persistent_worker_produces_serializable_latest_packet():
    original_capture = worker_module.cv2.VideoCapture
    worker_module.cv2.VideoCapture = FakeCapture
    service = TensorRTWorkerService(
        [
            {
                "id": "cam-01",
                "name": "CAM 01",
                "source": 0,
                "target_fps": 20,
            }
        ],
        FakeEngine(),
        width=64,
        height=48,
    )
    try:
        service.start(warmup_iterations=0)
        deadline = time.time() + 2.0
        packet = None
        while time.time() < deadline:
            packet = service.packet("cam-01", timeout_seconds=0.2)
            if packet and packet[0]["analysis_frame_id"] is not None:
                break
        assert packet is not None
        metadata, image_bytes = packet
        assert metadata["frame_id"] >= 1
        assert metadata["analysis_frame_id"] >= 1
        assert metadata["count"] == 0
        assert image_bytes.startswith(b"\xff\xd8")
        json.dumps(metadata)
    finally:
        service.stop()
        worker_module.cv2.VideoCapture = original_capture


def test_worker_http_packet_round_trip():
    original_capture = worker_module.cv2.VideoCapture
    worker_module.cv2.VideoCapture = FakeCapture
    worker = TensorRTWorkerService(
        [{"id": "cam-01", "source": 0, "target_fps": 20}],
        FakeEngine(),
        width=64,
        height=48,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), WorkerHandler)
    server.worker = worker
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    try:
        worker.start(warmup_iterations=0)
        thread.start()
        bridge = JetsonCameraService(
            [], worker_url="http://127.0.0.1:{}".format(server.server_port)
        )
        metadata, image_bytes = bridge._fetch_packet("cam-01", 0)
        assert metadata["protocol_version"] == 2
        assert metadata["camera_id"] == "cam-01"
        assert image_bytes.startswith(b"\xff\xd8")
    finally:
        server.shutdown()
        server.server_close()
        worker.stop()
        thread.join(2.0)
        worker_module.cv2.VideoCapture = original_capture
