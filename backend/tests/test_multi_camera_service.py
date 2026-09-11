import asyncio
from tempfile import TemporaryDirectory

import numpy as np

import services.multi_camera_service as multi_camera_module
from services.camera_registry import CameraDefinition
from services.multi_camera_service import MultiCameraService


class FakeVideoStream:
    def __init__(self, source, calibration_path=None):
        self.source = source
        self.released = False
        self.value = 0

    def get_frame(self):
        self.value += 1
        return np.full((48, 64, 3), self.value % 255, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeBatchDetector:
    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def detect_batch(self, frames):
        self.calls += 1
        self.batch_sizes.append(len(frames))
        return [[] for _frame in frames]

    def health(self):
        return {"backend": "fake", "available": True, "reason": None}


def test_two_cameras_stream_while_sharing_one_detector():
    async def scenario():
        with TemporaryDirectory() as directory:
            definitions = [
                CameraDefinition(
                    camera_id=f"cam-0{index}",
                    name=f"CAM 0{index}",
                    source=index,
                    facility="test",
                    location=f"zone-{index}",
                    target_fps=20.0,
                    roi_config_path=f"{directory}/cam-0{index}.json",
                )
                for index in (1, 2)
            ]
            original_video_stream = multi_camera_module.VideoStream
            multi_camera_module.VideoStream = FakeVideoStream
            detector = FakeBatchDetector()
            service = MultiCameraService(
                definitions,
                detector,
                lambda _results: [],
            )
            try:
                await service.start()
                first, second = await asyncio.gather(
                    service.wait_for_packet("cam-01", timeout=2),
                    service.wait_for_packet("cam-02", timeout=2),
                )
                await asyncio.sleep(0.15)
                assert first is not None and second is not None
                assert first.metadata["protocol_version"] == 2
                assert second.metadata["protocol_version"] == 2
                assert first.metadata["target_stream_fps"] == 20.0
                assert second.metadata["target_stream_fps"] == 20.0
                assert detector.calls > 0
                assert service.health()["inference"]["backend"] == "fake"
                assert set(service.camera_ids) == {"cam-01", "cam-02"}
                assert len(service.runtimes) == 2
            finally:
                await service.stop()
                multi_camera_module.VideoStream = original_video_stream

            assert all(
                runtime.video_stream.released
                for runtime in service.runtimes.values()
            )

    asyncio.run(scenario())
