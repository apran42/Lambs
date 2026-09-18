import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEncodedBuffer:
    def __init__(self, value):
        self.value = value

    def tobytes(self):
        return f"jpeg:{self.value}".encode()

import services.stream_service as stream_service_module
from services.stream_service import StreamService


class FakeFrame:
    shape = (32, 48, 3)

    def __init__(self, value):
        self.value = value


class FakeVideoStream:
    def __init__(self):
        self.read_count = 0
        self.released = False

    def get_frame(self):
        self.read_count += 1
        return FakeFrame(self.read_count)

    def release(self):
        self.released = True


class FakeDetector:
    def track_objects(self, frame):
        return fake_positions(frame)


def fake_positions(_results):
    return [{"box": [1.0, 2.0, 20.0, 25.0], "track_id": 7, "confidence": 0.9}]


def test_clients_share_one_encoded_packet():
    async def scenario():
        original_imencode = stream_service_module.cv2.imencode
        stream_service_module.cv2.imencode = lambda _extension, frame, _options: (
            True,
            FakeEncodedBuffer(frame.value),
        )
        video = FakeVideoStream()
        service = StreamService(video, FakeDetector(), fake_positions)
        try:
            await service.start()
            first, second = await asyncio.gather(
                service.wait_for_packet(timeout=3),
                service.wait_for_packet(timeout=3),
            )
            assert first is not None and second is not None
            assert first.frame_id == second.frame_id
            assert first.image_bytes == second.image_bytes
            assert first.metadata["frame_id"] == first.frame_id
            assert first.metadata["camera_id"] == "cam-01"
            assert first.metadata["detections"][0]["track_id"] == 7
            assert first.metadata["width"] == 48
            assert first.metadata["height"] == 32
            json.dumps(first.metadata)
            await service.stop()
            assert video.released
        finally:
            stream_service_module.cv2.imencode = original_imencode

    asyncio.run(scenario())
