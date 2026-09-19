import asyncio
import json
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from services.jetson_camera_service import JetsonCameraService


def test_worker_packet_is_enriched_without_opencv():
    async def scenario():
        with TemporaryDirectory() as directory:
            definition = SimpleNamespace(
                camera_id="cam-01",
                name="CAM 01",
                facility="test",
                location="zone-a",
                target_fps=20.0,
                roi_config_path="{}/roi.json".format(directory),
            )
            service = JetsonCameraService([definition])
            delivered = False

            def fake_fetch(_camera_id, _after):
                nonlocal delivered
                if delivered:
                    return None
                delivered = True
                metadata = {
                    "protocol_version": 2,
                    "camera_id": "cam-01",
                    "frame_id": 1,
                    "width": 640,
                    "height": 480,
                    "analysis_frame_id": 1,
                    "count": 1,
                    "detections": [
                        {
                            "box": [100.0, 100.0, 140.0, 240.0],
                            "confidence": 0.9,
                            "track_id": None,
                        }
                    ],
                    "capture_fps": 20.0,
                    "analysis_fps": 10.0,
                }
                return metadata, b"jpeg"

            service._fetch_packet = fake_fetch
            await service.start()
            try:
                packet = await service.wait_for_packet("cam-01", timeout=2)
                assert packet is not None
                assert packet.image_bytes == b"jpeg"
                assert packet.metadata["roi_count"] == 1
                assert packet.metadata["forecast"]["horizon_seconds"] == 60
                assert service.health()["inference"]["available"] is True
            finally:
                await service.stop()

    asyncio.run(scenario())
