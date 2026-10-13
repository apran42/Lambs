import asyncio
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from services.jetson_camera_service import JetsonCameraService
from ai.forecast import CrowdForecaster


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
                assert packet.metadata["forecast"]["method"] == "persistence-baseline"
                assert service.health()["inference"]["available"] is True
            finally:
                await service.stop()

    asyncio.run(scenario())


def test_worker_video_time_reaches_reviewed_demo_forecast():
    async def scenario():
        with TemporaryDirectory() as directory:
            definition = SimpleNamespace(
                camera_id="cam-01", name="CAM 01", facility="test",
                location="zone-a", target_fps=20.0,
                roi_config_path="{}/roi.json".format(directory),
            )
            service = JetsonCameraService([definition])
            reference = Path(__file__).resolve().parents[1] / "reviewed_demo_forecast.json"
            service.runtimes["cam-01"].forecaster = CrowdForecaster(
                mode="reviewed-demo", reviewed_demo_path=str(reference)
            )
            delivered = False

            def fake_fetch(_camera_id, _after):
                nonlocal delivered
                if delivered:
                    return None
                delivered = True
                return {
                    "protocol_version": 2, "camera_id": "cam-01", "frame_id": 1,
                    "width": 640, "height": 480, "analysis_frame_id": 1,
                    "source_video_id": "11", "analysis_source_frame_index": 900,
                    "analysis_source_fps": 30.0, "source_duration_seconds": 180.7,
                    "source_fps": 30.0, "capture_fps": 20.0,
                    "count": 1, "detections": [{
                        "box": [100.0, 100.0, 140.0, 240.0],
                        "confidence": 0.9, "track_id": None,
                    }],
                }, b"jpeg"

            service._fetch_packet = fake_fetch
            await service.start()
            try:
                packet = await service.wait_for_packet("cam-01", timeout=2)
                assert packet is not None
                forecast = packet.metadata["forecast"]
                assert forecast["method"] == "reviewed-video-demo"
                assert forecast["reference_video_time_seconds"] == 70.0
            finally:
                await service.stop()

    asyncio.run(scenario())
