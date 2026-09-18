import asyncio
from types import SimpleNamespace

from ai.detector import UnavailableDetector
from services.unavailable_camera_service import (
    InferenceUnavailableError,
    UnavailableCameraService,
)


def test_api_only_service_reports_explicit_inference_status():
    async def scenario():
        definitions = [
            SimpleNamespace(camera_id="cam-01", name="CAM 01", target_fps=20.0)
        ]
        detector = UnavailableDetector("worker offline")
        service = UnavailableCameraService(definitions, detector)

        await service.start()
        health = service.health()
        assert health["running"] is True
        assert health["mode"] == "api-only"
        assert health["inference"] == {
            "backend": "unavailable",
            "available": False,
            "reason": "worker offline",
        }
        try:
            await service.wait_for_packet("cam-01")
        except InferenceUnavailableError:
            pass
        else:
            raise AssertionError("Unavailable streaming must raise an explicit error")
        await service.stop()

    asyncio.run(scenario())
