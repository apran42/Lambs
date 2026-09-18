"""API-only service used before the Jetson inference worker is connected."""

from __future__ import annotations

from typing import Any


class InferenceUnavailableError(RuntimeError):
    pass


class UnavailableCameraService:
    streaming_available = False

    def __init__(self, definitions: list[Any], detector: Any) -> None:
        self.definitions = definitions
        self.detector = detector
        self._running = False

    @property
    def camera_ids(self) -> list[str]:
        return [definition.camera_id for definition in self.definitions]

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def wait_for_packet(self, *_args: Any, **_kwargs: Any) -> None:
        raise InferenceUnavailableError(self.detector.unavailable_reason)

    def get_density_analyzer(self, camera_id: str) -> Any:
        if camera_id not in self.camera_ids:
            raise KeyError(camera_id)
        raise InferenceUnavailableError(self.detector.unavailable_reason)

    def get_forecast(self, camera_id: str) -> None:
        if camera_id not in self.camera_ids:
            raise KeyError(camera_id)
        return None

    def health(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "mode": "api-only",
            "streaming_available": False,
            "inference": self.detector.health(),
            "shared_inference_batches_per_second": 0.0,
            "cameras": [
                {
                    "camera_id": definition.camera_id,
                    "name": definition.name,
                    "target_fps": definition.target_fps,
                    "capture_fps": 0.0,
                    "analysis_fps": 0.0,
                    "last_frame_id": None,
                    "last_analyzed_frame_id": None,
                    "last_error": self.detector.unavailable_reason,
                }
                for definition in self.definitions
            ],
        }
