from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ai.density import DensityAnalyzer
from ai.forecast import CrowdForecaster
from config import settings
from services.stream_types import StreamPacket


logger = logging.getLogger(__name__)


class JetsonCameraRuntime:
    def __init__(self, definition):
        self.definition = definition
        self.density_analyzer = DensityAnalyzer(
            definition.roi_config_path, camera_id=definition.camera_id
        )
        self.forecaster = CrowdForecaster(
            horizon_seconds=settings.FORECAST_HORIZON_SECONDS,
            history_seconds=settings.FORECAST_HISTORY_SECONDS,
            min_trend_span_seconds=settings.FORECAST_MIN_TREND_SPAN_SECONDS,
            bucket_seconds=settings.FORECAST_BUCKET_SECONDS,
        )
        self.condition = asyncio.Condition()
        self.latest_packet = None
        self.last_analysis_frame_id = 0
        self.latest_derived_analysis = None
        self.last_error = None
        self.last_received_at = None
        self.last_db_write = 0.0


class JetsonCameraService:
    """Bridge a Python 3.6 TensorRT worker into the Python 3.8 API."""

    streaming_available = True

    def __init__(self, definitions, db_manager=None, worker_url=None):
        self.worker_url = (worker_url or settings.JETSON_WORKER_URL).rstrip("/")
        self.db_manager = db_manager
        self.runtimes = {
            definition.camera_id: JetsonCameraRuntime(definition)
            for definition in definitions
        }
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(self.runtimes)), thread_name_prefix="jetson-worker"
        )
        self._tasks = []
        self._stopping = False

    @property
    def camera_ids(self):
        return list(self.runtimes)

    async def start(self):
        if self._tasks:
            return
        self._stopping = False
        self._tasks = [
            asyncio.create_task(self._poll(runtime), name="jetson-{}".format(camera_id))
            for camera_id, runtime in self.runtimes.items()
        ]

    async def stop(self):
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._executor.shutdown(wait=True)

    def _fetch_packet(self, camera_id, after_frame_id):
        query = urlencode(
            {
                "after_frame_id": int(after_frame_id),
                "timeout_ms": int(settings.JETSON_WORKER_POLL_TIMEOUT_SECONDS * 1000),
            }
        )
        url = "{}/api/packet/{}?{}".format(
            self.worker_url, quote(camera_id, safe=""), query
        )
        request = Request(url, headers={"Accept": "application/octet-stream"})
        try:
            with urlopen(
                request, timeout=settings.JETSON_WORKER_REQUEST_TIMEOUT_SECONDS
            ) as response:
                if response.status == 204:
                    return None
                payload = response.read()
        except HTTPError as exc:
            if exc.code == 204:
                return None
            raise RuntimeError("Worker HTTP {}".format(exc.code)) from exc
        if len(payload) < 4:
            raise RuntimeError("Worker returned a truncated packet")
        metadata_length = struct.unpack("!I", payload[:4])[0]
        boundary = 4 + metadata_length
        if len(payload) <= boundary:
            raise RuntimeError("Worker packet has no JPEG payload")
        metadata = json.loads(payload[4:boundary].decode("utf-8"))
        return metadata, payload[boundary:]

    async def _poll(self, runtime):
        loop = asyncio.get_running_loop()
        last_frame_id = 0
        while not self._stopping:
            try:
                fetched = await loop.run_in_executor(
                    self._executor,
                    self._fetch_packet,
                    runtime.definition.camera_id,
                    last_frame_id,
                )
                if fetched is None:
                    continue
                metadata, image_bytes = fetched
                frame_id = int(metadata["frame_id"])
                last_frame_id = frame_id
                analysis_frame_id = metadata.get("analysis_frame_id")
                if (
                    analysis_frame_id is not None
                    and int(analysis_frame_id) > runtime.last_analysis_frame_id
                ):
                    detections = metadata.get("detections", [])
                    density = runtime.density_analyzer.analyze(
                        detections,
                        frame_width=int(metadata["width"]),
                        frame_height=int(metadata["height"]),
                    )
                    completed = time.monotonic()
                    forecast = runtime.forecaster.update(
                        count=density["roi_count"],
                        local_peak_density=density[
                            "applied_peak_density_people_per_m2"
                        ],
                        zone_area_m2=density["zone_area_m2"],
                        relaxed_max=density["thresholds"]["relaxed_max"],
                        danger_min=density["thresholds"]["danger_min"],
                        measured_at=completed,
                    )
                    metadata.update(density)
                    metadata["forecast_5m"] = forecast
                    metadata["status"] = density["risk_level"]
                    runtime.latest_derived_analysis = {
                        **density,
                        "count": int(metadata.get("count", len(detections))),
                        "detections": detections,
                        "forecast_5m": forecast,
                        "status": density["risk_level"],
                    }
                    runtime.last_analysis_frame_id = int(analysis_frame_id)
                    self._write_metric_if_due(runtime, metadata)
                elif runtime.latest_derived_analysis is not None:
                    metadata.update(runtime.latest_derived_analysis)
                packet = StreamPacket(frame_id, image_bytes, metadata)
                async with runtime.condition:
                    runtime.latest_packet = packet
                    runtime.last_received_at = time.monotonic()
                    runtime.last_error = None
                    runtime.condition.notify_all()
            except asyncio.CancelledError:
                raise
            except (OSError, URLError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                runtime.last_error = str(exc)
                await asyncio.sleep(0.5)

    def _write_metric_if_due(self, runtime, metadata):
        if self.db_manager is None:
            return
        now = time.monotonic()
        if now - runtime.last_db_write < settings.DB_WRITE_INTERVAL_SECONDS:
            return
        runtime.last_db_write = now
        self.db_manager.save_crowd_stats(
            facility_name=runtime.definition.facility,
            location=runtime.definition.location,
            camera_id=runtime.definition.camera_id,
            count=metadata.get("count", 0),
            timestamp=(
                metadata.get("analysis_captured_at") or metadata.get("captured_at")
            ),
            roi_count=metadata.get("roi_count"),
            density=metadata.get("density_people_per_m2"),
            density_level=metadata.get("risk_level"),
        )

    async def wait_for_packet(self, camera_id, after_frame_id=0, timeout=2.0):
        runtime = self.runtimes.get(camera_id)
        if runtime is None:
            raise KeyError(camera_id)

        async def wait():
            async with runtime.condition:
                await runtime.condition.wait_for(
                    lambda: self._stopping
                    or (
                        runtime.latest_packet is not None
                        and runtime.latest_packet.frame_id > after_frame_id
                    )
                )
                if self._stopping or runtime.latest_packet is None:
                    raise asyncio.CancelledError
                return runtime.latest_packet

        try:
            return await asyncio.wait_for(wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def get_density_analyzer(self, camera_id):
        runtime = self.runtimes.get(camera_id)
        if runtime is None:
            raise KeyError(camera_id)
        return runtime.density_analyzer

    def get_forecast(self, camera_id):
        runtime = self.runtimes.get(camera_id)
        if runtime is None:
            raise KeyError(camera_id)
        return runtime.forecaster.latest()

    def health(self):
        now = time.monotonic()
        cameras = []
        for camera_id, runtime in self.runtimes.items():
            packet = runtime.latest_packet
            cameras.append(
                {
                    "camera_id": camera_id,
                    "name": runtime.definition.name,
                    "target_fps": runtime.definition.target_fps,
                    "capture_fps": packet.metadata.get("capture_fps", 0.0) if packet else 0.0,
                    "analysis_fps": packet.metadata.get("analysis_fps", 0.0) if packet else 0.0,
                    "last_frame_id": packet.frame_id if packet else None,
                    "last_analyzed_frame_id": runtime.last_analysis_frame_id or None,
                    "last_packet_age_seconds": (
                        now - runtime.last_received_at
                        if runtime.last_received_at is not None
                        else None
                    ),
                    "last_error": runtime.last_error,
                }
            )
        available = any(runtime.latest_packet is not None for runtime in self.runtimes.values())
        return {
            "running": bool(self._tasks) and not self._stopping,
            "mode": "jetson-worker-bridge",
            "worker_url": self.worker_url,
            "inference": {
                "backend": "jetson-tensorrt",
                "available": available,
                "reason": None if available else "Waiting for TensorRT worker packets",
            },
            "cameras": cameras,
        }
