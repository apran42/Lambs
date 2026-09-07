import asyncio
import contextlib
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2

from ai.camera import VideoStream
from ai.density import DensityAnalyzer
from ai.forecast import CrowdForecaster
from config import settings
from services.camera_registry import CameraDefinition
from services.stream_service import FrameSnapshot, StreamPacket


logger = logging.getLogger(__name__)


class CameraRuntime:
    def __init__(self, definition: CameraDefinition):
        self.definition = definition
        self.video_stream = VideoStream(
            source=definition.source,
            calibration_path=definition.calibration_path,
        )
        self.density_analyzer = DensityAnalyzer(
            definition.roi_config_path,
            camera_id=definition.camera_id,
        )
        self.forecaster = CrowdForecaster(
            horizon_seconds=settings.FORECAST_HORIZON_SECONDS,
            history_seconds=settings.FORECAST_HISTORY_SECONDS,
            min_trend_span_seconds=settings.FORECAST_MIN_TREND_SPAN_SECONDS,
            bucket_seconds=settings.FORECAST_BUCKET_SECONDS,
        )
        self.capture_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"capture-{definition.camera_id}",
        )
        self.packet_condition = asyncio.Condition()
        self.latest_snapshot: FrameSnapshot | None = None
        self.latest_packet: StreamPacket | None = None
        self.latest_analysis: dict | None = None
        self.last_analyzed_frame_id = 0
        self.capture_times: deque[float] = deque(maxlen=120)
        self.analysis_times: deque[float] = deque(maxlen=120)
        self.last_error: str | None = None

    def capture_fps(self) -> float:
        if len(self.capture_times) < 2:
            return 0.0
        elapsed = self.capture_times[-1] - self.capture_times[0]
        return (len(self.capture_times) - 1) / elapsed if elapsed > 0 else 0.0

    def analysis_fps(self) -> float:
        if len(self.analysis_times) < 2:
            return 0.0
        elapsed = self.analysis_times[-1] - self.analysis_times[0]
        return (len(self.analysis_times) - 1) / elapsed if elapsed > 0 else 0.0


class MultiCameraService:
    """Capture cameras independently and share one batch inference engine."""

    def __init__(self, definitions, detector, calculate_positions, db_manager=None):
        cv2.setNumThreads(max(1, settings.OPENCV_THREADS))
        self.detector = detector
        self.calculate_positions = calculate_positions
        self.db_manager = db_manager
        self.runtimes = {
            definition.camera_id: CameraRuntime(definition)
            for definition in definitions
        }
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="shared-inference",
        )
        self._new_frame_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._stopping = False
        self._last_db_write = {camera_id: 0.0 for camera_id in self.runtimes}
        self._inference_times: deque[float] = deque(maxlen=120)

    @property
    def camera_ids(self) -> list[str]:
        return list(self.runtimes)

    async def start(self) -> None:
        if self._tasks:
            return
        self._stopping = False
        self._tasks = [
            asyncio.create_task(
                self._capture_worker(runtime),
                name=f"capture-{camera_id}",
            )
            for camera_id, runtime in self.runtimes.items()
        ]
        self._tasks.append(
            asyncio.create_task(self._inference_worker(), name="shared-inference")
        )
        logger.info("Multi-camera pipeline started: %s", ", ".join(self.camera_ids))

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        for runtime in self.runtimes.values():
            runtime.video_stream.release()
            runtime.capture_executor.shutdown(wait=True, cancel_futures=True)
        self._inference_executor.shutdown(wait=True, cancel_futures=True)
        logger.info("Multi-camera pipeline stopped")

    @staticmethod
    def _encode_frame(frame) -> bytes:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buffer.tobytes()

    def _packet_metadata(self, runtime: CameraRuntime, snapshot: FrameSnapshot) -> dict:
        analysis = runtime.latest_analysis
        if analysis is None:
            density = runtime.density_analyzer.analyze(
                [],
                frame_width=int(snapshot.frame.shape[1]),
                frame_height=int(snapshot.frame.shape[0]),
            )
            analysis = {
                "count": 0,
                "detections": [],
                "analysis_frame_id": None,
                "analysis_captured_at": None,
                "analysis_processed_at": None,
                "analysis_completed_monotonic": None,
                "inference_ms": None,
                **density,
            }

        analysis_frame_id = analysis["analysis_frame_id"]
        completed = analysis["analysis_completed_monotonic"]
        metadata = {
            "protocol_version": 2,
            "camera_id": runtime.definition.camera_id,
            "camera_name": runtime.definition.name,
            "frame_id": snapshot.frame_id,
            "captured_at": snapshot.captured_at,
            "width": int(snapshot.frame.shape[1]),
            "height": int(snapshot.frame.shape[0]),
            "target_stream_fps": runtime.definition.target_fps,
            "capture_fps": runtime.capture_fps(),
            "analysis_fps": runtime.analysis_fps(),
            "analysis_frame_id": analysis_frame_id,
            "analysis_lag_frames": (
                snapshot.frame_id - analysis_frame_id
                if analysis_frame_id is not None
                else None
            ),
            "analysis_age_ms": (
                (time.monotonic() - completed) * 1000
                if completed is not None
                else None
            ),
            "stream_view": "camera-perspective",
            "density_coordinate_system": "bird-eye-metres",
            **{
                key: value
                for key, value in analysis.items()
                if key != "analysis_completed_monotonic"
            },
        }
        metadata["status"] = metadata.get("risk_level", "Unavailable")
        return metadata

    async def _capture_worker(self, runtime: CameraRuntime) -> None:
        loop = asyncio.get_running_loop()
        interval = 1.0 / runtime.definition.target_fps
        next_deadline = time.monotonic()
        frame_id = 0
        while not self._stopping:
            try:
                frame = await loop.run_in_executor(
                    runtime.capture_executor,
                    runtime.video_stream.get_frame,
                )
                if frame is None:
                    await asyncio.sleep(0.1)
                    next_deadline = time.monotonic()
                    continue
                image_bytes = await loop.run_in_executor(
                    runtime.capture_executor,
                    self._encode_frame,
                    frame,
                )
                frame_id += 1
                snapshot = FrameSnapshot(
                    frame_id=frame_id,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    frame=frame,
                )
                runtime.latest_snapshot = snapshot
                runtime.capture_times.append(time.monotonic())
                runtime.latest_packet = StreamPacket(
                    frame_id=frame_id,
                    image_bytes=image_bytes,
                    metadata=self._packet_metadata(runtime, snapshot),
                )
                async with runtime.packet_condition:
                    runtime.packet_condition.notify_all()
                runtime.last_error = None
                self._new_frame_event.set()
                next_deadline += interval
                now = time.monotonic()
                # Keep a stable cadence without accumulating sleep/scheduler drift.
                # If processing falls behind by a full frame, resume from now instead
                # of emitting an unbounded burst to catch up.
                if next_deadline < now - interval:
                    next_deadline = now
                await asyncio.sleep(max(0.0, next_deadline - now))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                runtime.last_error = f"capture: {exc}"
                logger.exception("Capture failed for %s", runtime.definition.camera_id)
                await asyncio.sleep(0.5)

    async def _inference_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stopping:
            await self._new_frame_event.wait()
            self._new_frame_event.clear()
            pending = [
                (runtime, runtime.latest_snapshot)
                for runtime in self.runtimes.values()
                if runtime.latest_snapshot is not None
                and runtime.latest_snapshot.frame_id > runtime.last_analyzed_frame_id
            ]
            if not pending:
                continue
            pending = pending[: max(1, settings.INFERENCE_BATCH_SIZE)]
            frames = [snapshot.frame for _runtime, snapshot in pending]
            started = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    self._inference_executor,
                    self.detector.detect_batch,
                    frames,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                completed = time.monotonic()
                self._inference_times.append(completed)
                for (runtime, snapshot), result in zip(pending, results):
                    detections = self.calculate_positions([result])
                    density = runtime.density_analyzer.analyze(
                        detections,
                        frame_width=int(snapshot.frame.shape[1]),
                        frame_height=int(snapshot.frame.shape[0]),
                    )
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
                    runtime.latest_analysis = {
                        "count": len(detections),
                        "detections": detections,
                        "analysis_frame_id": snapshot.frame_id,
                        "analysis_captured_at": snapshot.captured_at,
                        "analysis_processed_at": datetime.now(timezone.utc).isoformat(),
                        "analysis_completed_monotonic": completed,
                        "inference_ms": elapsed_ms,
                        "forecast_5m": forecast,
                        **density,
                    }
                    runtime.last_analyzed_frame_id = snapshot.frame_id
                    runtime.analysis_times.append(completed)
                    runtime.last_error = None
                    self._write_metric_if_due(runtime)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Shared batch inference failed")
                for runtime, _snapshot in pending:
                    runtime.last_error = f"inference: {exc}"
                await asyncio.sleep(0.2)
            finally:
                if any(
                    runtime.latest_snapshot is not None
                    and runtime.latest_snapshot.frame_id > runtime.last_analyzed_frame_id
                    for runtime in self.runtimes.values()
                ):
                    self._new_frame_event.set()

    def _write_metric_if_due(self, runtime: CameraRuntime) -> None:
        if self.db_manager is None or runtime.latest_analysis is None:
            return
        now = time.monotonic()
        camera_id = runtime.definition.camera_id
        if now - self._last_db_write[camera_id] < settings.DB_WRITE_INTERVAL_SECONDS:
            return
        self._last_db_write[camera_id] = now
        analysis = runtime.latest_analysis
        self.db_manager.save_crowd_stats(
            facility_name=runtime.definition.facility,
            location=runtime.definition.location,
            camera_id=camera_id,
            count=analysis["count"],
            roi_count=analysis["roi_count"],
            density=analysis["density_people_per_m2"],
            density_level=analysis["risk_level"],
        )

    async def wait_for_packet(
        self,
        camera_id: str,
        after_frame_id: int = 0,
        timeout: float = 2.0,
    ) -> StreamPacket | None:
        runtime = self.runtimes.get(camera_id)
        if runtime is None:
            raise KeyError(camera_id)

        async def wait() -> StreamPacket:
            async with runtime.packet_condition:
                await runtime.packet_condition.wait_for(
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

    def get_density_analyzer(self, camera_id: str) -> DensityAnalyzer:
        runtime = self.runtimes.get(camera_id)
        if runtime is None:
            raise KeyError(camera_id)
        return runtime.density_analyzer

    def get_forecast(self, camera_id: str) -> dict | None:
        runtime = self.runtimes.get(camera_id)
        if runtime is None:
            raise KeyError(camera_id)
        return runtime.forecaster.latest()

    def _inference_fps(self) -> float:
        if len(self._inference_times) < 2:
            return 0.0
        elapsed = self._inference_times[-1] - self._inference_times[0]
        batches_per_second = (
            (len(self._inference_times) - 1) / elapsed if elapsed > 0 else 0.0
        )
        return batches_per_second

    def health(self) -> dict:
        return {
            "running": bool(self._tasks) and not self._stopping,
            "shared_inference_batches_per_second": self._inference_fps(),
            "cameras": [
                {
                    "camera_id": camera_id,
                    "name": runtime.definition.name,
                    "target_fps": runtime.definition.target_fps,
                    "capture_fps": runtime.capture_fps(),
                    "analysis_fps": runtime.analysis_fps(),
                    "last_frame_id": (
                        runtime.latest_snapshot.frame_id
                        if runtime.latest_snapshot
                        else None
                    ),
                    "last_analyzed_frame_id": runtime.last_analyzed_frame_id or None,
                    "last_error": runtime.last_error,
                }
                for camera_id, runtime in self.runtimes.items()
            ],
        }
