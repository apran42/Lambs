from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import cv2

from config import settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameSnapshot:
    frame_id: int
    captured_at: str
    frame: Any


@dataclass(frozen=True)
class StreamPacket:
    frame_id: int
    image_bytes: bytes
    metadata: dict


class StreamService:
    """Single-producer capture/inference pipeline shared by every client."""

    def __init__(
        self,
        video_stream,
        detector,
        calculate_positions: Callable,
        db_manager=None,
        density_analyzer=None,
        *,
        camera_id: str | None = None,
        facility_name: str | None = None,
        location: str | None = None,
    ) -> None:
        self.video_stream = video_stream
        self.detector = detector
        # Kept in the constructor for compatibility with older callers. Detector
        # implementations now return normalized detection dictionaries directly.
        self.calculate_positions = calculate_positions
        self.db_manager = db_manager
        self.density_analyzer = density_analyzer
        self.camera_id = camera_id or settings.CAMERA_ID
        self.facility_name = facility_name or settings.FACILITY_NAME
        self.location = location or settings.CAMERA_LOCATION
        self._capture_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"capture-{self.camera_id}"
        )
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"inference-{self.camera_id}"
        )
        self._frame_condition = asyncio.Condition()
        self._packet_condition = asyncio.Condition()
        self._latest_frame: FrameSnapshot | None = None
        self._latest_packet: StreamPacket | None = None
        self._tasks: list[asyncio.Task] = []
        self._stopping = False
        self._last_db_write = 0.0
        self._last_error: str | None = None

    async def start(self) -> None:
        if self._tasks:
            return
        self._stopping = False
        self._tasks = [
            asyncio.create_task(
                self._capture_worker(), name=f"capture-{self.camera_id}"
            ),
            asyncio.create_task(
                self._inference_worker(), name=f"inference-{self.camera_id}"
            ),
        ]
        logger.info("Stream pipeline started for %s", self.camera_id)

    async def stop(self) -> None:
        self._stopping = True
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self.video_stream.release()
        self._capture_executor.shutdown(wait=True)
        self._inference_executor.shutdown(wait=True)
        logger.info("Stream pipeline stopped for %s", self.camera_id)

    async def _capture_worker(self) -> None:
        loop = asyncio.get_running_loop()
        frame_id = 0
        interval = 1.0 / max(settings.CAPTURE_FPS, 1.0)
        while not self._stopping:
            started = time.monotonic()
            try:
                frame = await loop.run_in_executor(
                    self._capture_executor, self.video_stream.get_frame
                )
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                frame_id += 1
                snapshot = FrameSnapshot(
                    frame_id=frame_id,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    frame=frame,
                )
                async with self._frame_condition:
                    self._latest_frame = snapshot
                    self._frame_condition.notify_all()
                self._last_error = None
                await asyncio.sleep(max(0.0, interval - (time.monotonic() - started)))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = f"capture: {exc}"
                logger.exception("Capture worker failed for %s", self.camera_id)
                await asyncio.sleep(0.5)

    async def _next_frame(self, after_frame_id: int) -> FrameSnapshot:
        async with self._frame_condition:
            await self._frame_condition.wait_for(
                lambda: self._stopping
                or (
                    self._latest_frame is not None
                    and self._latest_frame.frame_id > after_frame_id
                )
            )
            if self._stopping or self._latest_frame is None:
                raise asyncio.CancelledError
            return self._latest_frame

    def _infer_and_encode(self, snapshot: FrameSnapshot) -> StreamPacket:
        detections = self.detector.track_objects(snapshot.frame)
        ok, buffer = cv2.imencode(
            ".jpg",
            snapshot.frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), settings.JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")

        count = len(detections)
        density = (
            self.density_analyzer.analyze(
                detections,
                frame_width=int(snapshot.frame.shape[1]),
                frame_height=int(snapshot.frame.shape[0]),
            )
            if self.density_analyzer is not None
            else {
                "roi_count": count,
                "density_people_per_m2": None,
                "density_level": "Unavailable",
                "grid": [],
                "local_peak": None,
            }
        )
        metadata = {
            "protocol_version": 1,
            "camera_id": self.camera_id,
            "frame_id": snapshot.frame_id,
            "captured_at": snapshot.captured_at,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "width": int(snapshot.frame.shape[1]),
            "height": int(snapshot.frame.shape[0]),
            "count": count,
            "status": density.get("risk_level", density["density_level"]),
            "detections": detections,
            **density,
        }
        return StreamPacket(snapshot.frame_id, buffer.tobytes(), metadata)

    async def _inference_worker(self) -> None:
        loop = asyncio.get_running_loop()
        last_frame_id = 0
        while not self._stopping:
            try:
                snapshot = await self._next_frame(last_frame_id)
                last_frame_id = snapshot.frame_id
                packet = await loop.run_in_executor(
                    self._inference_executor, self._infer_and_encode, snapshot
                )
                async with self._packet_condition:
                    self._latest_packet = packet
                    self._packet_condition.notify_all()
                self._write_metric_if_due(packet)
                self._last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = f"inference: {exc}"
                logger.exception("Inference worker failed for %s", self.camera_id)
                await asyncio.sleep(0.2)

    def _write_metric_if_due(self, packet: StreamPacket) -> None:
        if self.db_manager is None:
            return
        now = time.monotonic()
        if now - self._last_db_write < settings.DB_WRITE_INTERVAL_SECONDS:
            return
        self._last_db_write = now
        self.db_manager.save_crowd_stats(
            facility_name=self.facility_name,
            location=self.location,
            camera_id=self.camera_id,
            count=packet.metadata["count"],
            roi_count=packet.metadata.get("roi_count"),
            density=packet.metadata.get("density_people_per_m2"),
            density_level=packet.metadata.get("density_level"),
        )

    async def wait_for_packet(
        self, after_frame_id: int = 0, timeout: float = 2.0
    ) -> StreamPacket | None:
        async def wait() -> StreamPacket:
            async with self._packet_condition:
                await self._packet_condition.wait_for(
                    lambda: self._stopping
                    or (
                        self._latest_packet is not None
                        and self._latest_packet.frame_id > after_frame_id
                    )
                )
                if self._stopping or self._latest_packet is None:
                    raise asyncio.CancelledError
                return self._latest_packet

        try:
            return await asyncio.wait_for(wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def health(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "running": bool(self._tasks) and not self._stopping,
            "last_frame_id": self._latest_frame.frame_id if self._latest_frame else None,
            "last_processed_frame_id": (
                self._latest_packet.frame_id if self._latest_packet else None
            ),
            "last_error": self._last_error,
        }
