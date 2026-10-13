from __future__ import annotations

import threading
import time
import json
import math
from bisect import bisect_left
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


class CrowdForecaster:
    """One-minute forecast with persistence as the safe default."""

    def __init__(
        self,
        horizon_seconds: float = 60.0,
        history_seconds: float = 120.0,
        min_trend_span_seconds: float = 30.0,
        bucket_seconds: float = 5.0,
        mode: str = "persistence",
        reviewed_demo_path: str | None = None,
    ) -> None:
        if mode not in ("persistence", "trend", "reviewed-demo"):
            raise ValueError("Forecast mode must be 'persistence', 'trend', or 'reviewed-demo'")
        self.horizon_seconds = float(horizon_seconds)
        self.history_seconds = float(history_seconds)
        self.min_trend_span_seconds = float(min_trend_span_seconds)
        self.bucket_seconds = float(bucket_seconds)
        self.mode = mode
        self._reviewed_videos = {}
        if mode == "reviewed-demo":
            if not reviewed_demo_path:
                raise ValueError("FORECAST_REVIEWED_DEMO_PATH is required for reviewed-demo mode")
            payload = json.loads(Path(reviewed_demo_path).read_text(encoding="utf-8"))
            if int(payload.get("horizon_seconds", -1)) != int(self.horizon_seconds):
                raise ValueError("Reviewed demo data must match forecast horizon")
            for video_id, video in payload.get("videos", {}).items():
                samples = sorted((float(t), int(n)) for t, n in video["samples"])
                if not samples or float(video["duration_seconds"]) <= 0:
                    raise ValueError("Reviewed demo video needs a duration and samples")
                self._reviewed_videos[str(video_id)] = (
                    float(video["duration_seconds"]),
                    [t for t, _ in samples],
                    [n for _, n in samples],
                )
        self._samples: deque[tuple[float, int, float, float, float]] = deque()
        self._latest: dict | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _risk_level(density: float, relaxed_max: float, danger_min: float) -> str:
        if density <= relaxed_max:
            return "Relaxed"
        if density < danger_min:
            return "Caution"
        return "Danger"

    def _bucketed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        buckets: dict[int, list[tuple[int, float]]] = {}
        for measured_at, count, peak_density, _area, _danger in self._samples:
            bucket = int(measured_at // self.bucket_seconds)
            buckets.setdefault(bucket, []).append((count, peak_density))

        times: list[float] = []
        counts: list[float] = []
        peaks: list[float] = []
        for bucket, values in sorted(buckets.items()):
            times.append((bucket + 0.5) * self.bucket_seconds)
            counts.append(float(np.median([value[0] for value in values])))
            peaks.append(float(np.median([value[1] for value in values])))
        return np.asarray(times), np.asarray(counts), np.asarray(peaks)

    @staticmethod
    def _linear_projection(
        times: np.ndarray,
        values: np.ndarray,
        horizon_seconds: float,
        damping: float,
    ) -> tuple[float, float]:
        centred_times = times - times[-1]
        slope, intercept = np.polyfit(centred_times, values, 1)
        fitted = slope * centred_times + intercept
        residual = float(np.sum((values - fitted) ** 2))
        total = float(np.sum((values - np.mean(values)) ** 2))
        r_squared = 1.0 if total <= 1e-12 else max(0.0, 1.0 - residual / total)
        projected = float(intercept + slope * horizon_seconds * damping)
        return projected, r_squared

    def _reviewed_demo_count(
        self, source_video_id, video_time_seconds, source_duration_seconds,
        video_seconds_per_wall_second,
    ):
        video = self._reviewed_videos.get(str(source_video_id))
        if video is None or video_time_seconds is None:
            return None
        duration, times, counts = video
        position = float(video_time_seconds)
        if not math.isfinite(position) or position < 0:
            return None
        if source_duration_seconds is not None and (
            not math.isfinite(float(source_duration_seconds))
            or abs(float(source_duration_seconds) - duration) > 2.0
        ):
            return None
        speed = float(video_seconds_per_wall_second)
        if not math.isfinite(speed) or speed <= 0:
            return None
        target = (position + self.horizon_seconds * speed) % duration
        index = bisect_left(times, target)
        if index < len(times) and abs(times[index] - target) < 1e-6:
            return counts[index], target
        if 0 < index < len(times) and times[index] - times[index - 1] <= 10.0:
            before, after = times[index - 1], times[index]
            weight = (target - before) / (after - before)
            return round(counts[index - 1] * (1.0 - weight) + counts[index] * weight), target
        nearest = min(
            (max(0, index - 1), min(index, len(times) - 1)),
            key=lambda candidate: abs(times[candidate] - target),
        )
        if abs(times[nearest] - target) <= 5.0:
            return counts[nearest], target
        return None

    def update(
        self,
        count: int,
        local_peak_density: float,
        zone_area_m2: float,
        relaxed_max: float,
        danger_min: float,
        measured_at: float | None = None,
        source_video_id: str | int | None = None,
        video_time_seconds: float | None = None,
        source_duration_seconds: float | None = None,
        video_seconds_per_wall_second: float = 1.0,
    ) -> dict:
        now = float(measured_at if measured_at is not None else time.monotonic())
        with self._lock:
            self._samples.append(
                (
                    now,
                    int(count),
                    float(local_peak_density),
                    float(zone_area_m2),
                    float(danger_min),
                )
            )
            cutoff = now - self.history_seconds
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

            times, counts, peaks = self._bucketed()
            span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
            trend_ready = len(times) >= 2 and span >= self.min_trend_span_seconds
            trend_fit_score = None

            demo_result = (
                self._reviewed_demo_count(
                    source_video_id, video_time_seconds, source_duration_seconds,
                    video_seconds_per_wall_second,
                )
                if self.mode == "reviewed-demo" else None
            )
            if demo_result is not None:
                projected_count = float(demo_result[0])
                projected_peak = float(local_peak_density)
                method = "reviewed-video-demo"
                ready = True
            elif self.mode in ("persistence", "reviewed-demo"):
                # Keep the most recently observed count. A straight-line trend
                # performed substantially worse on the reviewed videos.
                projected_count = float(count)
                projected_peak = float(local_peak_density)
                method = "persistence-baseline"
                ready = True
            elif trend_ready:
                # Short histories must not be extrapolated at full strength. The
                # trend gradually receives full weight once the observed history
                # is at least as long as the forecast horizon.
                damping = min(1.0, span / self.horizon_seconds)
                projected_count, count_fit = self._linear_projection(
                    times, counts, self.horizon_seconds, damping
                )
                projected_peak, peak_fit = self._linear_projection(
                    times, peaks, self.horizon_seconds, damping
                )
                method = "damped-linear-trend"
                ready = True
                trend_fit_score = min(1.0, span / (2.0 * self.horizon_seconds)) * (
                    0.5 + 0.5 * min(count_fit, peak_fit)
                )
            else:
                projected_count = float(np.median(counts[-3:]))
                projected_peak = float(np.median(peaks[-3:]))
                method = "persistence-warmup"
                ready = False

            predicted_count = max(0, int(round(projected_count)))
            predicted_density = float(predicted_count / zone_area_m2)
            predicted_peak = max(predicted_density, float(max(0.0, projected_peak)))
            generated_at = datetime.now(timezone.utc)
            self._latest = {
                "horizon_seconds": int(self.horizon_seconds),
                "generated_at": generated_at.isoformat(),
                "forecast_for": (
                    generated_at + timedelta(seconds=self.horizon_seconds)
                ).isoformat(),
                "predicted_roi_count": int(predicted_count),
                "predicted_density_people_per_m2": predicted_density,
                "predicted_local_peak_density_people_per_m2": predicted_peak,
                "predicted_risk_level": self._risk_level(
                    predicted_peak, relaxed_max, danger_min
                ),
                "method": method,
                "ready": bool(ready),
                "confidence": None,
                "confidence_label": "unvalidated",
                "trend_fit_score": trend_fit_score,
                "assumption": (
                    "same-reviewed-video-timeline" if demo_result is not None
                    else "current-count-unchanged" if method == "persistence-baseline"
                    else None
                ),
                "reference_video_time_seconds": demo_result[1] if demo_result else None,
                "history_span_seconds": span,
                "sample_count": int(len(self._samples)),
            }
            return dict(self._latest)

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None
