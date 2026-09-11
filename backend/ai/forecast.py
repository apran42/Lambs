from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np


class CrowdForecaster:
    """Conservative online baseline for a five-minute crowd forecast."""

    def __init__(
        self,
        horizon_seconds: float = 300.0,
        history_seconds: float = 600.0,
        min_trend_span_seconds: float = 60.0,
        bucket_seconds: float = 5.0,
    ) -> None:
        self.horizon_seconds = float(horizon_seconds)
        self.history_seconds = float(history_seconds)
        self.min_trend_span_seconds = float(min_trend_span_seconds)
        self.bucket_seconds = float(bucket_seconds)
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

    def update(
        self,
        count: int,
        local_peak_density: float,
        zone_area_m2: float,
        relaxed_max: float,
        danger_min: float,
        measured_at: float | None = None,
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
            ready = len(times) >= 2 and span >= self.min_trend_span_seconds

            if ready:
                # Short histories must not be extrapolated at full strength over
                # five minutes. The trend gradually receives full weight once the
                # observed history is at least as long as the forecast horizon.
                damping = min(1.0, span / self.horizon_seconds)
                projected_count, count_fit = self._linear_projection(
                    times, counts, self.horizon_seconds, damping
                )
                projected_peak, peak_fit = self._linear_projection(
                    times, peaks, self.horizon_seconds, damping
                )
                method = "damped-linear-trend"
                confidence = min(1.0, span / (2.0 * self.horizon_seconds)) * (
                    0.5 + 0.5 * min(count_fit, peak_fit)
                )
            else:
                projected_count = float(np.median(counts[-3:]))
                projected_peak = float(np.median(peaks[-3:]))
                method = "persistence-warmup"
                confidence = min(0.2, span / max(1.0, self.min_trend_span_seconds) * 0.2)

            predicted_count = max(0, int(round(projected_count)))
            predicted_density = float(predicted_count / zone_area_m2)
            predicted_peak = max(predicted_density, float(max(0.0, projected_peak)))
            generated_at = datetime.now(timezone.utc)
            confidence_label = (
                "high" if confidence >= 0.7 else "medium" if confidence >= 0.35 else "low"
            )
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
                "confidence": float(confidence),
                "confidence_label": confidence_label,
                "history_span_seconds": span,
                "sample_count": int(len(self._samples)),
            }
            return dict(self._latest)

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest is not None else None
