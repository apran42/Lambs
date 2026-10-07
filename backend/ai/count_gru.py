"""NumPy-only count GRU inference for Jetson.

File inputs use source-video seconds; a live camera uses wall-clock seconds.
The caller must not mix these time bases.
"""

from bisect import bisect_right
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time

import numpy as np


def select_video_horizon(target_wall_seconds, source_fps, capture_fps,
                         available_horizons=(30, 40, 60), tolerance_seconds=5.0):
    """Choose a source-time model only when it closely matches wall time.

    A file advancing at 15 captured frames/s from a 30-fps source advances
    30 source seconds in roughly 60 wall seconds. If no available checkpoint
    matches closely, return None so the caller can use a safer baseline.
    """
    values = (target_wall_seconds, source_fps, capture_fps)
    if any(not np.isfinite(value) or value <= 0 for value in values):
        return None
    if tolerance_seconds < 0:
        raise ValueError("Tolerance must be non-negative")
    horizons = tuple(available_horizons)
    if not horizons:
        return None
    model_horizon = min(
        horizons,
        key=lambda horizon: abs(horizon * source_fps / capture_fps
                                - target_wall_seconds),
    )
    effective_wall_seconds = model_horizon * source_fps / capture_fps
    if abs(effective_wall_seconds - target_wall_seconds) > tolerance_seconds:
        return None
    return int(model_horizon), float(effective_wall_seconds)


def build_history(samples, origin_seconds, history_seconds, stride_seconds,
                  max_sample_age_seconds=10.0):
    """Return training-compatible [count / 10, sample age / 10] entries.

    ``samples`` contains (time_seconds, detected_count) pairs in time order.
    None means that the observed history is too short or has a large gap.
    """
    if history_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("History and stride must be positive")
    if history_seconds % stride_seconds:
        raise ValueError("History must be divisible by stride")
    if not samples:
        return None
    times = [float(sample[0]) for sample in samples]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Sample times must be strictly increasing")
    origin = float(origin_seconds)
    history = []
    for offset in range(int(history_seconds), -1, -int(stride_seconds)):
        query = origin - offset
        index = bisect_right(times, query) - 1
        if index < 0 or query - times[index] > max_sample_age_seconds:
            return None
        count = float(samples[index][1])
        if not np.isfinite(count) or count < 0:
            raise ValueError("Counts must be finite and non-negative")
        history.append([count / 10.0, (query - times[index]) / 10.0])
    return np.asarray(history, dtype=np.float32)


class NumpyCountGRU:
    """Evaluate the single-layer GRU produced by train_one_minute_gru.py."""

    def __init__(self, weights_path):
        with np.load(weights_path, allow_pickle=False) as weights:
            self.ih = weights["gru.weight_ih_l0"].astype(np.float32)
            self.hh = weights["gru.weight_hh_l0"].astype(np.float32)
            self.bi = weights["gru.bias_ih_l0"].astype(np.float32)
            self.bh = weights["gru.bias_hh_l0"].astype(np.float32)
            self.head_weight = weights["head.weight"].astype(np.float32)
            self.head_bias = weights["head.bias"].astype(np.float32)
        self.hidden_size = self.hh.shape[1]
        expected = self.hidden_size * 3
        if (self.ih.shape != (expected, 2)
                or self.hh.shape != (expected, self.hidden_size)
                or self.bi.shape != (expected,)
                or self.bh.shape != (expected,)
                or self.head_weight.shape != (1, self.hidden_size)
                or self.head_bias.shape != (1,)):
            raise ValueError("Unsupported GRU weight shapes")

    @staticmethod
    def _sigmoid(values):
        return 1.0 / (1.0 + np.exp(-values))

    def predict(self, history, current_count):
        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 2 or not len(history):
            raise ValueError("History must have shape (steps, 2)")
        if not np.all(np.isfinite(history)) or not np.isfinite(current_count):
            raise ValueError("GRU inputs must be finite")
        hidden = np.zeros(self.hidden_size, dtype=np.float32)
        size = self.hidden_size
        for entry in history:
            from_input = np.dot(self.ih, entry) + self.bi
            from_hidden = np.dot(self.hh, hidden) + self.bh
            reset = self._sigmoid(from_input[:size] + from_hidden[:size])
            update = self._sigmoid(
                from_input[size:2 * size] + from_hidden[size:2 * size]
            )
            candidate = np.tanh(
                from_input[2 * size:] + reset * from_hidden[2 * size:]
            )
            hidden = (1.0 - update) * candidate + update * hidden
        correction = (np.dot(self.head_weight, hidden)[0]
                      + self.head_bias[0]) * 10.0
        return float(current_count + correction)


class CountGRUForecaster:
    """Rolling one-minute count forecast with explicit video-time handling."""

    def __init__(self, model_dir, target_wall_seconds=60.0):
        self.target_wall_seconds = float(target_wall_seconds)
        self.models = {}
        self.history_seconds = None
        self.stride_seconds = None
        for directory in Path(model_dir).glob("video*_runtime"):
            metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
            if metadata.get("format") != "lambs-count-gru-v1":
                raise ValueError("Unsupported count GRU metadata")
            horizon = int(metadata["horizon_video_seconds"])
            history = int(metadata["history_seconds"])
            stride = int(metadata["stride_seconds"])
            if self.history_seconds is not None and (history, stride) != (
                    self.history_seconds, self.stride_seconds):
                raise ValueError("GRU models have incompatible history settings")
            self.models[horizon] = NumpyCountGRU(directory / "weights.npz")
            self.history_seconds = history
            self.stride_seconds = stride
        if not self.models:
            raise ValueError("No exported count GRU models found")
        self._samples = deque()
        self._latest = None
        self._lock = threading.RLock()

    def reset(self):
        with self._lock:
            self._samples.clear()
            self._latest = None

    def latest(self):
        with self._lock:
            return dict(self._latest) if self._latest is not None else None

    def update(self, count, local_peak_density, zone_area_m2, relaxed_max,
               danger_min, measured_at=None, source_seconds=None,
               source_fps=None, capture_fps=None, video_source=False):
        now = float(measured_at if measured_at is not None else time.monotonic())
        source_time = float(source_seconds if source_seconds is not None else now)
        is_file = bool(video_source or source_seconds is not None)
        if zone_area_m2 <= 0:
            raise ValueError("Zone area must be positive")
        with self._lock:
            if self._samples and source_time < self._samples[-1][0]:
                self._samples.clear()  # File playback loop or seek.
            if self._samples and source_time == self._samples[-1][0]:
                self._samples.pop()
            self._samples.append((source_time, int(count)))
            cutoff = source_time - self.history_seconds - 10.0
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

            if is_file:
                selection = select_video_horizon(
                    self.target_wall_seconds, source_fps, capture_fps,
                    self.models.keys(),
                )
            else:
                horizon = int(self.target_wall_seconds)
                selection = ((horizon, self.target_wall_seconds)
                             if horizon in self.models else None)
            history = build_history(
                self._samples, source_time, self.history_seconds,
                self.stride_seconds,
            )
            ready = selection is not None and history is not None
            if ready:
                model_horizon, effective_wall_seconds = selection
                raw_count = self.models[model_horizon].predict(history, count)
                predicted_count = max(0, int(round(raw_count)))
                method = "gru-count-video" if is_file else "gru-count-live"
            else:
                model_horizon = None
                effective_wall_seconds = None
                predicted_count = max(0, int(count))
                method = ("persistence-timebase-mismatch" if selection is None
                          else "persistence-warmup")

            predicted_density = float(predicted_count / zone_area_m2)
            predicted_peak = max(predicted_density, float(local_peak_density))
            if predicted_peak <= relaxed_max:
                risk = "Relaxed"
            elif predicted_peak < danger_min:
                risk = "Caution"
            else:
                risk = "Danger"
            generated_at = datetime.now(timezone.utc)
            self._latest = {
                "horizon_seconds": int(self.target_wall_seconds),
                "generated_at": generated_at.isoformat(),
                "forecast_for": (generated_at + timedelta(
                    seconds=self.target_wall_seconds)).isoformat(),
                "predicted_roi_count": predicted_count,
                "predicted_density_people_per_m2": predicted_density,
                "predicted_local_peak_density_people_per_m2": predicted_peak,
                "predicted_risk_level": risk,
                "method": method,
                "ready": bool(ready),
                "confidence": 0.0,  # A calibrated coverage interval is unavailable.
                "confidence_label": "low",
                "history_span_seconds": float(
                    self._samples[-1][0] - self._samples[0][0]),
                "sample_count": int(len(self._samples)),
                "model_video_horizon_seconds": model_horizon,
                "effective_wall_horizon_seconds": effective_wall_seconds,
            }
            return dict(self._latest)
