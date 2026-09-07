import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import settings


@dataclass(frozen=True)
class ROIConfig:
    camera_id: str
    roi_points_normalized: tuple[tuple[float, float], ...]
    zone_width_m: float
    zone_height_m: float
    grid_rows: int
    grid_cols: int
    local_window_width_m: float
    local_window_height_m: float
    local_window_step_m: float
    relaxed_max: float
    danger_min: float

    @property
    def zone_area_m2(self) -> float:
        return self.zone_width_m * self.zone_height_m

    @classmethod
    def default(cls, camera_id: str | None = None) -> "ROIConfig":
        return cls(
            camera_id=camera_id or settings.CAMERA_ID,
            roi_points_normalized=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            zone_width_m=settings.ZONE_WIDTH_M,
            zone_height_m=settings.ZONE_HEIGHT_M,
            grid_rows=settings.DENSITY_GRID_ROWS,
            grid_cols=settings.DENSITY_GRID_COLS,
            local_window_width_m=settings.LOCAL_WINDOW_WIDTH_M,
            local_window_height_m=settings.LOCAL_WINDOW_HEIGHT_M,
            local_window_step_m=settings.LOCAL_WINDOW_STEP_M,
            relaxed_max=settings.DENSITY_RELAXED_MAX,
            danger_min=settings.DENSITY_DANGER_MIN,
        )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ROIConfig":
        points = tuple(tuple(map(float, point)) for point in value["roi_points_normalized"])
        config = cls(
            camera_id=str(value["camera_id"]),
            roi_points_normalized=points,
            zone_width_m=float(value["zone_width_m"]),
            zone_height_m=float(value["zone_height_m"]),
            grid_rows=int(value.get("grid_rows", 3)),
            grid_cols=int(value.get("grid_cols", 3)),
            local_window_width_m=float(value.get("local_window_width_m", 1.0)),
            local_window_height_m=float(value.get("local_window_height_m", 1.0)),
            local_window_step_m=float(value.get("local_window_step_m", 0.1)),
            relaxed_max=float(value.get("relaxed_max", 2.0)),
            danger_min=float(value.get("danger_min", 5.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.roi_points_normalized) != 4:
            raise ValueError("ROI requires four points ordered TL, TR, BR, BL")
        if any(len(point) != 2 for point in self.roi_points_normalized):
            raise ValueError("Each ROI point must contain x and y")
        if any(coordinate < 0 or coordinate > 1 for point in self.roi_points_normalized for coordinate in point):
            raise ValueError("Normalized ROI coordinates must be between 0 and 1")
        polygon = np.asarray(self.roi_points_normalized, dtype=np.float32)
        if abs(cv2.contourArea(polygon)) < 0.0001:
            raise ValueError("ROI polygon area is too small")
        if self.zone_width_m <= 0 or self.zone_height_m <= 0:
            raise ValueError("Zone dimensions must be positive")
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("Grid dimensions must be positive")
        if self.local_window_width_m <= 0 or self.local_window_height_m <= 0:
            raise ValueError("Local window dimensions must be positive")
        if self.local_window_width_m > self.zone_width_m or self.local_window_height_m > self.zone_height_m:
            raise ValueError("Local window must fit inside the zone")
        if self.local_window_step_m <= 0:
            raise ValueError("Local window step must be positive")
        horizontal_steps = self.zone_width_m / self.local_window_step_m
        vertical_steps = self.zone_height_m / self.local_window_step_m
        if horizontal_steps * vertical_steps > 10000:
            raise ValueError("Local window step creates too many scan positions")
        if self.relaxed_max < 0 or self.danger_min <= self.relaxed_max:
            raise ValueError("Density thresholds are invalid")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["roi_points_normalized"] = [list(point) for point in self.roi_points_normalized]
        data["zone_area_m2"] = self.zone_area_m2
        return data


class DensityAnalyzer:
    """Map person footpoints into a configurable plane and calculate density."""

    def __init__(
        self,
        config_path: str,
        camera_id: str | None = None,
        enable_local_peak: bool | None = None,
    ):
        self.config_path = Path(config_path)
        self.camera_id = camera_id
        self.enable_local_peak = (
            settings.ENABLE_LOCAL_PEAK_DENSITY
            if enable_local_peak is None
            else bool(enable_local_peak)
        )
        self._lock = threading.RLock()
        self._config = self._load()

    def _load(self) -> ROIConfig:
        if not self.config_path.is_file():
            config = ROIConfig.default(self.camera_id)
            config.validate()
            return config
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        return ROIConfig.from_mapping(data)

    def get_config(self) -> dict[str, Any]:
        with self._lock:
            return self._config.to_dict()

    def update_config(self, value: dict[str, Any]) -> dict[str, Any]:
        config = ROIConfig.from_mapping(value)
        payload = config.to_dict()
        payload.pop("zone_area_m2", None)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.config_path)
        with self._lock:
            self._config = config
        return config.to_dict()

    @staticmethod
    def _density_level(density: float, config: ROIConfig) -> str:
        if density <= config.relaxed_max:
            return "Relaxed"
        if density < config.danger_min:
            return "Caution"
        return "Danger"

    @staticmethod
    def _axis_positions(maximum: float, window: float, step: float) -> list[float]:
        last = max(0.0, maximum - window)
        values = [
            float(value)
            for value in np.arange(0.0, last + step * 0.5, step, dtype=float)
        ]
        if not values or abs(values[-1] - last) > 1e-6:
            values.append(float(last))
        return values

    def _sliding_window_peak(
        self, points_m: list[tuple[float, float]], config: ROIConfig
    ) -> dict[str, Any]:
        width = config.local_window_width_m
        height = config.local_window_height_m
        area = width * height
        best_count = 0
        best_origin = (0.0, 0.0)
        for origin_y in self._axis_positions(config.zone_height_m, height, config.local_window_step_m):
            for origin_x in self._axis_positions(config.zone_width_m, width, config.local_window_step_m):
                count = int(sum(
                    origin_x <= x <= origin_x + width
                    and origin_y <= y <= origin_y + height
                    for x, y in points_m
                ))
                if count > best_count:
                    best_count = count
                    best_origin = (float(origin_x), float(origin_y))
        density = float(best_count / area)
        return {
            "count": int(best_count),
            "density_people_per_m2": density,
            "level": self._density_level(density, config),
            "origin_m": [float(best_origin[0]), float(best_origin[1])],
            "width_m": float(width),
            "height_m": float(height),
            "area_m2": float(area),
        }

    def analyze(
        self, detections: list[dict], frame_width: int, frame_height: int
    ) -> dict[str, Any]:
        with self._lock:
            config = self._config

        scale = np.asarray([frame_width - 1, frame_height - 1], dtype=np.float32)
        roi_pixels = np.asarray(config.roi_points_normalized, dtype=np.float32) * scale
        destination = np.asarray(
            [
                [0.0, 0.0],
                [config.zone_width_m, 0.0],
                [config.zone_width_m, config.zone_height_m],
                [0.0, config.zone_height_m],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(roi_pixels, destination)
        if not np.isfinite(homography).all() or abs(np.linalg.det(homography)) < 1e-12:
            raise ValueError("ROI points cannot produce a valid homography")
        inverse_homography = np.linalg.inv(homography)
        points_m: list[tuple[float, float]] = []

        for detection in detections:
            x1, _y1, x2, y2 = detection["box"]
            footpoint = (float((x1 + x2) / 2), float(y2))
            inside = cv2.pointPolygonTest(roi_pixels, footpoint, False) >= 0
            detection["footpoint"] = list(footpoint)
            detection["in_roi"] = bool(inside)
            if not inside:
                continue
            transformed = cv2.perspectiveTransform(
                np.asarray([[footpoint]], dtype=np.float32), homography
            )[0, 0]
            point_m = (float(transformed[0]), float(transformed[1]))
            detection["bird_eye_m"] = list(point_m)
            points_m.append(point_m)

        cell_width = config.zone_width_m / config.grid_cols
        cell_height = config.zone_height_m / config.grid_rows
        cell_area = cell_width * cell_height
        cells = [0] * (config.grid_rows * config.grid_cols)
        for x, y in points_m:
            column = min(config.grid_cols - 1, max(0, int(x / cell_width)))
            row = min(config.grid_rows - 1, max(0, int(y / cell_height)))
            cells[row * config.grid_cols + column] += 1

        grid = []
        for index, count in enumerate(cells):
            count = int(count)
            density = float(count / cell_area)
            row = index // config.grid_cols
            column = index % config.grid_cols
            metric_polygon = np.asarray(
                [[
                    [column * cell_width, row * cell_height],
                    [(column + 1) * cell_width, row * cell_height],
                    [(column + 1) * cell_width, (row + 1) * cell_height],
                    [column * cell_width, (row + 1) * cell_height],
                ]],
                dtype=np.float32,
            )
            pixel_polygon = cv2.perspectiveTransform(
                metric_polygon, inverse_homography
            )[0]
            grid.append(
                {
                    "id": index + 1,
                    "count": count,
                    "area_m2": float(cell_area),
                    "density_people_per_m2": density,
                    "level": self._density_level(density, config),
                    "polygon": pixel_polygon.tolist(),
                }
            )

        roi_count = int(len(points_m))
        density = float(roi_count / config.zone_area_m2)
        max_grid_density = float(
            max((cell["density_people_per_m2"] for cell in grid), default=0.0)
        )
        peak = None
        applied_peak_density = max_grid_density
        density_mode = "fixed-grid"
        if self.enable_local_peak:
            peak = self._sliding_window_peak(points_m, config)
            peak_x, peak_y = peak["origin_m"]
            peak_metric_polygon = np.asarray(
                [[
                    [peak_x, peak_y],
                    [peak_x + peak["width_m"], peak_y],
                    [peak_x + peak["width_m"], peak_y + peak["height_m"]],
                    [peak_x, peak_y + peak["height_m"]],
                ]],
                dtype=np.float32,
            )
            peak["polygon"] = cv2.perspectiveTransform(
                peak_metric_polygon, inverse_homography
            )[0].tolist()
            applied_peak_density = float(peak["density_people_per_m2"])
            density_mode = "sliding-local-window"
        return {
            "roi_count": roi_count,
            "density_people_per_m2": density,
            "density_level": self._density_level(density, config),
            "risk_level": self._density_level(applied_peak_density, config),
            "zone_width_m": config.zone_width_m,
            "zone_height_m": config.zone_height_m,
            "zone_area_m2": config.zone_area_m2,
            "roi_points": roi_pixels.tolist(),
            "grid_rows": config.grid_rows,
            "grid_cols": config.grid_cols,
            "grid": grid,
            "max_grid_density_people_per_m2": max_grid_density,
            "applied_peak_density_people_per_m2": applied_peak_density,
            "density_mode": density_mode,
            "local_peak_enabled": bool(self.enable_local_peak),
            "local_peak": peak,
            "thresholds": {
                "relaxed_max": config.relaxed_max,
                "danger_min": config.danger_min,
            },
            "formula": "roi_count / zone_area_m2",
        }
