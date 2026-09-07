import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> str:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    return str(path)


def _env_video_source(name: str, default: Path) -> str:
    value = os.getenv(name)
    if value and value.strip().isdigit():
        return value.strip()
    return _env_path(name, default)


@dataclass(frozen=True)
class Settings:
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "Lambs")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "crowd_monitor")

    YOLO_MODEL_PATH: str = _env_path("YOLO_MODEL_PATH", PROJECT_DIR / "yolov8n.pt")
    VIDEO_PATH: str = _env_video_source(
        "VIDEO_PATH", PROJECT_DIR / "data" / "sample_data (1).mp4"
    )
    VIDEO_LOOP: bool = _env_bool("VIDEO_LOOP", True)
    FRAME_WIDTH: int = _env_int("FRAME_WIDTH", 640)
    FRAME_HEIGHT: int = _env_int("FRAME_HEIGHT", 480)
    CAPTURE_FPS: float = _env_float("CAPTURE_FPS", 20.0)
    INFERENCE_BATCH_SIZE: int = _env_int("INFERENCE_BATCH_SIZE", 2)
    CPU_INFERENCE_THREADS: int = _env_int("CPU_INFERENCE_THREADS", 2)
    CPU_INTEROP_THREADS: int = _env_int("CPU_INTEROP_THREADS", 1)
    OPENCV_THREADS: int = _env_int("OPENCV_THREADS", 1)
    CAMERAS_CONFIG_PATH: str = _env_path(
        "CAMERAS_CONFIG_PATH", BACKEND_DIR / "cameras.json"
    )

    YOLO_CONFIDENCE: float = _env_float("YOLO_CONFIDENCE", 0.35)
    YOLO_IOU: float = _env_float("YOLO_IOU", 0.45)
    YOLO_IMAGE_SIZE: int = _env_int("YOLO_IMAGE_SIZE", 640)
    YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "")
    YOLO_TRACKER: str = os.getenv("YOLO_TRACKER", "bytetrack.yaml")

    JPEG_QUALITY: int = _env_int("JPEG_QUALITY", 80)
    DB_WRITE_INTERVAL_SECONDS: float = _env_float(
        "DB_WRITE_INTERVAL_SECONDS", 1.0
    )
    CROWDED_COUNT_THRESHOLD: int = _env_int("CROWDED_COUNT_THRESHOLD", 10)
    CALIBRATION_PATH: str = _env_path(
        "CALIBRATION_PATH", BACKEND_DIR / "calibration_data.json"
    )
    CALIBRATION_IMAGES_PATH: str = _env_path(
        "CALIBRATION_IMAGES_PATH", BACKEND_DIR / "calib_images"
    )
    ROI_CONFIG_PATH: str = _env_path(
        "ROI_CONFIG_PATH", BACKEND_DIR / "roi_config.json"
    )
    ZONE_WIDTH_M: float = _env_float("ZONE_WIDTH_M", 3.0)
    ZONE_HEIGHT_M: float = _env_float("ZONE_HEIGHT_M", 3.0)
    DENSITY_RELAXED_MAX: float = _env_float("DENSITY_RELAXED_MAX", 2.0)
    DENSITY_DANGER_MIN: float = _env_float("DENSITY_DANGER_MIN", 5.0)
    DENSITY_GRID_ROWS: int = _env_int("DENSITY_GRID_ROWS", 3)
    DENSITY_GRID_COLS: int = _env_int("DENSITY_GRID_COLS", 3)
    LOCAL_WINDOW_WIDTH_M: float = _env_float("LOCAL_WINDOW_WIDTH_M", 1.0)
    LOCAL_WINDOW_HEIGHT_M: float = _env_float("LOCAL_WINDOW_HEIGHT_M", 1.0)
    LOCAL_WINDOW_STEP_M: float = _env_float("LOCAL_WINDOW_STEP_M", 0.1)
    ENABLE_LOCAL_PEAK_DENSITY: bool = _env_bool("ENABLE_LOCAL_PEAK_DENSITY", False)
    FORECAST_HORIZON_SECONDS: float = _env_float("FORECAST_HORIZON_SECONDS", 300.0)
    FORECAST_HISTORY_SECONDS: float = _env_float("FORECAST_HISTORY_SECONDS", 600.0)
    FORECAST_MIN_TREND_SPAN_SECONDS: float = _env_float(
        "FORECAST_MIN_TREND_SPAN_SECONDS", 60.0
    )
    FORECAST_BUCKET_SECONDS: float = _env_float("FORECAST_BUCKET_SECONDS", 5.0)

    CAMERA_ID: str = os.getenv("CAMERA_ID", "cam-01")
    FACILITY_NAME: str = os.getenv("FACILITY_NAME", "model-zone")
    CAMERA_LOCATION: str = os.getenv("CAMERA_LOCATION", "zone-a")
    CORS_ORIGINS: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        ).split(",")
        if origin.strip()
    )


settings = Settings()
