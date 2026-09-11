from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from config import BACKEND_DIR, settings


VideoSource = Union[str, int]


@dataclass(frozen=True)
class CameraDefinition:
    camera_id: str
    name: str
    source: VideoSource
    facility: str
    location: str
    target_fps: float
    roi_config_path: str
    calibration_path: str | None = None


def _resolve_path(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def _resolve_source(value, base: Path) -> VideoSource:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return _resolve_path(text, base)


def load_camera_definitions(config_path: str | None = None) -> list[CameraDefinition]:
    path = Path(config_path or settings.CAMERAS_CONFIG_PATH)
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent if path.is_absolute() else BACKEND_DIR
    cameras = []
    seen_ids = set()
    for item in data.get("cameras", []):
        if not item.get("enabled", True):
            continue
        camera_id = str(item["id"])
        if camera_id in seen_ids:
            raise ValueError(f"Duplicate camera id: {camera_id}")
        seen_ids.add(camera_id)
        target_fps = float(item.get("target_fps", settings.CAPTURE_FPS))
        if target_fps < 20:
            raise ValueError(f"{camera_id} target_fps must be at least 20")
        calibration_value = item.get("calibration_path")
        cameras.append(
            CameraDefinition(
                camera_id=camera_id,
                name=str(item.get("name", camera_id)),
                source=_resolve_source(item["source"], base),
                facility=str(item.get("facility", settings.FACILITY_NAME)),
                location=str(item.get("location", camera_id)),
                target_fps=target_fps,
                roi_config_path=_resolve_path(
                    item.get("roi_config_path", f"camera_configs/{camera_id}-roi.json"),
                    base,
                ),
                calibration_path=(
                    _resolve_path(calibration_value, base)
                    if calibration_value
                    else None
                ),
            )
        )
    if not cameras:
        raise ValueError("At least one enabled camera is required")
    return cameras
