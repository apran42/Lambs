from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def calibration_matrix(calibration: dict) -> np.ndarray:
    points = calibration.get("image_points", [])
    width_m = as_float(calibration.get("width_m"))
    height_m = as_float(calibration.get("height_m"))
    if len(points) != 4 or width_m <= 0 or height_m <= 0:
        raise ValueError("Calibration requires four image points and positive real dimensions")
    source = np.asarray(points, dtype=np.float32)
    target = np.asarray([[0, 0], [width_m, 0], [width_m, height_m], [0, height_m]], dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)


def ground_point(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = np.asarray([[[x, y]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, matrix)[0][0]
    return float(transformed[0]), float(transformed[1])


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def create_app(dataset_dir: Path) -> FastAPI:
    dataset_dir = dataset_dir.resolve()
    frame_path = dataset_dir / "frame_review.csv"
    detection_path = dataset_dir / "auto_detection_review.csv"
    annotation_path = dataset_dir / "manual_annotations.csv"
    timeseries_path = dataset_dir / "crowd_timeseries_review.csv"
    calibration_path = dataset_dir / "calibration.json"
    export_root = dataset_dir / "exports"
    image_dir = (dataset_dir / "images").resolve()
    html_path = Path(__file__).with_name("review_ui.html")
    metadata_path = dataset_dir / "metadata.json"

    required = [frame_path, detection_path, annotation_path, timeseries_path, image_dir, html_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing review dataset files: {missing}")

    app = FastAPI(title="Lambs training-data reviewer")
    lock = threading.RLock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/images/{image_name}")
    def image(image_name: str) -> FileResponse:
        candidate = (image_dir / image_name).resolve()
        if candidate.parent != image_dir or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(candidate, media_type="image/jpeg")

    @app.get("/api/frames")
    def frames() -> dict:
        with lock:
            _, rows = read_csv(frame_path)
        items = [
            {
                "sample_id": row["sample_id"],
                "video_time_seconds": as_float(row.get("video_time_seconds")),
                "auto_detected_count": as_int(row.get("auto_detected_count")),
                "manual_ground_truth_count": row.get("manual_ground_truth_count", ""),
                "manual_review_status": row.get("manual_review_status", ""),
                "manual_box_review_status": row.get("manual_box_review_status", ""),
            }
            for row in rows
        ]
        return {
            "frames": items,
            "total": len(items),
            "count_reviewed": sum(
                item["manual_ground_truth_count"] != ""
                or item["manual_review_status"] == "rejected"
                for item in items
            ),
            "box_reviewed": sum(item["manual_box_review_status"] == "completed" for item in items),
            "reviewed": sum(
                (item["manual_ground_truth_count"] != "" or item["manual_review_status"] == "rejected")
                and item["manual_box_review_status"] == "completed"
                for item in items
            ),
        }

    @app.get("/api/frames/{sample_id}")
    def frame(sample_id: str) -> dict:
        with lock:
            _, frame_rows = read_csv(frame_path)
            _, detection_rows = read_csv(detection_path)
            _, annotation_rows = read_csv(annotation_path)
            _, timeseries_rows = read_csv(timeseries_path)
        frame_row = next((row for row in frame_rows if row["sample_id"] == sample_id), None)
        if frame_row is None:
            raise HTTPException(status_code=404, detail="Frame not found")

        boxes = []
        for row in detection_rows:
            if row["sample_id"] != sample_id:
                continue
            action = row.get("manual_action_keep_delete_edit", "")
            edited = action == "edit" and row.get("manual_x1_px_if_edited", "") != ""
            prefix = "manual" if edited else "auto"
            boxes.append(
                {
                    "source": "auto",
                    "detection_index": as_int(row["detection_index"]),
                    "confidence": as_float(row.get("auto_confidence")),
                    "action": action,
                    "x1": as_float(row.get(f"{prefix}_x1_px")),
                    "y1": as_float(row.get(f"{prefix}_y1_px")),
                    "x2": as_float(row.get(f"{prefix}_x2_px")),
                    "y2": as_float(row.get(f"{prefix}_y2_px")),
                    "foot_x": (as_float(row.get(f"{prefix}_x1_px")) + as_float(row.get(f"{prefix}_x2_px"))) / 2,
                    "foot_y": as_float(row.get(f"{prefix}_y2_px")),
                    "occluded": row.get("manual_occluded_0_or_1", "") == "1",
                    "truncated": row.get("manual_truncated_0_or_1", "") == "1",
                    "notes": row.get("manual_notes", ""),
                }
            )

        # Previously saved manual-only boxes are those that do not correspond to any
        # active automatic box. This keeps added boxes editable after reopening.
        existing_annotations = [row for row in annotation_rows if row["sample_id"] == sample_id]
        auto_active = [box for box in boxes if box["action"] != "delete"]
        for annotation in existing_annotations:
            annotation_box = [
                as_float(annotation.get("x1_px")),
                as_float(annotation.get("y1_px")),
                as_float(annotation.get("x2_px")),
                as_float(annotation.get("y2_px")),
            ]
            matched_box = next((
                box for box in auto_active
                if max(abs(annotation_box[index] - [box["x1"], box["y1"], box["x2"], box["y2"]][index]) for index in range(4)) < 1.0
            ), None)
            if matched_box is not None:
                matched_box["foot_x"] = as_float(annotation.get("foot_x_px"), matched_box["foot_x"])
                matched_box["foot_y"] = as_float(annotation.get("foot_y_px"), matched_box["foot_y"])
            else:
                boxes.append(
                    {
                        "source": "manual",
                        "detection_index": None,
                        "confidence": None,
                        "action": "keep",
                        "x1": annotation_box[0],
                        "y1": annotation_box[1],
                        "x2": annotation_box[2],
                        "y2": annotation_box[3],
                        "foot_x": as_float(annotation.get("foot_x_px"), (annotation_box[0] + annotation_box[2]) / 2),
                        "foot_y": as_float(annotation.get("foot_y_px"), annotation_box[3]),
                        "occluded": annotation.get("occluded_0_or_1", "") == "1",
                        "truncated": annotation.get("truncated_0_or_1", "") == "1",
                        "notes": annotation.get("notes", ""),
                    }
                )

        return {
            "frame": frame_row,
            "timeseries": next((row for row in timeseries_rows if row["sample_id"] == sample_id), {}),
            "image_url": f"/images/{Path(frame_row['image_path']).name}",
            "boxes": boxes,
        }

    @app.post("/api/frames/{sample_id}")
    def save_frame(sample_id: str, payload: dict) -> dict:
        boxes = payload.get("boxes")
        if not isinstance(boxes, list):
            raise HTTPException(status_code=400, detail="boxes must be a list")
        status = str(payload.get("manual_review_status", "")).strip()
        if status not in {"approved", "corrected", "rejected"}:
            raise HTTPException(status_code=400, detail="Invalid review status")

        requested_count = payload.get("manual_ground_truth_count")
        if status != "rejected":
            if isinstance(requested_count, bool):
                raise HTTPException(status_code=400, detail="Actual count must be a non-negative integer")
            try:
                requested_count = int(requested_count)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="Actual count must be a non-negative integer")
            if requested_count < 0:
                raise HTTPException(status_code=400, detail="Actual count must be a non-negative integer")

        with lock:
            frame_fields, frame_rows = read_csv(frame_path)
            detection_fields, detection_rows = read_csv(detection_path)
            annotation_fields, annotation_rows = read_csv(annotation_path)
            timeseries_fields, timeseries_rows = read_csv(timeseries_path)
            frame_row = next((row for row in frame_rows if row["sample_id"] == sample_id), None)
            if frame_row is None:
                raise HTTPException(status_code=404, detail="Frame not found")

            auto_payload = {
                as_int(box.get("detection_index")): box
                for box in boxes
                if box.get("source") == "auto"
            }
            for row in detection_rows:
                if row["sample_id"] != sample_id:
                    continue
                detection_index = as_int(row["detection_index"])
                box = auto_payload.get(detection_index)
                if box is None:
                    raise HTTPException(status_code=400, detail=f"Missing auto box {detection_index}")
                action = str(box.get("action", "keep"))
                if action not in {"keep", "delete", "edit"}:
                    raise HTTPException(status_code=400, detail=f"Invalid action: {action}")
                row["manual_action_keep_delete_edit"] = action
                for coordinate in ("x1", "y1", "x2", "y2"):
                    field = f"manual_{coordinate}_px_if_edited"
                    row[field] = round(as_float(box.get(coordinate)), 2) if action == "edit" else ""
                row["manual_occluded_0_or_1"] = "1" if box.get("occluded") else "0"
                row["manual_truncated_0_or_1"] = "1" if box.get("truncated") else "0"
                row["manual_notes"] = str(box.get("notes", ""))

            active_boxes = [] if status == "rejected" else [box for box in boxes if box.get("action") != "delete"]
            if status != "rejected" and requested_count != len(active_boxes):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Actual count ({requested_count}) and active box count "
                        f"({len(active_boxes)}) do not match"
                    ),
                )
            frame_row["manual_ground_truth_count"] = str(requested_count) if status != "rejected" else ""
            frame_row["manual_review_status"] = status
            if "manual_box_review_status" not in frame_fields:
                frame_fields.append("manual_box_review_status")
            frame_row["manual_box_review_status"] = "completed"
            frame_row["manual_scene_condition"] = str(payload.get("manual_scene_condition", ""))
            frame_row["manual_notes"] = str(payload.get("manual_notes", ""))

            annotation_rows = [row for row in annotation_rows if row["sample_id"] != sample_id]
            for person_index, box in enumerate(active_boxes, start=1):
                x1, y1 = as_float(box.get("x1")), as_float(box.get("y1"))
                x2, y2 = as_float(box.get("x2")), as_float(box.get("y2"))
                if x2 <= x1 or y2 <= y1:
                    raise HTTPException(status_code=400, detail="Every active box must have positive area")
                annotation_rows.append(
                    {
                        "sample_id": sample_id,
                        "person_index": person_index,
                        "class_id": 0,
                        "x1_px": round(x1, 2),
                        "y1_px": round(y1, 2),
                        "x2_px": round(x2, 2),
                        "y2_px": round(y2, 2),
                        "foot_x_px": round(as_float(box.get("foot_x"), (x1 + x2) / 2), 2),
                        "foot_y_px": round(as_float(box.get("foot_y"), y2), 2),
                        "ground_x_m_optional": "",
                        "ground_y_m_optional": "",
                        "occluded_0_or_1": "1" if box.get("occluded") else "0",
                        "truncated_0_or_1": "1" if box.get("truncated") else "0",
                        "ignore_0_or_1": "0",
                        "notes": str(box.get("notes", "")),
                    }
                )

            for row in timeseries_rows:
                if row["sample_id"] == sample_id:
                    row["manual_ground_truth_count"] = frame_row["manual_ground_truth_count"]
                    row["timestamp_utc_if_known"] = str(payload.get("timestamp_utc_if_known", ""))
                    row["camera_id"] = str(payload.get("camera_id", row.get("camera_id", "")))
                    row["zone_id"] = str(payload.get("zone_id", row.get("zone_id", "")))
                    row["manual_sequence_is_continuous_0_or_1"] = (
                        "1" if payload.get("sequence_is_continuous") else "0"
                    )
                    row["manual_event_or_scene_context"] = str(payload.get("event_or_scene_context", ""))
                    row["calibration_version_if_available"] = (
                        json.loads(calibration_path.read_text(encoding="utf-8")).get("version", "")
                        if calibration_path.exists() else ""
                    )

            write_csv(frame_path, frame_fields, frame_rows)
            write_csv(detection_path, detection_fields, detection_rows)
            write_csv(annotation_path, annotation_fields, annotation_rows)
            write_csv(timeseries_path, timeseries_fields, timeseries_rows)

        return {
            "ok": True,
            "sample_id": sample_id,
            "ground_truth_count": len(active_boxes),
            "status": status,
        }

    def review_status() -> dict:
        _, frame_rows = read_csv(frame_path)
        _, annotation_rows = read_csv(annotation_path)
        _, timeseries_rows = read_csv(timeseries_path)
        annotation_counts: dict[str, int] = {}
        for row in annotation_rows:
            if row.get("ignore_0_or_1") != "1":
                annotation_counts[row["sample_id"]] = annotation_counts.get(row["sample_id"], 0) + 1
        completed = [row for row in frame_rows if row.get("manual_box_review_status") == "completed"]
        accepted = [row for row in completed if row.get("manual_review_status") != "rejected"]
        mismatches = [
            row["sample_id"]
            for row in accepted
            if as_int(row.get("manual_ground_truth_count"), -1) != annotation_counts.get(row["sample_id"], 0)
        ]
        calibration = None
        calibration_error = ""
        if calibration_path.exists():
            try:
                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                calibration_matrix(calibration)
            except (ValueError, json.JSONDecodeError) as error:
                calibration_error = str(error)
        exports = []
        if export_root.exists():
            exports = sorted(
                [path.name for path in export_root.iterdir() if path.is_dir()], reverse=True
            )[:5]
        accepted_times = sorted(as_float(row.get("video_time_seconds")) for row in accepted)
        reviewed_span = accepted_times[-1] - accepted_times[0] if len(accepted_times) >= 2 else 0.0
        timeseries_by_id = {row["sample_id"]: row for row in timeseries_rows}
        continuous_accepted = sum(
            timeseries_by_id.get(row["sample_id"], {}).get("manual_sequence_is_continuous_0_or_1") == "1"
            for row in accepted
        )
        calibration_ready = calibration is not None and not calibration_error
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        return {
            "dataset_path": str(dataset_dir),
            "source_video": metadata.get("source_video", ""),
            "sample_fps": metadata.get("sample_fps"),
            "total_frames": len(frame_rows),
            "box_reviewed_frames": len(completed),
            "accepted_frames": len(accepted),
            "rejected_frames": sum(row.get("manual_review_status") == "rejected" for row in completed),
            "count_annotation_mismatches": mismatches,
            "calibration_ready": calibration_ready,
            "calibration_error": calibration_error,
            "calibration": calibration,
            "recent_exports": exports,
            "reviewed_time_span_seconds": round(reviewed_span, 3),
            "continuous_reviewed_frames": continuous_accepted,
            "forecast_5min_source_ready": reviewed_span >= 300 and continuous_accepted >= 2,
            "ready_to_export": bool(accepted) and not mismatches,
        }

    @app.get("/api/status")
    def status() -> dict:
        with lock:
            return review_status()

    @app.get("/api/calibration")
    def get_calibration() -> dict:
        if not calibration_path.exists():
            return {"calibration": None}
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration_matrix(calibration)
        except (ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=400, detail=f"Invalid saved calibration: {error}")
        return {"calibration": calibration}

    @app.post("/api/calibration")
    def save_calibration(payload: dict) -> dict:
        points = payload.get("image_points")
        if not isinstance(points, list) or len(points) != 4:
            raise HTTPException(status_code=400, detail="Exactly four image points are required")
        normalized_points = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise HTTPException(status_code=400, detail="Each image point must be [x, y]")
            normalized_points.append([round(as_float(point[0]), 2), round(as_float(point[1]), 2)])
        calibration = {
            "version": str(payload.get("version") or datetime.now().strftime("cal-%Y%m%d-%H%M%S")),
            "image_points": normalized_points,
            "point_order": "top_left,top_right,bottom_right,bottom_left",
            "width_m": as_float(payload.get("width_m")),
            "height_m": as_float(payload.get("height_m")),
            "grid_size_m": 1.0,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        try:
            matrix = calibration_matrix(calibration)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            raise HTTPException(status_code=400, detail="Calibration points form an invalid quadrilateral")
        with lock:
            atomic_json(calibration_path, calibration)
        return {"ok": True, "calibration": calibration}

    @app.post("/api/export")
    def export_training_data(payload: dict | None = None) -> dict:
        payload = payload or {}
        include_metric_density = payload.get("include_metric_density") is True
        with lock:
            status_value = review_status()
            if not status_value["ready_to_export"]:
                raise HTTPException(
                    status_code=400,
                    detail="At least one completed frame is required and count/annotation mismatches must be fixed",
                )
            if include_metric_density and not status_value["calibration_ready"]:
                raise HTTPException(
                    status_code=400,
                    detail="A valid measured spatial calibration is required for metric density export",
                )
            _, frame_rows = read_csv(frame_path)
            _, annotation_rows = read_csv(annotation_path)
            _, timeseries_rows = read_csv(timeseries_path)
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            calibration = status_value["calibration"] if include_metric_density else None

            accepted = [
                row for row in frame_rows
                if row.get("manual_box_review_status") == "completed"
                and row.get("manual_review_status") != "rejected"
            ]
            accepted.sort(key=lambda row: as_float(row.get("video_time_seconds")))
            sample_ids = {row["sample_id"] for row in accepted}
            annotations_by_sample: dict[str, list[dict[str, str]]] = {sample_id: [] for sample_id in sample_ids}
            for row in annotation_rows:
                if row["sample_id"] in sample_ids and row.get("ignore_0_or_1") != "1":
                    annotations_by_sample[row["sample_id"]].append(row)

            total = len(accepted)
            train_end = max(1, int(total * 0.70))
            validation_count = int(total * 0.15)
            if total >= 3:
                validation_count = max(1, validation_count)
            validation_end = min(total, train_end + validation_count)
            split_rows = {
                "train": accepted[:train_end],
                "val": accepted[train_end:validation_end],
                "test": accepted[validation_end:],
            }

            export_id = datetime.now().strftime("export_%Y%m%d_%H%M%S_%f")
            export_dir = export_root / export_id
            export_dir.mkdir(parents=True, exist_ok=False)
            image_width = as_int(metadata.get("source_width"))
            image_height = as_int(metadata.get("source_height"))
            if image_width <= 0 or image_height <= 0:
                raise HTTPException(status_code=400, detail="Invalid source image dimensions in metadata")

            for split, rows in split_rows.items():
                split_image_dir = export_dir / "images" / split
                split_label_dir = export_dir / "labels" / split
                split_image_dir.mkdir(parents=True)
                split_label_dir.mkdir(parents=True)
                for frame_row in rows:
                    sample_id = frame_row["sample_id"]
                    source_image = (dataset_dir / frame_row["image_path"]).resolve()
                    destination_image = split_image_dir / source_image.name
                    hardlink_or_copy(source_image, destination_image)
                    yolo_lines = []
                    for annotation in annotations_by_sample[sample_id]:
                        x1, y1 = as_float(annotation["x1_px"]), as_float(annotation["y1_px"])
                        x2, y2 = as_float(annotation["x2_px"]), as_float(annotation["y2_px"])
                        centre_x = ((x1 + x2) / 2) / image_width
                        centre_y = ((y1 + y2) / 2) / image_height
                        width = (x2 - x1) / image_width
                        height = (y2 - y1) / image_height
                        yolo_lines.append(f"0 {centre_x:.6f} {centre_y:.6f} {width:.6f} {height:.6f}")
                    (split_label_dir / f"{sample_id}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

            (export_dir / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n",
                encoding="utf-8",
            )

            person_ground_rows: list[dict[str, object]] = []
            density_rows: list[dict[str, object]] = []
            density_summary: dict[str, dict[str, float]] = {}
            for frame_row in accepted:
                sample_id = frame_row["sample_id"]
                for annotation in annotations_by_sample[sample_id]:
                    person_ground_rows.append({
                        "sample_id": sample_id,
                        "person_index": annotation["person_index"],
                        "foot_x_px": round(as_float(annotation.get("foot_x_px")), 2),
                        "foot_y_px": round(as_float(annotation.get("foot_y_px")), 2),
                        "ground_x_m": "",
                        "ground_y_m": "",
                        "inside_roi_0_or_1": "",
                    })
            if calibration:
                matrix = calibration_matrix(calibration)
                width_m = as_float(calibration["width_m"])
                height_m = as_float(calibration["height_m"])
                columns, rows_count = math.ceil(width_m), math.ceil(height_m)
                for frame_row in accepted:
                    sample_id = frame_row["sample_id"]
                    cell_counts: dict[tuple[int, int], int] = {}
                    roi_count = 0
                    for annotation in annotations_by_sample[sample_id]:
                        foot_x = as_float(annotation.get("foot_x_px"))
                        foot_y = as_float(annotation.get("foot_y_px"))
                        x_m, y_m = ground_point(matrix, foot_x, foot_y)
                        inside = -1e-4 <= x_m <= width_m + 1e-4 and -1e-4 <= y_m <= height_m + 1e-4
                        if inside:
                            x_m = min(max(x_m, 0.0), max(0.0, width_m - 1e-6))
                            y_m = min(max(y_m, 0.0), max(0.0, height_m - 1e-6))
                            cell = (min(int(x_m), columns - 1), min(int(y_m), rows_count - 1))
                            cell_counts[cell] = cell_counts.get(cell, 0) + 1
                            roi_count += 1
                        person_row = next(
                            row for row in person_ground_rows
                            if row["sample_id"] == sample_id
                            and str(row["person_index"]) == str(annotation["person_index"])
                        )
                        person_row["ground_x_m"] = round(x_m, 4)
                        person_row["ground_y_m"] = round(y_m, 4)
                        person_row["inside_roi_0_or_1"] = "1" if inside else "0"
                    max_density = 0.0
                    for row_index in range(rows_count):
                        for column_index in range(columns):
                            x_min, y_min = float(column_index), float(row_index)
                            x_max, y_max = min(x_min + 1.0, width_m), min(y_min + 1.0, height_m)
                            area = (x_max - x_min) * (y_max - y_min)
                            count = cell_counts.get((column_index, row_index), 0)
                            density = count / area if area > 0 else 0.0
                            max_density = max(max_density, density)
                            density_rows.append({
                                "sample_id": sample_id, "grid_row": row_index, "grid_column": column_index,
                                "x_min_m": x_min, "y_min_m": y_min, "x_max_m": round(x_max, 4),
                                "y_max_m": round(y_max, 4), "cell_area_m2": round(area, 4),
                                "person_count": count, "people_per_m2": round(density, 4),
                            })
                    density_summary[sample_id] = {"roi_count": roi_count, "max_density": max_density}

            def output_csv(name: str, fields: list[str], rows: list[dict[str, object]]) -> None:
                write_csv(export_dir / name, fields, rows)

            output_csv(
                "person_ground_coordinates.csv",
                ["sample_id", "person_index", "foot_x_px", "foot_y_px", "ground_x_m", "ground_y_m", "inside_roi_0_or_1"],
                person_ground_rows,
            )
            output_csv(
                "density_grid_1m.csv",
                ["sample_id", "grid_row", "grid_column", "x_min_m", "y_min_m", "x_max_m", "y_max_m", "cell_area_m2", "person_count", "people_per_m2"],
                density_rows,
            )

            timeseries_by_id = {row["sample_id"]: row for row in timeseries_rows}
            reviewed_timeseries = []
            for frame_row in accepted:
                sample_id = frame_row["sample_id"]
                original = timeseries_by_id.get(sample_id, {})
                summary = density_summary.get(sample_id, {})
                reviewed_timeseries.append({
                    "sample_id": sample_id,
                    "video_time_seconds": frame_row["video_time_seconds"],
                    "timestamp_utc_if_known": original.get("timestamp_utc_if_known", ""),
                    "camera_id": original.get("camera_id", ""),
                    "zone_id": original.get("zone_id", ""),
                    "ground_truth_count": frame_row["manual_ground_truth_count"],
                    "roi_ground_truth_count": summary.get("roi_count", ""),
                    "max_grid_density_people_per_m2": round(summary["max_density"], 4) if summary else "",
                    "sequence_is_continuous_0_or_1": original.get("manual_sequence_is_continuous_0_or_1", ""),
                    "event_or_scene_context": original.get("manual_event_or_scene_context", ""),
                    "calibration_version": calibration.get("version", "") if calibration else "",
                })
            timeseries_fields_out = [
                "sample_id", "video_time_seconds", "timestamp_utc_if_known", "camera_id", "zone_id",
                "ground_truth_count", "roi_ground_truth_count", "max_grid_density_people_per_m2",
                "sequence_is_continuous_0_or_1", "event_or_scene_context", "calibration_version",
            ]
            output_csv("reviewed_timeseries.csv", timeseries_fields_out, reviewed_timeseries)

            forecast_rows = []
            if len(reviewed_timeseries) >= 2:
                intervals = [
                    as_float(reviewed_timeseries[i + 1]["video_time_seconds"]) - as_float(reviewed_timeseries[i]["video_time_seconds"])
                    for i in range(len(reviewed_timeseries) - 1)
                ]
                positive_intervals = sorted(value for value in intervals if value > 0)
                tolerance = max(0.51, positive_intervals[len(positive_intervals) // 2] * 0.6) if positive_intervals else 0.51
                for current_index, current_row in enumerate(reviewed_timeseries):
                    target_time = as_float(current_row["video_time_seconds"]) + 300.0
                    candidates = reviewed_timeseries[current_index + 1:]
                    if not candidates:
                        continue
                    target_row = min(candidates, key=lambda row: abs(as_float(row["video_time_seconds"]) - target_time))
                    target_index = reviewed_timeseries.index(target_row)
                    continuous = all(
                        row.get("sequence_is_continuous_0_or_1") == "1"
                        for row in reviewed_timeseries[current_index:target_index + 1]
                    )
                    if continuous and abs(as_float(target_row["video_time_seconds"]) - target_time) <= tolerance:
                        forecast_rows.append({
                            "sample_id": current_row["sample_id"],
                            "video_time_seconds": current_row["video_time_seconds"],
                            "current_count": current_row["ground_truth_count"],
                            "current_roi_count": current_row["roi_ground_truth_count"],
                            "current_max_density": current_row["max_grid_density_people_per_m2"],
                            "target_sample_id_5min": target_row["sample_id"],
                            "target_time_seconds_5min": target_row["video_time_seconds"],
                            "target_count_5min": target_row["ground_truth_count"],
                            "target_roi_count_5min": target_row["roi_ground_truth_count"],
                            "target_max_density_5min": target_row["max_grid_density_people_per_m2"],
                        })
            forecast_fields = [
                "sample_id", "video_time_seconds", "current_count", "current_roi_count", "current_max_density",
                "target_sample_id_5min", "target_time_seconds_5min", "target_count_5min",
                "target_roi_count_5min", "target_max_density_5min",
            ]
            output_csv("forecast_5min.csv", forecast_fields, forecast_rows)

            report = {
                "export_id": export_id,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "accepted_frames": total,
                "split_counts": {key: len(value) for key, value in split_rows.items()},
                "person_annotations": sum(len(value) for value in annotations_by_sample.values()),
                "calibration_included": bool(calibration),
                "metric_density_available": bool(calibration),
                "density_grid_rows": len(density_rows),
                "forecast_5min_pairs": len(forecast_rows),
                "fixed_density_grid_m": 1.0,
                "variable_density_enabled": False,
                "warnings": (
                    (["One or more train/val/test splits are empty"] if any(not rows for rows in split_rows.values()) else [])
                    + (["No valid continuous five-minute forecast pairs were generated"] if not forecast_rows else [])
                    + (["Metric density was intentionally omitted because measured calibration was not selected"] if not calibration else [])
                ),
            }
            atomic_json(export_dir / "quality_report.json", report)
            if calibration:
                atomic_json(export_dir / "calibration.json", calibration)

        return {"ok": True, "export_id": export_id, "export_path": str(export_dir), "report": report}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Lambs box-review UI")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    uvicorn.run(create_app(Path(arguments.dataset)), host=arguments.host, port=arguments.port)
