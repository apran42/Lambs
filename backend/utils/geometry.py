from __future__ import annotations

import numpy as np


def build_homography(src_points: list, dst_points: list):
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src_points and dst_points must each contain four [x, y] points")
    rows = []
    values = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    try:
        coefficients = np.linalg.solve(
            np.asarray(rows, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not calculate a homography matrix") from exc
    return np.append(coefficients, 1.0).reshape(3, 3)


def to_bird_eye(pixel_x: float, pixel_y: float, matrix: np.ndarray):
    point = np.asarray([pixel_x, pixel_y, 1.0], dtype=np.float64)
    result = np.asarray(matrix, dtype=np.float64).dot(point)
    if abs(result[2]) < 1e-12:
        raise ValueError("Point cannot be transformed by this homography")
    return float(result[0] / result[2]), float(result[1] / result[2])


def calculate_positions(results, homography: np.ndarray | None = None) -> list[dict]:
    if not results or results[0].boxes is None:
        return []

    result = results[0]
    boxes = result.boxes
    coordinates = boxes.xyxy.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    track_ids = (
        boxes.id.int().cpu().tolist()
        if boxes.id is not None
        else [None] * len(coordinates)
    )
    frame_height, frame_width = result.orig_shape
    min_width, min_height = frame_width * 0.01, frame_height * 0.01
    max_width, max_height = frame_width * 0.95, frame_height * 0.95

    detections: list[dict] = []
    for coords, class_id, confidence, track_id in zip(
        coordinates, classes, confidences, track_ids
    ):
        if int(class_id) != 0:
            continue
        x1, y1, x2, y2 = coords
        x1, x2 = np.clip((x1, x2), 0.0, float(frame_width))
        y1, y2 = np.clip((y1, y2), 0.0, float(frame_height))
        width, height = x2 - x1, y2 - y1
        if width < min_width or height < min_height:
            continue
        if width > max_width or height > max_height:
            continue

        detection = {
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "track_id": track_id,
            "confidence": float(confidence),
        }
        if homography is not None:
            bx, by = to_bird_eye(x1 + width / 2, y2, homography)
            detection["bird_eye"] = [bx, by]
        detections.append(detection)
    return detections
