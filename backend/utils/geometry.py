import cv2
import numpy as np


def build_homography(src_points: list, dst_points: list):
    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src_points and dst_points must each contain four [x, y] points")
    matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if matrix is None:
        raise ValueError("Could not calculate a homography matrix")
    return matrix


def to_bird_eye(pixel_x: float, pixel_y: float, matrix: np.ndarray):
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(point, matrix)
    x, y = result[0][0]
    return float(x), float(y)


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
