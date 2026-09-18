"""YOLO preprocessing and postprocessing compatible with Python 3.6."""

import cv2
import numpy as np


def letterbox(image, target_width, target_height, color=(114, 114, 114)):
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")

    scale = min(float(target_width) / width, float(target_height) / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal = target_width - resized_width
    vertical = target_height - resized_height
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, left, top


def prepare_input(image, input_shape):
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(
            "Expected a static NCHW input shaped [1, 3, H, W], got {}".format(
                input_shape
            )
        )
    target_height = int(input_shape[2])
    target_width = int(input_shape[3])
    padded, scale, pad_x, pad_y = letterbox(
        image,
        target_width,
        target_height,
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor[None], dtype=np.float32)
    tensor /= 255.0
    transform = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": int(image.shape[1]),
        "original_height": int(image.shape[0]),
    }
    return tensor, transform


def _xywh_to_xyxy(boxes):
    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return converted


def _nms(boxes, scores, iou_threshold):
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        overlap_x1 = np.maximum(x1[current], x1[remaining])
        overlap_y1 = np.maximum(y1[current], y1[remaining])
        overlap_x2 = np.minimum(x2[current], x2[remaining])
        overlap_y2 = np.minimum(y2[current], y2[remaining])
        overlap = np.maximum(0.0, overlap_x2 - overlap_x1) * np.maximum(
            0.0, overlap_y2 - overlap_y1
        )
        union = areas[current] + areas[remaining] - overlap
        iou = np.divide(
            overlap,
            union,
            out=np.zeros_like(overlap),
            where=union > 0,
        )
        order = remaining[iou <= iou_threshold]
    return keep


def decode_yolo_output(output, transform, confidence_threshold=0.35, iou_threshold=0.45):
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported in the first worker")
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError("Expected a 2D or 3D YOLO output, got {}".format(prediction.shape))

    # Ultralytics exports [channels, anchors], while some exporters transpose it.
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.transpose()
    if prediction.shape[1] < 5:
        raise ValueError("Expected at least five YOLO output channels")

    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    selected = (class_ids == 0) & (scores >= float(confidence_threshold))
    boxes = _xywh_to_xyxy(prediction[selected, :4].astype(np.float32, copy=False))
    scores = scores[selected].astype(np.float32, copy=False)

    if boxes.size == 0:
        return []

    scale = float(transform["scale"])
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(transform["pad_x"])) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(transform["pad_y"])) / scale
    width = float(transform["original_width"])
    height = float(transform["original_height"])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, height)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    keep = _nms(boxes, scores, float(iou_threshold))

    detections = []
    for index in keep:
        box = boxes[index]
        detections.append(
            {
                "box": [float(value) for value in box],
                "confidence": float(scores[index]),
                "track_id": None,
            }
        )
    return detections


def draw_detections(image, detections):
    rendered = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection["box"]]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            rendered,
            "person {:.2f}".format(detection["confidence"]),
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered

