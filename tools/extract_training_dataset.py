from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean

import cv2


FRAME_FIELDS = [
    "sample_id",
    "image_path",
    "source_frame_index",
    "video_time_seconds",
    "auto_detected_count",
    "auto_confidence_mean",
    "manual_ground_truth_count",
    "manual_review_status",
    "manual_scene_condition",
    "manual_notes",
]

DETECTION_FIELDS = [
    "sample_id",
    "detection_index",
    "auto_confidence",
    "auto_x1_px",
    "auto_y1_px",
    "auto_x2_px",
    "auto_y2_px",
    "manual_action_keep_delete_edit",
    "manual_x1_px_if_edited",
    "manual_y1_px_if_edited",
    "manual_x2_px_if_edited",
    "manual_y2_px_if_edited",
    "manual_occluded_0_or_1",
    "manual_truncated_0_or_1",
    "manual_notes",
]

ANNOTATION_FIELDS = [
    "sample_id",
    "person_index",
    "class_id",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "foot_x_px",
    "foot_y_px",
    "ground_x_m_optional",
    "ground_y_m_optional",
    "occluded_0_or_1",
    "truncated_0_or_1",
    "ignore_0_or_1",
    "notes",
]

TIMESERIES_FIELDS = [
    "sample_id",
    "video_time_seconds",
    "timestamp_utc_if_known",
    "camera_id",
    "zone_id",
    "auto_detected_count",
    "manual_ground_truth_count",
    "manual_roi_ground_truth_count",
    "manual_sequence_is_continuous_0_or_1",
    "manual_event_or_scene_context",
    "calibration_version_if_available",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract reviewable object-detection and time-series training data."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument("--zone-id", default="zone-a")
    return parser.parse_args()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_frames(video_path: Path, image_dir: Path, sample_fps: float) -> tuple[dict, list[dict]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or sample_fps <= 0:
        raise ValueError("Source FPS and sample FPS must be positive")

    interval = source_fps / sample_fps
    next_sample = 0.0
    frame_index = 0
    samples: list[dict] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index + 1e-9 >= next_sample:
            time_seconds = frame_index / source_fps
            sample_id = f"frame_{len(samples):06d}_t{round(time_seconds * 1000):09d}ms"
            image_name = f"{sample_id}.jpg"
            if not cv2.imwrite(str(image_dir / image_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write frame: {image_name}")
            samples.append(
                {
                    "sample_id": sample_id,
                    "image_name": image_name,
                    "source_frame_index": int(frame_index),
                    "video_time_seconds": round(float(time_seconds), 3),
                }
            )
            next_sample += interval
        frame_index += 1

    capture.release()
    metadata = {
        "source_video": str(video_path.resolve()),
        "source_fps": source_fps,
        "source_frame_count": total_frames,
        "duration_seconds": round(total_frames / source_fps, 3),
        "source_width": width,
        "source_height": height,
        "sample_fps": sample_fps,
        "sample_count": len(samples),
    }
    return metadata, samples


def run_detection(
    samples: list[dict],
    image_dir: Path,
    auto_label_dir: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
    batch_size: int,
) -> tuple[list[dict], dict[str, dict]]:
    os.environ.setdefault("YOLO_CONFIG_DIR", str((auto_label_dir.parent / ".ultralytics").resolve()))
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detection_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        paths = [str(image_dir / item["image_name"]) for item in batch]
        results = model.predict(
            source=paths,
            classes=[0],
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )
        for sample, result in zip(batch, results):
            height, width = result.orig_shape
            confidences: list[float] = []
            yolo_lines: list[str] = []
            boxes = result.boxes
            if boxes is not None:
                coordinates = boxes.xyxy.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                for detection_index, (coords, score) in enumerate(zip(coordinates, scores), start=1):
                    x1, y1, x2, y2 = (float(value) for value in coords)
                    score = float(score)
                    confidences.append(score)
                    detection_rows.append(
                        {
                            "sample_id": sample["sample_id"],
                            "detection_index": detection_index,
                            "auto_confidence": round(score, 6),
                            "auto_x1_px": round(x1, 2),
                            "auto_y1_px": round(y1, 2),
                            "auto_x2_px": round(x2, 2),
                            "auto_y2_px": round(y2, 2),
                        }
                    )
                    centre_x = ((x1 + x2) / 2) / width
                    centre_y = ((y1 + y2) / 2) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    yolo_lines.append(
                        f"0 {centre_x:.6f} {centre_y:.6f} {box_width:.6f} {box_height:.6f}"
                    )
            (auto_label_dir / f"{sample['sample_id']}.txt").write_text(
                "\n".join(yolo_lines), encoding="utf-8"
            )
            summaries[sample["sample_id"]] = {
                "count": len(confidences),
                "confidence_mean": round(mean(confidences), 6) if confidences else "",
            }
        print(f"Detected {min(start + batch_size, len(samples))}/{len(samples)} frames", flush=True)
    return detection_rows, summaries


def write_review_guide(path: Path, metadata: dict) -> None:
    forecast_note = (
        "이 영상은 5분보다 짧아서 5분 뒤 예측 정답 쌍을 만들 수 없습니다. "
        if metadata["duration_seconds"] < 300
        else "5분 간격의 예측 정답 쌍을 만들 수 있습니다. "
    )
    path.write_text(
        f"""# 수동 검수 안내

추출 프레임: {metadata['sample_count']}장, 영상 길이: {metadata['duration_seconds']}초

## 반드시 채울 파일

### `frame_review.csv`

- `manual_ground_truth_count`: 프레임에 실제로 보이는 사람의 총수
- `manual_review_status`: `approved`, `corrected`, `rejected` 중 하나
- `manual_scene_condition`: 예: `sparse`, `crowded`, `occluded`, `empty`, `blurred`
- `manual_notes`: 판단이 어려운 이유나 특이사항. 없으면 공란 가능

### `auto_detection_review.csv`

자동 검출 박스마다 다음 값을 확인합니다.

- `manual_action_keep_delete_edit`: 정확하면 `keep`, 오탐이면 `delete`, 위치가 틀리면 `edit`
- `manual_x1_px_if_edited` ~ `manual_y2_px_if_edited`: `edit`일 때만 수정 박스 좌표
- `manual_occluded_0_or_1`: 다른 물체에 가려졌으면 `1`
- `manual_truncated_0_or_1`: 화면 밖으로 잘렸으면 `1`
- `manual_notes`: 선택 사항

자동 검출이 놓친 사람은 이 표에 억지로 추가하지 말고 `manual_annotations.csv`에 추가합니다.

### `manual_annotations.csv`

최종 정답용 표입니다. 프레임에 실제로 존재하는 사람마다 한 행을 추가합니다.

- `sample_id`: `frame_review.csv`와 동일한 프레임 ID
- `person_index`: 프레임 안에서 1부터 시작하는 일련번호
- `class_id`: 사람은 `0`
- `x1_px, y1_px, x2_px, y2_px`: 실제 사람 Bounding Box
- `foot_x_px, foot_y_px`: 박스 하단 중앙에 해당하는 실제 발 위치
- `ground_x_m_optional, ground_y_m_optional`: 실측 Bird-eye 기준점이 있을 때만 미터 좌표
- `occluded_0_or_1`, `truncated_0_or_1`: 가림/잘림 여부
- `ignore_0_or_1`: 판단 불가능하여 학습·평가에서 제외하려면 `1`

`manual_annotations.csv`의 한 프레임 사람 행 개수와 `manual_ground_truth_count`가 일치해야 합니다.

### `crowd_timeseries_review.csv`

- `timestamp_utc_if_known`: 실제 촬영 시작 시각을 아는 경우에만 UTC 시각 입력
- `manual_ground_truth_count`: 같은 프레임의 실제 총인원
- `manual_roi_ground_truth_count`: 설정한 ROI 내부의 실제 인원
- `manual_sequence_is_continuous_0_or_1`: 장면이 편집 없이 시간순으로 이어지면 `1`
- `manual_event_or_scene_context`: 배치 변경, 입장, 퇴장, 장면 전환 등의 설명
- `calibration_version_if_available`: 적용한 ROI/호모그래피 보정 버전

{forecast_note}현재 파일은 객체 인식 검수와 짧은 시계열 분석용입니다.

## 사용 금지 항목

`labels_auto_yolo`는 기존 모델이 만든 후보 라벨입니다. 검수 없이 정답 라벨로 학습하면 안 됩니다.
최종 학습용 YOLO 라벨은 `manual_annotations.csv` 검수가 끝난 뒤 변환해야 합니다.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    model_path = Path(args.model)
    output_dir = Path(args.output)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new dataset directory so existing reviews are not overwritten."
        )
    image_dir = output_dir / "images"
    auto_label_dir = output_dir / "labels_auto_yolo"
    image_dir.mkdir(parents=True, exist_ok=True)
    auto_label_dir.mkdir(parents=True, exist_ok=True)

    metadata, samples = extract_frames(video_path, image_dir, args.sample_fps)
    detection_rows, summaries = run_detection(
        samples,
        image_dir,
        auto_label_dir,
        model_path,
        args.confidence,
        args.image_size,
        args.batch_size,
    )

    frame_rows: list[dict] = []
    timeseries_rows: list[dict] = []
    for sample in samples:
        summary = summaries[sample["sample_id"]]
        frame_rows.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": f"images/{sample['image_name']}",
                "source_frame_index": sample["source_frame_index"],
                "video_time_seconds": sample["video_time_seconds"],
                "auto_detected_count": summary["count"],
                "auto_confidence_mean": summary["confidence_mean"],
            }
        )
        timeseries_rows.append(
            {
                "sample_id": sample["sample_id"],
                "video_time_seconds": sample["video_time_seconds"],
                "camera_id": args.camera_id,
                "zone_id": args.zone_id,
                "auto_detected_count": summary["count"],
            }
        )

    metadata.update(
        {
            "detector_model": str(model_path.resolve()),
            "detector_confidence": args.confidence,
            "detector_image_size": args.image_size,
            "auto_detection_count": len(detection_rows),
            "five_minute_forecast_pairs_available": metadata["duration_seconds"] >= 300,
        }
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "frame_review.csv", FRAME_FIELDS, frame_rows)
    write_csv(output_dir / "auto_detection_review.csv", DETECTION_FIELDS, detection_rows)
    write_csv(output_dir / "manual_annotations.csv", ANNOTATION_FIELDS, [])
    write_csv(output_dir / "crowd_timeseries_review.csv", TIMESERIES_FIELDS, timeseries_rows)
    write_review_guide(output_dir / "REVIEW_GUIDE.md", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
